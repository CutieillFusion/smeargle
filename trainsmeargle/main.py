import argparse
import deepspeed

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--basepath", type=str, required=True)
parser.add_argument("--trainpath", type=str, required=True)
parser.add_argument("--testpath", type=str, required=True)
parser.add_argument("--savedir", type=str, required=True)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="local_rank for distributed training on gpus",
)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

import json
import re

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)

train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": 40,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpoint": True,
}

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from cnets import padding

try:
    from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler
    from deepspeed.runtime.zero.config import ZeroStageEnum

    torch.serialization.add_safe_globals([DynamicLossScaler, ZeroStageEnum])
except ImportError:
    pass  # DeepSpeed not loaded yet, will add later if needed

# Add fragment_address separately as it may not be available in all DeepSpeed versions
# Import the module first to ensure it's loaded, then get the attribute
_fragment_address_registered = False
try:
    import deepspeed.utils.tensor_fragment as tf_module

    if hasattr(tf_module, "fragment_address"):
        fragment_address = tf_module.fragment_address
        torch.serialization.add_safe_globals([fragment_address])
        _fragment_address_registered = True
except (ImportError, AttributeError) as e:
    # Fallback: try direct import
    try:
        from deepspeed.utils.tensor_fragment import fragment_address

        torch.serialization.add_safe_globals([fragment_address])
        _fragment_address_registered = True
    except (ImportError, AttributeError):
        pass  # fragment_address not available, may cause issues with checkpoint loading

torch.backends.cuda.matmul.allow_tf32 = True
from accelerate.utils import set_seed

set_seed(0)
from cnets import Model
from configs import EConfig
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

# import accelerate
import numpy as np
from transformers import PreTrainedTokenizerBase, get_linear_schedule_with_warmup


def build_dataset_rank(tokenizer, datapath):

    ds = load_dataset("json", data_files=datapath)
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    ds1 = ds
    original_columns1 = ds1.column_names
    num_proc = 8

    def preprocess_function(examples):
        new_examples = {"attention_mask": [], "input_ids": [], "loss_mask": []}
        for i in range(len(examples["id"])):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                },
            ]
            convroles = ["user", "assistant"]
            roles = {"human": "user", "gpt": "assistant"}
            source = examples["conversations"][i]
            if not source:
                continue
            if roles[source[0]["from"]] != "user":
                # Skip the first one if it is not from human
                source = source[1:]
            for j, sentence in enumerate(source):
                role = roles[sentence["from"]]
                assert role == convroles[j % 2], f"{i}"
                # if sentence["from"]=="gpt":
                #     sentence["value"]=" "+sentence["value"]
                messages.append({"role": role, "content": sentence["value"]})
            conversation = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.unk_token_id

            input_ids = tokenizer(
                conversation,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            # filtering out the samples which is longer than max_len
            if len(input_ids) > train_config["max_len"]:
                continue
            loss_mask = torch.ones_like(input_ids)
            # print(i)

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            total_len = len(input_ids)

            sep2 = "<|eot_id|><|start_header_id|>user<|end_header_id|>"
            turns = conversation.split(sep2)

            turns[1] = turns[0] + sep2 + turns[1]
            turns = turns[1:]

            cur_len = 1
            loss_mask[:cur_len] = 0
            for i, turn in enumerate(turns):
                if turn == "":
                    break
                turn_len = len(tokenizer(turn).input_ids)

                parts = turn.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

                # Ignore the user instructions
                if i == 0:
                    loss_mask[cur_len : cur_len + instruction_len - 2] = 0
                else:
                    loss_mask[cur_len - 3 : cur_len + instruction_len + 1] = 0
                cur_len += turn_len
                if i != 0:
                    cur_len += 3
                # cur_len+=2

                # if i != 0 and not tokenizer.legacy:
                #     # The legacy and non-legacy modes handle special tokens differently
                #     cur_len -= 1

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            # new_examples["conversation"].append(conversation)
            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds1 = ds1.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns1,
        load_from_cache_file=False,
    )

    ds1.set_format(type="torch")
    return ds1


class DataCollatorWithPadding:

    def paddingtensor(self, intensors, N):
        B, n, S = intensors.shape
        # padding_tensor = torch.zeros(B, N - n, S,dtype=intensors.dtype)
        padding_tensor = torch.zeros(B, N - n, S, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def paddingtensor2D(self, intensors, N):
        B, n = intensors.shape
        padding_tensor = torch.zeros(B, N - n, dtype=intensors.dtype)
        outtensors = torch.cat((intensors, padding_tensor), dim=1)
        return outtensors

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(item["input_ids"].shape[1] for item in features)
        batch_input_ids = torch.cat(
            [self.paddingtensor2D(item["input_ids"], max_length) for item in features]
        )
        batch_attention_mask = torch.cat(
            [
                self.paddingtensor2D(item["attention_mask"], max_length)
                for item in features
            ]
        )
        batch_loss_mask = torch.cat(
            [self.paddingtensor2D(item["loss_mask"], max_length) for item in features]
        )

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "loss_mask": batch_loss_mask,
        }
        return batch


tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

config = EConfig.from_pretrained(train_config["config_path"])
model = Model(
    config, ds_config, train_config, path=args.basepath, load_emb=True, load_head=True
)
model.scandata(args.trainpath, args.basepath)


criterion = nn.SmoothL1Loss(reduction="none")

num_epochs = train_config["num_epochs"]

# Create PyTorch AdamW optimizer manually to bypass DeepSpeed's FusedAdam (which fails on compute_90)
# Extract optimizer params from ds_config
opt_params = ds_config["optimizer"]["params"]
optimizer = optim.AdamW(
    model.parameters(),
    lr=(
        opt_params["lr"]
        if opt_params["lr"] > 0
        else ds_config["scheduler"]["params"]["warmup_max_lr"]
    ),
    betas=tuple(opt_params["betas"]),
    weight_decay=opt_params["weight_decay"],
    eps=1e-8,
)

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    optimizer=optimizer,
    model_parameters=model.parameters(),
)

# Ensure checkpoint engine uses weights_only=False if fragment_address registration failed
if not _fragment_address_registered and hasattr(model_engine, "checkpoint_engine"):
    original_load = model_engine.checkpoint_engine.load

    def patched_checkpoint_load(path, map_location=None):
        return torch.load(path, map_location=map_location, weights_only=False)

    model_engine.checkpoint_engine.load = patched_checkpoint_load

global_rank = deepspeed.comm.get_rank()
rank = deepspeed.comm.get_local_rank()
world_size = deepspeed.comm.get_world_size()
if global_rank == 0:
    import wandb
    import os

    # Disable wandb by default (option 3 = "Don't visualize my results")
    # Set WANDB_MODE=online in environment to enable tracking
    wandb_mode = os.environ.get("WANDB_MODE", "disabled")

    # Only login if wandb is enabled
    if wandb_mode != "disabled":
        wandb.login(key="")

    wandb.init(
        project="SMEARGLE", entity="dylan-norquist", config=ds_config, mode=wandb_mode
    )

args.savedir = f"models/{args.savedir}"

os.makedirs(args.savedir, exist_ok=True)

sampler = DistributedSampler(
    testdataset, num_replicas=world_size, rank=global_rank, shuffle=False
)
test_loader = DataLoader(
    testdataset,
    batch_size=train_config["bs"],
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataCollatorWithPadding(),
)

train_sampler = DistributedSampler(
    traindataset, num_replicas=world_size, rank=global_rank, shuffle=True
)
train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataCollatorWithPadding(),
)


def find_max_state_with_file(directory, filename="zero_to_fp32.py"):
    max_a = -1
    for subdir in os.listdir(directory):
        match = re.match(r"state_(\d+)", subdir)
        if match:
            a_value = int(match.group(1))
            subdir_path = os.path.join(directory, subdir)
            file_path = os.path.join(subdir_path, filename)
            if os.path.isdir(subdir_path) and os.path.exists(file_path):
                max_a = max(max_a, a_value)
    if max_a == -1:
        return None, 0
    return f"{directory}/state_{max_a}", max_a + 1


checkpoint_path, start_epoch = find_max_state_with_file(args.savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    # Ensure fragment_address is registered before loading checkpoint (safety check)
    if not _fragment_address_registered:
        try:
            import deepspeed.utils.tensor_fragment as tf_module

            if hasattr(tf_module, "fragment_address"):
                fragment_address = tf_module.fragment_address
                torch.serialization.add_safe_globals([fragment_address])
                _fragment_address_registered = True
        except (ImportError, AttributeError):
            try:
                from deepspeed.utils.tensor_fragment import fragment_address

                torch.serialization.add_safe_globals([fragment_address])
                _fragment_address_registered = True
            except (ImportError, AttributeError):
                pass
    # Set strict=False to allow missing frozen parameters (like embed_tokens.weight)
    # which are excluded when saving with exclude_frozen_parameters=True
    model_engine.load_checkpoint(checkpoint_path, load_module_strict=False)


for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    # Debugging counters
    skipped_batches = 0
    nan_loss_count = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        try:
            model.zero_grad()

            device = next(model_engine.module.parameters()).device

            # Get loss scale before forward pass
            if hasattr(model_engine, "optimizer") and hasattr(
                model_engine.optimizer, "loss_scaler"
            ):
                current_loss_scale = model_engine.optimizer.loss_scaler.cur_scale

            plosses, acces = model_engine(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                loss_mask=data["loss_mask"].to(device),
            )

            # Check for NaN/Inf in individual losses
            valid_plosses = []
            for i, ploss in enumerate(plosses):
                if not torch.isfinite(ploss):
                    if global_rank == 0:
                        print(
                            f"[WARNING] Batch {batch_idx}, Position {i}: Invalid loss (NaN/Inf): {ploss.item()}"
                        )
                    nan_loss_count += 1
                    # Replace with zero to prevent propagation
                    valid_plosses.append(
                        torch.tensor(0.0, device=ploss.device, dtype=ploss.dtype)
                    )
                else:
                    valid_plosses.append(ploss)
            plosses = valid_plosses

            ploss_weight = [0.8**i for i in range(len(plosses))]
            ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            loss = ploss

            # Check final loss for NaN/Inf before backward
            if not torch.isfinite(loss):
                if global_rank == 0:
                    print(
                        f"[ERROR] Batch {batch_idx}: Final loss is NaN/Inf: {loss.item()}, skipping batch"
                    )
                skipped_batches += 1
                continue

            # Scale down loss if it's too large to prevent gradient explosion
            # Large losses can cause gradients to overflow even with loss scaling
            max_loss_value = 100.0
            loss_abs = loss.abs()
            if loss_abs > max_loss_value:
                if global_rank == 0 and batch_idx % 10 == 0:
                    print(
                        f"[WARNING] Batch {batch_idx}: Loss too large ({loss.item():.2f}), scaling down"
                    )
                loss = loss * (max_loss_value / loss_abs)

            # Log loss scale periodically
            if (
                batch_idx % 10 == 0
                and global_rank == 0
                and hasattr(model_engine, "optimizer")
                and hasattr(model_engine.optimizer, "loss_scaler")
            ):
                loss_scale = model_engine.optimizer.loss_scaler.cur_scale

            model_engine.backward(loss)

            # Try to step optimizer - this is where the loss scale exception is raised
            try:
                model_engine.step()
            except Exception as step_e:
                if "loss scale already at minimum" in str(step_e):
                    if global_rank == 0:
                        print(
                            f"[ERROR] Batch {batch_idx}: Loss scale at minimum, attempting to recover..."
                        )
                    # Try to reset gradients and continue
                    try:
                        model_engine.optimizer.zero_grad()
                    except:
                        pass
                    skipped_batches += 1
                    # Don't re-raise, continue to next batch
                    continue
                else:
                    # Re-raise other exceptions to be caught by outer handler
                    raise

        except Exception as e:
            if global_rank == 0:
                print(f"[ERROR] Batch {batch_idx}: Exception during training: {e}")
                import traceback

                traceback.print_exc()
            skipped_batches += 1
            # Try to reset gradients before continuing
            try:
                model_engine.optimizer.zero_grad()
            except:
                pass
            # Continue to next batch instead of crashing
            continue

        if global_rank == 0:
            logdict = {"train/lr": optimizer.optimizer.param_groups[0]["lr"]}
            for i in range(len(plosses)):
                logdict[f"train/ploss_{i}"] = plosses[i].item()
            for i in range(len(acces)):
                logdict[f"train/acc_{i}"] = acces[i]
            # Add debugging metrics
            if hasattr(model_engine, "optimizer") and hasattr(
                model_engine.optimizer, "loss_scaler"
            ):
                logdict["train/loss_scale"] = (
                    model_engine.optimizer.loss_scaler.cur_scale
                )
            logdict["train/skipped_batches"] = skipped_batches
            logdict["train/nan_loss_count"] = nan_loss_count
            wandb.log(logdict)
        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [
            epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
        ]

    # Log summary at end of epoch
    if global_rank == 0:
        print(
            f"[SUMMARY] Epoch {epoch}: Skipped batches: {skipped_batches}, NaN losses: {nan_loss_count}"
        )
        if hasattr(model_engine, "optimizer") and hasattr(
            model_engine.optimizer, "loss_scaler"
        ):
            print(
                f"[SUMMARY] Final loss scale: {model_engine.optimizer.loss_scaler.cur_scale}"
            )

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochacc_{i}": acc_i})
            print(
                f"Train Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"train/epochploss_{i}": loss_i})
            print(
                f"Train Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for batch_idx, data in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            device = next(model_engine.module.parameters()).device
            plosses, acces = model_engine(
                input_ids=data["input_ids"].to(device),
                attention_mask=data["attention_mask"].to(device),
                loss_mask=data["loss_mask"].to(device),
            )
            epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
            epoch_plosses = [
                epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
            ]

    for i in range(len(epoch_acces)):
        acc_i = torch.tensor(epoch_acces[i]).cuda().mean()
        deepspeed.comm.all_reduce(acc_i, op=deepspeed.comm.ReduceOp.AVG)
        acc_i = acc_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochacc_{i}": acc_i})
            print(
                f"Test Epoch [{epoch + 1}/{num_epochs}], position {i},  Acc: {acc_i:.2f}"
            )

    for i in range(len(epoch_plosses)):
        loss_i = torch.tensor(epoch_plosses[i]).cuda().mean()
        deepspeed.comm.all_reduce(loss_i, op=deepspeed.comm.ReduceOp.AVG)
        loss_i = loss_i.item()
        if global_rank == 0:
            wandb.log({f"test/epochploss_{i}": loss_i})
            print(
                f"Test Epoch [{epoch + 1}/{num_epochs}], position {i}, pLoss: {loss_i:.2f}"
            )
    # clear out the redundance cahce after each step
    torch.cuda.empty_cache()

    model_engine.save_16bit_model(
        f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True
    )
    if epoch % 10 == 0:
        try:
            # Save checkpoint with frozen parameters excluded to avoid tracking issues
            # The target_model is not a registered submodule, so DeepSpeed can't track its frozen params
            model_engine.save_checkpoint(
                save_dir=f"{args.savedir}/state_{epoch}", exclude_frozen_parameters=True
            )
        except ValueError as e:
            if "failed to find frozen" in str(e):
                # If frozen parameter tracking fails, skip checkpoint save but log warning
                # The 16-bit model is already saved above, so training can continue
                if global_rank == 0:
                    print(
                        f"Warning: Could not save DeepSpeed checkpoint due to frozen parameter tracking issue."
                    )
                    print(
                        f"         The 16-bit model has been saved and training will continue."
                    )
                    print(
                        f"         Note: You may need to restart training from the 16-bit model if resuming."
                    )
            else:
                raise

import argparse
import deepspeed
import json
import re
import os
import torch
from cnets import Model
from configs import EagleConfig
from datasets import load_dataset
from typing import Any, Dict, List
from torch import optim
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from deepspeed.runtime.fp16.loss_scaler import DynamicLossScaler, LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from deepspeed.utils.tensor_fragment import fragment_address
from accelerate.utils import set_seed

set_seed(0)
# This makes the model run faster on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
# Add safe globals to prevent issues with checkpoint loading
torch.serialization.add_safe_globals([DynamicLossScaler, ZeroStageEnum, fragment_address, LossScaler])

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
parser.add_argument(
    "--patience",
    type=int,
    default=1,
    help="Early stopping patience based on best test pLoss at position 0. None means no early stopping.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=40,
    help="Number of epochs to train",
)
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

deepspeed_config = args.deepspeed_config
with open(deepspeed_config) as f:
    ds_config = json.load(f)

train_config = {
    "bs": ds_config["train_micro_batch_size_per_gpu"],
    "num_epochs": args.epochs,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpoint": True,
}

def build_dataset_rank(tokenizer, datapath):

    ds = load_dataset("json", data_files=datapath)
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    num_proc = 48

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

            sep = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

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

            loss_mask[cur_len:] = 0
            attention_mask = torch.ones_like(loss_mask)

            new_examples["input_ids"].append(input_ids[None, :])
            new_examples["loss_mask"].append(loss_mask[None, :])
            new_examples["attention_mask"].append(attention_mask[None, :])

        return new_examples

    ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=ds.column_names,
        load_from_cache_file=False,
    )
    
    ds.set_format(type="torch")
    return ds


class DataCollatorWithPadding:

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

config = EagleConfig.from_pretrained(train_config["config_path"])
model = Model(
    config, train_config, path=args.basepath
)
model.scandata(args.trainpath, args.basepath, args.local_rank)

# Load target model before DeepSpeed init so all params are registered (fixes save_checkpoint)
_ = model._target_model

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

global_rank = deepspeed.comm.get_rank()
world_size = deepspeed.comm.get_world_size()

savedir = f"models/{args.savedir}"

os.makedirs(savedir, exist_ok=True)

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


checkpoint_path, start_epoch = find_max_state_with_file(savedir)
if checkpoint_path:
    print(f"load from {checkpoint_path}")
    model_engine.load_checkpoint(checkpoint_path)

def print_rank(message: str):
    if global_rank == 0:
        print(message)

def reduce_and_print(epoch_metrics: list[list[float]], mode: str, metric_name: str, epoch: int) -> float:
    total_metric = 0
    for i, metric in enumerate(epoch_metrics):
        metric = torch.tensor(metric).cuda().mean()
        torch.cuda.empty_cache()
        deepspeed.comm.all_reduce(metric, op=deepspeed.comm.ReduceOp.AVG)
        print_rank(f"{mode} Epoch [{epoch + 1}/{num_epochs}], position {i}, {metric_name}: {metric.item():.2f}")
        total_metric += metric.item()
    return total_metric / len(epoch_metrics)

best_test_ploss = float("inf")
patience_counter = 0
best_epoch = -1

for epoch in range(start_epoch, num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print_rank(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    for data in tqdm(train_loader):
        model.zero_grad()

        device = next(model_engine.module.parameters()).device
        plosses, acces = model_engine(
            input_ids=data["input_ids"].to(device),
            attention_mask=data["attention_mask"].to(device),
            loss_mask=data["loss_mask"].to(device),
        )

        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
        loss = ploss
        model_engine.backward(loss)

        model_engine.step()

        epoch_acces = [epoch_acces[i] + [acces[i]] for i in range(len(acces))]
        epoch_plosses = [
            epoch_plosses[i] + [plosses[i].item()] for i in range(len(plosses))
        ]

    reduce_and_print(epoch_acces, "Train", "Acc", epoch)
    reduce_and_print(epoch_plosses, "Train", "pLoss", epoch)

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    model.eval()
    for data in tqdm(test_loader):
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

    reduce_and_print(epoch_acces, "Test", "Acc", epoch)
    test_ploss = reduce_and_print(epoch_plosses, "Test", "pLoss", epoch)

    # Early stopping based on test pLoss on average test position loss
    if args.patience is not None:
        if test_ploss < best_test_ploss:
            best_test_ploss = test_ploss
            best_epoch = epoch
            patience_counter = 0
            print_rank(f"New best test pLoss: {best_test_ploss:.4f} at epoch {epoch + 1}")
            model_engine.save_16bit_model(f"{savedir}/best_model", exclude_frozen_parameters=True)
        else:
            print_rank(f"No improvement in test pLoss. Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print_rank(f"Early stopping triggered! Best test pLoss: {best_test_ploss:.4f} at epoch {best_epoch + 1}")
                break
            patience_counter += 1

    # clear out the redundance cache after each step
    torch.cuda.empty_cache()
    
    model_engine.save_checkpoint(f"{savedir}/state_{epoch}")
    model_engine.save_16bit_model(
        f"{savedir}/state_{epoch}", exclude_frozen_parameters=True
    )

# Explicit cleanup to prevent leaking resources
deepspeed.comm.barrier()
deepspeed.comm.destroy_process_group()
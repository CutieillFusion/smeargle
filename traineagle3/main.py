import argparse
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from modeling_llama3_eagle3 import Eagle3
from configs import EagleConfig
from datasets import load_dataset
from typing import Any, Dict, List
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer
from accelerate.utils import set_seed

set_seed(0)
# This makes the model run faster on Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True

parser = argparse.ArgumentParser(description="sp")
parser.add_argument("--basepath", type=str, required=True)
parser.add_argument("--trainpath", type=str, required=True)
parser.add_argument("--testpath", type=str, required=True)
parser.add_argument("--savedir", type=str, required=True)
parser.add_argument(
    "--tp_size",
    type=int,
    default=4,
    help="Tensor parallelism degree for the target model",
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
args = parser.parse_args()

# Training hyperparameters (previously in ds_config.json)
BATCH_SIZE = 1
GRAD_ACCUM = 2
GRAD_CLIP = 0.5

train_config = {
    "bs": BATCH_SIZE,
    "num_epochs": args.epochs,
    "num_workers": 2,
    "max_len": 2048,
    "config_path": "config.json",
    "gradient_checkpoint": True,
}


def find_subsequence(seq, pattern):
    """Find all start indices of pattern in seq."""
    indices = []
    plen = len(pattern)
    for i in range(len(seq) - plen + 1):
        if seq[i : i + plen] == pattern:
            indices.append(i)
    return indices


def compute_assistant_loss_mask(input_ids_list, assistant_header_ids, eot_ids):
    """Token-ID based loss mask: only compute loss on assistant response tokens."""
    seq = input_ids_list
    mask = [0] * len(seq)
    header_starts = find_subsequence(seq, assistant_header_ids)
    header_len = len(assistant_header_ids)

    for hs in header_starts:
        response_start = hs + header_len
        # Find next eot_id after the response start
        eot_starts = find_subsequence(seq[response_start:], eot_ids)
        if eot_starts:
            response_end = response_start + eot_starts[0]
        else:
            response_end = len(seq)
        for j in range(response_start, response_end):
            mask[j] = 1

    return mask


def build_dataset_rank(tokenizer, datapath):
    ds = load_dataset("json", data_files=datapath)
    ds = ds["train"]
    ds = ds.shuffle(seed=42)
    num_proc = 48

    # Pre-compute token patterns for loss mask (robust, tokenizer-agnostic)
    assistant_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False
    )
    eot_ids = tokenizer.encode("<|eot_id|>", add_special_tokens=False)

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

            if len(input_ids) > train_config["max_len"]:
                continue

            loss_mask_list = compute_assistant_loss_mask(
                input_ids.tolist(), assistant_header_ids, eot_ids
            )
            loss_mask = torch.tensor(loss_mask_list, dtype=input_ids.dtype)
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


# --- Distributed initialization ---
dist.init_process_group(backend="nccl")
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)

tp_size = args.tp_size
assert world_size % tp_size == 0, (
    f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"
)
dp_size = world_size // tp_size

mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
tp_rank = mesh["tp"].get_local_rank()
dp_rank = mesh["dp"].get_local_rank()
is_draft_rank = tp_rank == 0
dp_group = mesh["dp"].get_group() if dp_size > 1 else None

device = torch.device(f"cuda:{local_rank}")

tokenizer = AutoTokenizer.from_pretrained(args.basepath)
traindataset = build_dataset_rank(tokenizer, args.trainpath)
testdataset = build_dataset_rank(tokenizer, args.testpath)

config = EagleConfig.from_json(train_config["config_path"])
model = Eagle3(config, train_config, path=args.basepath, device_mesh=mesh)
model.scandata(args.trainpath, args.basepath, global_rank)

# Move draft components to the local device (target model is already placed by TP)
for name, param in model.named_parameters():
    if not name.startswith("_target_model") and param.device.type == "cpu":
        param.data = param.data.to(device)
for name, buf in model.named_buffers():
    if not name.startswith("_target_model") and buf.device.type == "cpu":
        buf.data = buf.data.to(device)

num_epochs = train_config["num_epochs"]

# Optimizer and scheduler -- draft rank only
max_lr = 5e-5
if is_draft_rank:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        trainable_params,
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
        eps=1e-8,
    )
else:
    trainable_params = None
    optimizer = None

savedir = f"models/{args.savedir}"
os.makedirs(savedir, exist_ok=True)

# DataLoader uses dp_rank so all TP ranks in a group get the same batch
sampler = DistributedSampler(
    testdataset, num_replicas=dp_size, rank=dp_rank, shuffle=False
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
    traindataset, num_replicas=dp_size, rank=dp_rank, shuffle=True
)
train_loader = DataLoader(
    traindataset,
    batch_size=train_config["bs"],
    sampler=train_sampler,
    num_workers=4,
    pin_memory=True,
    collate_fn=DataCollatorWithPadding(),
)

# LR scheduler (draft rank only)
if is_draft_rank:
    steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_steps = steps_per_epoch * num_epochs
    warmup_ratio = 0.015
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    warmup_scheduler = LinearLR(
        optimizer, start_factor=1e-6 / max_lr, total_iters=warmup_steps
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    )
    lr_scheduler = SequentialLR(
        optimizer, [warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )


def print_rank(message: str):
    if global_rank == 0:
        print(message)


def reduce_and_print(
    epoch_metrics: list[list[float]], mode: str, metric_name: str, epoch: int
) -> list[float]:
    """Reduce metrics across DP group (draft ranks only)."""
    reduced = []
    for i, metric in enumerate(epoch_metrics):
        metric = torch.tensor(metric).cuda().mean()
        if dp_group is not None:
            dist.all_reduce(metric, op=dist.ReduceOp.AVG, group=dp_group)
        print_rank(
            f"{mode} Epoch [{epoch + 1}/{num_epochs}], position {i}, {metric_name}: {metric.item():.2f}"
        )
        reduced.append(metric.item())
    return reduced


def simulated_acceptance_length(acces: list[float]) -> float:
    cumulative = 1.0
    acc_length = 0.0
    for a in acces:
        cumulative *= a
        acc_length += cumulative
    return acc_length


best_test_ploss = float("inf")
patience_counter = 0
best_epoch = -1

for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch + 1)
    print_rank(f"Now training epoch {epoch}")

    model.train()
    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    if is_draft_rank:
        optimizer.zero_grad()

    grad_step = 0

    for data in tqdm(train_loader, disable=(global_rank != 0)):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        loss_mask = data["loss_mask"].to(device)

        hidden_states, target, loss_mask_out, input_ids_out = model.target_forward(
            input_ids, attention_mask, loss_mask
        )

        if is_draft_rank:
            plosses, acces = model.draft_forward(
                hidden_states, target, loss_mask_out, input_ids_out
            )

            ploss_stack = torch.stack(plosses)
            loss = (model.ploss_weights.to(device) * ploss_stack).sum()

            scaled_loss = loss / GRAD_ACCUM
            scaled_loss.backward()

            grad_step += 1
            if grad_step % GRAD_ACCUM == 0:
                if dp_group is not None:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(
                                p.grad, op=dist.ReduceOp.AVG, group=dp_group
                            )

                torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            for i in range(len(acces)):
                epoch_acces[i].append(acces[i])
                epoch_plosses[i].append(plosses[i].item())

    if is_draft_rank:
        train_acces = reduce_and_print(epoch_acces, "Train", "Acc", epoch)
        reduce_and_print(epoch_plosses, "Train", "pLoss", epoch)
        print_rank(
            f"Train Epoch [{epoch + 1}/{num_epochs}], Simulated Acceptance Length: {simulated_acceptance_length(train_acces):.2f}"
        )

    epoch_acces = [[] for _ in range(model.length)]
    epoch_plosses = [[] for _ in range(model.length)]

    model.eval()
    for data in tqdm(test_loader, disable=(global_rank != 0)):
        with torch.no_grad():
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)
            loss_mask = data["loss_mask"].to(device)

            hidden_states, target, loss_mask_out, input_ids_out = model.target_forward(
                input_ids, attention_mask, loss_mask
            )

            if is_draft_rank:
                plosses, acces = model.draft_forward(
                    hidden_states, target, loss_mask_out, input_ids_out
                )
                for i in range(len(acces)):
                    epoch_acces[i].append(acces[i])
                    epoch_plosses[i].append(plosses[i].item())

    should_stop = torch.zeros(1, dtype=torch.long, device=device)
    if is_draft_rank:
        test_acces = reduce_and_print(epoch_acces, "Test", "Acc", epoch)
        test_plosses = reduce_and_print(epoch_plosses, "Test", "pLoss", epoch)
        test_ploss = sum(test_plosses) / len(test_plosses)
        test_acc_length = simulated_acceptance_length(test_acces)
        print_rank(
            f"Test Epoch [{epoch + 1}/{num_epochs}], Simulated Acceptance Length: {test_acc_length:.2f}"
        )

        if args.patience is not None:
            if test_ploss < best_test_ploss:
                best_test_ploss = test_ploss
                best_epoch = epoch
                patience_counter = 0
                print_rank(
                    f"New best test pLoss: {best_test_ploss:.4f} at epoch {epoch + 1}"
                )
            else:
                print_rank(
                    f"No improvement in test pLoss. Patience: {patience_counter}/{args.patience}"
                )

                if patience_counter >= args.patience:
                    print_rank(
                        f"Early stopping triggered! Best test pLoss: {best_test_ploss:.4f} at epoch {best_epoch + 1}"
                    )
                    should_stop.fill_(1)
                patience_counter += 1

    if global_rank == 0:
        trainable_state = {
            k: v for k, v in model.state_dict().items()
            if not k.startswith("_target_model")
        }

        if best_epoch == epoch:
            os.makedirs(f"{savedir}/best_model", exist_ok=True)
            torch.save(trainable_state, f"{savedir}/best_model/draft_model.pt")
            print_rank(f"Saved best model to {savedir}/best_model/")

        checkpoint_dir = f"{savedir}/state_{epoch}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(trainable_state, f"{checkpoint_dir}/draft_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            },
            f"{checkpoint_dir}/training_state.pt",
        )
        print_rank(f"Saved checkpoint to {checkpoint_dir}/")

    # Broadcast stop decision to all ranks so non-draft ranks exit too
    dist.broadcast(should_stop, src=0)
    if should_stop.item():
        break

    torch.cuda.empty_cache()

# Explicit cleanup
dist.barrier()
dist.destroy_process_group()

import argparse
import json
import os
import subprocess
import threading
from accelerate.utils import set_seed

set_seed(0)

import numpy as np
import time
import shortuuid
import torch


class PowerMonitor:
    """Samples GPU power in a background thread to compute energy (Joules)."""

    def __init__(self, gpu_index=0, interval=0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "-i", str(self.gpu_index),
                     "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    stderr=subprocess.DEVNULL,
                ).decode().strip()
                self.samples.append((time.time(), float(out)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self.samples.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()

    def energy_joules(self):
        if len(self.samples) < 2:
            return 0.0
        energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_w = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            energy += avg_w * dt
        return energy


from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from model.eagle_model import EagleModel
from model.utils import prepare_logits_processor


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe.  Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain "
    "why instead of answering something not correct. If you don't know the answer "
    "to a question, please don't share false information."
)


def build_input_ids(tokenizer, messages_or_prompt, use_chat_template: bool):
    """Tokenize either via chat template (instruct) or raw prompt (base model)."""
    if use_chat_template:
        prompt = tokenizer.apply_chat_template(
            messages_or_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_ids = tokenizer([prompt], add_special_tokens=False).input_ids
    else:
        # For base model: messages_or_prompt is expected to be a raw string.
        input_ids = tokenizer([messages_or_prompt], add_special_tokens=True).input_ids
    return input_ids


def maybe_truncate_on_newline(output: str, stop_on_newline: bool) -> str:
    if not stop_on_newline:
        return output
    idx = output.find("\n\n")
    if idx >= 0:
        return output[:idx]
    return output


def run_eval(
    base_model_path,
    eagle3_model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    temperature,
    total_token,
    depth,
    top_k,
    warmup_steps,
    use_eagle3,
    draft_kv_window=None,
    use_chat_template=True,
    stop_on_newline=False,
    max_prompt_tokens=None,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    [
        get_model_answers(
            base_model_path,
            eagle3_model_path,
            total_token,
            depth,
            top_k,
            warmup_steps,
            use_eagle3,
            questions[i : i + chunk_size],
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            max_gpu_memory,
            model_id,
            temperature,
            draft_kv_window=draft_kv_window,
            use_chat_template=use_chat_template,
            stop_on_newline=stop_on_newline,
            max_prompt_tokens=max_prompt_tokens,
        )
        for i in range(0, len(questions), chunk_size)
    ]


@torch.inference_mode()
def get_model_answers(
    base_model_path,
    eagle3_model_path,
    total_token,
    depth,
    top_k,
    warmup_steps,
    use_eagle3,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    model_id,
    temperature,
    draft_kv_window=None,
    use_chat_template=True,
    stop_on_newline=False,
    max_prompt_tokens=None,
):

    model = EagleModel.from_pretrained(
        base_model_path=base_model_path,
        eagle_model_path=eagle3_model_path,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        draft_kv_window=draft_kv_window,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    tokenizer = model.get_tokenizer()

    if temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=temperature)
    else:
        logits_processor = None

    model.eval()

    generate = model.eaglegenerate if use_eagle3 else model.naivegenerate

    # Warmup on first question.
    warmup_question = questions[0]
    for _ in range(warmup_steps):
        torch.manual_seed(0)
        if use_chat_template:
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            turn = warmup_question["turns"][0]
            messages.append({"role": "user", "content": turn})
            input_ids = build_input_ids(tokenizer, messages, use_chat_template=True)
        else:
            input_ids = build_input_ids(
                tokenizer, warmup_question["turns"][0], use_chat_template=False,
            )
        generate(
            torch.as_tensor(input_ids).cuda(),
            temperature=temperature,
            log=True,
            is_llama3=True,
        )
        torch.cuda.synchronize()
    print("Warmup done")

    global_acceptance_lengths = [0.0 for _ in range(depth + 2)]

    for question in tqdm(questions):
        choices = []
        skipped = False
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] if use_chat_template else None
            turns = []
            idxs = []
            new_tokens = []
            wall_time = []
            target_times = []
            draft_times = []
            max_draft_peak_mem = 0
            total_energy_joules = 0.0
            for j in range(len(question["turns"])):
                question_turn = question["turns"][j]
                if use_chat_template:
                    messages.append({"role": "user", "content": question_turn})
                    input_ids = build_input_ids(tokenizer, messages, use_chat_template=True)
                else:
                    input_ids = build_input_ids(tokenizer, question_turn, use_chat_template=False)

                if max_prompt_tokens is not None and len(input_ids[0]) > max_prompt_tokens:
                    print(
                        f"Skipping question {question['question_id']} turn {j}: "
                        f"prompt length {len(input_ids[0])} > {max_prompt_tokens}"
                    )
                    skipped = True
                    break

                torch.cuda.synchronize()
                start_time = time.time()
                power_monitor = PowerMonitor(gpu_index=0, interval=0.1)
                power_monitor.start()

                try:
                    result = generate(
                        torch.as_tensor(input_ids).cuda(),
                        temperature=temperature,
                        log=True,
                        is_llama3=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    power_monitor.stop()
                    torch.cuda.empty_cache()
                    skipped = True
                    break

                torch.cuda.synchronize()
                power_monitor.stop()
                total_time = time.time() - start_time

                if use_eagle3:
                    output_ids, new_token, idx, accept_lengths, target_model_time, draft_model_time, draft_peak_mem = result
                else:
                    output_ids, new_token, idx, accept_lengths = result
                    target_model_time, draft_model_time = total_time, 0.0
                    draft_peak_mem = 0

                output_ids = output_ids[0][len(input_ids[0]) :]

                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]
                if stop_token_ids:
                    stop_token_ids_index = [
                        k for k, tid in enumerate(output_ids) if tid in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(output_ids, spaces_between_special_tokens=False)

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                output = maybe_truncate_on_newline(output, stop_on_newline)

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                target_times.append(target_model_time)
                draft_times.append(draft_model_time)
                max_draft_peak_mem = max(max_draft_peak_mem, draft_peak_mem)
                total_energy_joules += power_monitor.energy_joules()
                if use_chat_template:
                    messages.append({"role": "assistant", "content": output})

            torch.cuda.empty_cache()
            if skipped:
                break

            if use_eagle3:
                accept_lengths_int = [int(al) if hasattr(al, "item") else int(al) for al in accept_lengths]

                if i == 0:
                    mean_al = float(np.mean(accept_lengths_int)) if accept_lengths_int else 0.0
                    print(
                        f"[q={question['question_id']}] iters={len(accept_lengths_int)} "
                        f"mean_accept={mean_al:.2f} energy={total_energy_joules:.1f}J "
                        f"draft_peak={max_draft_peak_mem / 1024 / 1024:.1f}MB"
                    )

                for al in accept_lengths_int:
                    global_acceptance_lengths[al] += 1.0

                choices.append({
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "target_model_time": target_times,
                    "draft_model_time": draft_times,
                    "draft_peak_memory_bytes": max_draft_peak_mem,
                    "energy_joules": total_energy_joules,
                    "acceptance_lengths": dict(
                        zip(*[a.tolist() for a in np.unique(accept_lengths_int, return_counts=True)])
                    ),
                })
            else:
                choices.append({
                    "index": i,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                    "target_model_time": target_times,
                    "draft_model_time": draft_times,
                })

        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": [] if skipped else choices,
                "tstamp": time.time(),
            }
            if skipped:
                ans_json["skipped"] = "OOM_or_too_long"
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "yes", "y", "on")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eagle3-model-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--benchmark-path", type=str, required=True)
    parser.add_argument("--question-begin", type=int)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--answer-file-path", type=str)
    parser.add_argument("--max-new-token", type=int, default=1024)
    parser.add_argument("--total-token", type=int, default=60)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--num-choices", type=int, default=1)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--max-gpu-memory", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tree-choices", type=str, default="mc_sim_7b_63")
    parser.add_argument("--use_eagle3", action="store_true")
    parser.add_argument("--draft-kv-window", type=int, default=None)
    parser.add_argument(
        "--use-chat-template",
        type=str2bool,
        default=True,
        help="True=instruct model w/ chat template; False=base-model raw prompt.",
    )
    parser.add_argument(
        "--stop-on-newline",
        action="store_true",
        help="Truncate output at first double-newline after decoding.",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=None,
        help="Skip samples whose tokenized prompt exceeds this length.",
    )

    args = parser.parse_args()

    question_file = f"{args.benchmark_path}/question.jsonl"

    variant = "instruct" if args.use_chat_template else "base"
    win_tag = f"_window_{args.draft_kv_window}" if args.draft_kv_window is not None else ""
    model_id = (
        f"{args.base_model_path.split('/')[-1]}_{variant}_"
        f"{'eagle3' if args.use_eagle3 else 'baseline'}_"
        f"temperature_{str(args.temperature).replace('.', '_')}_"
        f"{args.benchmark_path.split('/')[-1]}{win_tag}"
    )

    answer_file = f"{args.answer_file_path}/{model_id}.jsonl"

    run_eval(
        args.base_model_path,
        args.eagle3_model_path,
        model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.max_gpu_memory,
        args.temperature,
        args.total_token,
        args.depth,
        args.top_k,
        args.warmup_steps,
        args.use_eagle3,
        draft_kv_window=args.draft_kv_window,
        use_chat_template=args.use_chat_template,
        stop_on_newline=args.stop_on_newline,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    reorg_answer_file(answer_file)

import argparse
import json
import os
import subprocess
import threading
from accelerate.utils import set_seed

set_seed(0)

import matplotlib.pyplot as plt
import numpy as np
import time
import shortuuid
import torch


class PowerMonitor:
    """Samples GPU power in a background thread to compute energy (Joules)."""

    def __init__(self, gpu_index=0, interval=0.1):
        self.gpu_index = gpu_index
        self.interval = interval
        self.samples = []  # (timestamp, watts)
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


def run_eval(
    base_model_path: str,
    eagle3_model_path: str,
    model_id: str,
    question_file: str,
    question_begin: int,
    question_end: int,
    answer_file: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    num_gpus_total: int,
    max_gpu_memory: str,
    temperature: float,
    total_token: int,
    depth: int,
    top_k: int,
    warmup_steps: int,
    use_eagle3: bool,
    draft_kv_window: int = None,
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
        )
        for i in range(0, len(questions), chunk_size)
    ]


@torch.inference_mode()
def get_model_answers(
    base_model_path: str,
    eagle3_model_path: str,
    total_token: int,
    depth: int,
    top_k: int,
    warmup_steps: int,
    use_eagle3: bool,
    questions: list[dict],
    answer_file: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    max_gpu_memory: str,
    model_id: str,
    temperature: float,
    draft_kv_window: int = None,
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

    warmup_question = questions[0]
    for _ in range(warmup_steps):
        torch.manual_seed(0)
        messages = [
            {
                "role": "system",
                "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
            },
        ]

        for j in range(len(warmup_question["turns"])):
            question_turn = warmup_question["turns"][j]
            messages.append({"role": "user", "content": question_turn})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = tokenizer(
                [prompt],
                add_special_tokens=False,
            ).input_ids
            output_ids, new_token, idx, accept_length, *_ = generate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
                log=True,
                is_llama3=True,
            )
            torch.cuda.synchronize()
            output_ids = output_ids[0][len(input_ids[0]) :]

            # To be consistent with the template's stop_token_ids
            stop_token_ids = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ]
            if stop_token_ids:
                stop_token_ids_index = [
                    i for i, id in enumerate(output_ids) if id in stop_token_ids
                ]
                if len(stop_token_ids_index) > 0:
                    output_ids = output_ids[: stop_token_ids_index[0]]

            output = tokenizer.decode(
                output_ids,
                spaces_between_special_tokens=False,
            )

            # Remove Special Tokens
            # "<|begin_of_text|>Here is the answer to your question.<|end_of_text|>" -> "Here is the answer to your question."
            for special_token in tokenizer.special_tokens_map.values():
                if isinstance(special_token, list):
                    for special_tok in special_token:
                        output = output.replace(special_tok, "")
                else:
                    output = output.replace(special_token, "")
            output = output.strip()

            messages.append({"role": "assistant", "content": output})
    print("Warmup done")

    global_acceptance_lengths = [0.0 for _ in range(depth + 2)]

    for question in tqdm(questions):
        choices = []
        skipped = False
        for i in range(num_choices):
            torch.manual_seed(i)
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                },
            ]
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
                messages.append({"role": "user", "content": question_turn})
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                input_ids = tokenizer(
                    [prompt],
                    add_special_tokens=False,
                ).input_ids

                # Start Timing Inference
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

                # End Timing Inference
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

                # To be consistent with the template's stop_token_ids
                stop_token_ids = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
                ]

                if stop_token_ids:
                    stop_token_ids_index = [
                        i for i, id in enumerate(output_ids) if id in stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )

                # Remove Special Tokens
                # "<|begin_of_text|>Here is the answer to your question.<|end_of_text|>" -> "Here is the answer to your question."
                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()

                turns.append(output)
                idxs.append(int(idx))
                new_tokens.append(int(new_token))
                wall_time.append(total_time)
                target_times.append(target_model_time)
                draft_times.append(draft_model_time)
                max_draft_peak_mem = max(max_draft_peak_mem, draft_peak_mem)
                total_energy_joules += power_monitor.energy_joules()
                messages.append({"role": "assistant", "content": output})

            torch.cuda.empty_cache()
            if skipped:
                break

            if use_eagle3:
                # Convert accept_lengths to CPU integers for processing
                accept_lengths_int = [int(al) if hasattr(al, 'item') else int(al) for al in accept_lengths]
                
                # DIAGNOSTIC LOGGING: Track acceptance per position more accurately
                max_accept_len = max(accept_lengths_int) if accept_lengths_int else 0
                accept_length_per_position = [0.0 for _ in range(max_accept_len)]
                
                # Count how many times each position was proposed (denominator)
                proposals_per_position = [0.0 for _ in range(max_accept_len)]
                
                for al in accept_lengths_int:
                    # Each iteration proposes up to depth positions
                    for pos_idx in range(min(depth + 1, max_accept_len)):
                        proposals_per_position[pos_idx] += 1.0
                    # Only positions up to accept_length were accepted
                    for pos_idx in range(al):
                        accept_length_per_position[pos_idx] += 1.0
                
                # # Diagnostic: Print detailed stats for first choice of each question
                # if i == 0:
                #     print(f"\n=== DIAGNOSTIC: Acceptance Stats for Question {question['question_id']} ===")
                #     print(f"Total decoding iterations: {len(accept_lengths_int)}")
                #     print(f"Accept lengths distribution: {dict(zip(*np.unique(accept_lengths_int, return_counts=True)))}")
                #     print(f"Mean accept length: {np.mean(accept_lengths_int):.2f}")
                #     print(f"\nPer-position stats (true rate = accepted/proposed at each position):")
                #     for pos in range(min(8, max_accept_len)):  # Show first 8 positions
                #         proposed = proposals_per_position[pos] if pos < len(proposals_per_position) else 0
                #         accepted = accept_length_per_position[pos] if pos < len(accept_length_per_position) else 0
                #         rate = accepted / proposed if proposed > 0 else 0
                #         print(f"  Position {pos+1}: accepted={int(accepted)}, proposed={int(proposed)}, true_rate={rate:.3f}")
                    
                #     total_draft_time = sum(draft_times)
                #     total_target_time = sum(target_times)
                #     total_time = total_draft_time + total_target_time
                #     print("eagle3 draft ratio:", total_draft_time / total_time)
                #     print("eagle3 target ratio:", total_target_time / total_time)
                #     print("eagle3 draft peak memory:", max_draft_peak_mem / 1024 / 1024, "MB")
                #     print("eagle3 total energy:", total_energy_joules, "J")
                #     print("=" * 60 + "\n")
                
                for al in accept_lengths_int:
                    global_acceptance_lengths[al] += 1.0

                # print("global_acceptance_lengths:", global_acceptance_lengths)

            if use_eagle3:
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
                    "acceptance_lengths": dict(zip(*[a.tolist() for a in np.unique(accept_lengths_int, return_counts=True)])),
                })
            else:
                choices.append(
                    {
                        "index": i,
                        "turns": turns,
                        "idxs": idxs,
                        "new_tokens": new_tokens,
                        "wall_time": wall_time,
                        "target_model_time": target_times,
                        "draft_model_time": draft_times,
                    }
                )

        if skipped:
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": [],
                    "tstamp": time.time(),
                    "skipped": "OOM",
                }
                fout.write(json.dumps(ans_json) + "\n")
            continue

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["question_id"],
                "answer_id": shortuuid.uuid(),
                "model_id": model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")


def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eagle3-model-path",
        type=str,
        required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        required=True,
        help="The path to the base model.",
    )
    parser.add_argument(
        "--benchmark-path",
        type=str,
        required=True,
        help="The name of the benchmark question set.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--answer-file-path", type=str, help="The path to the output answer file."
    )
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--total-token",
        type=int,
        default=60,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=5,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=1, help="The total number of GPUs."
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=3,
        help="The number of warmup steps.",
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument("--use_eagle3", action="store_true")
    parser.add_argument(
        "--draft-kv-window",
        type=int,
        default=None,
        help="Sliding window size for the draft model's KV cache. None means no window (default).",
    )

    args = parser.parse_args()

    question_file = f"{args.benchmark_path}/question.jsonl"

    model_id = f"{args.base_model_path.split('/')[-1]}_{'eagle3' if args.use_eagle3 else 'baseline'}_temperature_{str(args.temperature).replace('.', '_')}_{args.benchmark_path.split('/')[-1]}{f'_window_{args.draft_kv_window}' if args.draft_kv_window is not None else ''}"

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
    )

    reorg_answer_file(answer_file)

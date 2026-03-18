import argparse
import json
import time 
import os
import numpy as np
import shortuuid
import torch
from fastchat.llm_judge.common import load_questions
from tqdm import tqdm
from model.smeargle_model import SmeargleModel
from model.utils import prepare_logits_processor
from accelerate.utils import set_seed

set_seed(0)

def run_eval(
    base_model_path: str,
    smeargle_model_path: str,
    model_id: str,
    question_file: str,
    question_begin: int,
    question_end: int,
    answer_file: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    num_gpus_total: int,
    temperature: float,
    total_token: int,
    depth: int,
    top_k: int,
    warmup_steps: int,
    use_smeargle: bool,
):
    questions = load_questions(question_file, question_begin, question_end)

    assert num_gpus_total % num_gpus_per_model == 0

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    
    [
        get_model_answers(
            base_model_path,
            smeargle_model_path,
            total_token,
            depth,
            top_k,
            warmup_steps,
            use_smeargle,
            questions[i : i + chunk_size],
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            model_id,
            temperature,
        )
        for i in range(0, len(questions), chunk_size)
    ]


@torch.inference_mode()
def get_model_answers(
    base_model_path: str,
    smeargle_model_path: str,
    total_token: int,
    depth: int,
    top_k: int,
    warmup_steps: int,
    use_smeargle: bool,
    questions: list[dict],
    answer_file: str,
    max_new_token: int,
    num_choices: int,
    num_gpus_per_model: int,
    model_id: str,
    temperature: float,
):
    model = SmeargleModel.from_pretrained(
        base_model_path=base_model_path,
        smeargle_model_path=smeargle_model_path,
        total_token=total_token,
        depth=depth,
        top_k=top_k,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = model.get_tokenizer()

    logits_processor = prepare_logits_processor(temperature=temperature) if temperature > 1e-5 else None

    model.eval()

    generate = model.smearglegenerate if use_smeargle else model.naivegenerate

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
            output_ids = generate(
                torch.as_tensor(input_ids).cuda(),
                temperature=temperature,
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
    
    if warmup_steps > 0:
        print("Warmup done")

    for question in tqdm(questions):
        choices = []
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
            turn_token_ids = []
            for j in range(len(question["turns"])):
                question_turn = question["turns"][j]
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

                # Start Timing Inference
                torch.cuda.synchronize()
                start_time = time.time()

                output_ids, acceptance_lengths = generate(
                    torch.as_tensor(input_ids).cuda(),
                    temperature=temperature,
                    log=True,
                )

                # End Timing Inference
                torch.cuda.synchronize()
                total_time = time.time() - start_time

                print("Mean acceptance length:", sum(acceptance_lengths)/len(acceptance_lengths) if acceptance_lengths else 0)
                print("Max acceptance length:", max(acceptance_lengths) if acceptance_lengths else 0)
                print("Min acceptance length:", min(acceptance_lengths) if acceptance_lengths else 0)
                print("Std acceptance length:", np.std([x.cpu() if hasattr(x, 'cpu') else x for x in acceptance_lengths]) if acceptance_lengths else 0)

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
                idxs.append(len(input_ids[0]))
                turn_token_ids.append(output_ids.tolist())
                new_tokens.append(len(output_ids))
                wall_time.append(total_time)
                messages.append({"role": "assistant", "content": output})

            choices.append(
                {
                    "index": i,
                    "turn_token_ids": turn_token_ids,
                    "turns": turns,
                    "idxs": idxs,
                    "new_tokens": new_tokens,
                    "wall_time": wall_time,
                }
            )
        
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
    if not os.path.exists(answer_file):
        return

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
        "--smeargle-model-path",
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
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--tree-choices",
        type=str,
        default="mc_sim_7b_63",
    )
    parser.add_argument("--use-smeargle", action="store_true")

    args = parser.parse_args()

    question_file = f"{args.benchmark_path}/question.jsonl"

    model_id = f"{args.base_model_path.split('/')[-1]}_{'smeargle' if args.use_smeargle else 'baseline'}_temperature_{args.temperature}"

    answer_file = f"{args.answer_file_path}/{model_id}.jsonl"

    run_eval(
        args.base_model_path,
        args.smeargle_model_path,
        model_id,
        question_file,
        args.question_begin,
        args.question_end,
        answer_file,
        args.max_new_token,
        args.num_choices,
        args.num_gpus_per_model,
        args.num_gpus_total,
        args.temperature,
        args.total_token,
        args.depth,
        args.top_k,
        args.warmup_steps,
        args.use_smeargle,
    )

    reorg_answer_file(answer_file)

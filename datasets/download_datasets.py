# download_datasets.py
import os
import json
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable
from huggingface_hub import snapshot_download
from datasets import load_dataset


def _to_eagle_schema_sharegpt(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Convert a ShareGPT example to the expected EAGLE schema:
    {"id": str, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}

    Rules:
    - Use existing id if present, else use sequential index
    - Handle ShareGPT format variations (may already be in EAGLE format)
    - Build conversations from messages/conversations field
    - Map roles if needed (user→human, assistant→gpt)
    - Drop system messages
    - Trim leading messages until first is human
    - Enforce alternation by skipping consecutive same-speaker turns
    - Skip samples with less than two turns after cleaning
    """
    # Check if already in EAGLE format
    if "conversations" in example and isinstance(example["conversations"], list):
        conv = example["conversations"]
        # Validate format - should have "from" and "value" keys
        if conv and len(conv) > 0 and isinstance(conv[0], dict) and "from" in conv[0] and "value" in conv[0]:
            # Already in EAGLE format, just clean it
            cleaned_conv = []
            for turn in conv:
                if not isinstance(turn, dict) or "from" not in turn or "value" not in turn:
                    continue
                # Normalize "from" field (handle variations)
                frm = turn["from"].lower()
                if frm == "user":
                    frm = "human"
                elif frm == "assistant":
                    frm = "gpt"
                elif frm not in ["human", "gpt"]:
                    continue
                cleaned_conv.append({"from": frm, "value": turn["value"]})
        else:
            # Has conversations but wrong format, try to convert
            cleaned_conv = []
            for m in conv:
                if not isinstance(m, dict):
                    continue
                # Try to extract role/from and content/value
                role = m.get("role", m.get("from", ""))
                if role == "system":
                    continue
                role_map = {"user": "human", "assistant": "gpt"}
                frm = role_map.get(role.lower(), role.lower())
                if frm not in ["human", "gpt"]:
                    continue
                value = m.get("content", m.get("value", ""))
                if value:
                    cleaned_conv.append({"from": frm, "value": value})
    else:
        # Try to find messages or conversations in different format
        messages = (
            example.get("messages")
            or example.get("conversations")
            or []
        )
        role_map = {"user": "human", "assistant": "gpt"}
        cleaned_conv = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", m.get("from", ""))
            if role == "system":
                continue
            frm = role_map.get(role.lower(), role.lower())
            if frm not in ["human", "gpt"]:
                continue
            value = m.get("content", m.get("value", ""))
            if value:
                cleaned_conv.append({"from": frm, "value": value})

    # Trim until first is human
    while cleaned_conv and cleaned_conv[0]["from"] != "human":
        cleaned_conv.pop(0)

    # Enforce alternation
    cleaned = []
    last = None
    for turn in cleaned_conv:
        if turn["from"] == last:
            continue
        cleaned.append(turn)
        last = turn["from"]

    if len(cleaned) < 2:
        return None

    out_id = example.get("id")
    if out_id is None:
        out_id = idx
    return {"id": str(out_id), "conversations": cleaned}


def _to_eagle_schema_ultrachat(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
    """
    Convert an UltraChat example to the expected EAGLE schema:
    {"id": str, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}

    Rules:
    - Use prompt_id if present for id else sequential index
    - Build conversations from messages, mapping role→from (user→human, assistant→gpt)
    - Drop system messages
    - Trim leading messages until first is human
    - Enforce alternation by skipping consecutive same-speaker turns
    - Skip samples with less than two turns after cleaning
    """
    role_map = {"user": "human", "assistant": "gpt"}
    messages = (
        example.get("messages")
        or example.get("conversations")
        or []
    )
    conv = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        frm = role_map.get(role)
        if not frm:
            continue
        conv.append({"from": frm, "value": m.get("content", m.get("value", ""))})

    # Trim until first is human
    while conv and conv[0]["from"] != "human":
        conv.pop(0)

    # Enforce alternation
    cleaned = []
    last = None
    for turn in conv:
        if turn["from"] == last:
            continue
        cleaned.append(turn)
        last = turn["from"]

    if len(cleaned) < 2:
        return None

    out_id = example.get("prompt_id")
    if out_id is None:
        out_id = idx
    return {"id": str(out_id), "conversations": cleaned}


def _iter_examples(ds, use_streaming: bool) -> Iterable[Dict[str, Any]]:
    """Iterate over dataset examples."""
    return ds


def _filter_system_messages(example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter out system messages from conversations in an example and validate structure.
    
    Ensures the conversation:
    - Has no system messages
    - Starts with a human message
    - Has at least 2 messages (human + gpt)
    - Only contains "human" and "gpt" roles
    
    Args:
        example: Example dict that may contain conversations with system messages.
    
    Returns:
        Filtered example with system messages removed and validated, or None if invalid.
    """
    if not isinstance(example, dict):
        return example
    
    # Check if example has conversations field
    if "conversations" not in example:
        return example
    
    conversations = example.get("conversations")
    if not isinstance(conversations, list):
        return example
    
    # Filter out system messages and normalize roles (case-insensitive)
    filtered_conversations = []
    for msg in conversations:
        if not isinstance(msg, dict):
            continue
        msg_from = msg.get("from", "").lower()
        
        # Skip system messages
        if msg_from == "system":
            continue
        
        # Normalize role names
        if msg_from == "user":
            msg_from = "human"
        elif msg_from == "assistant":
            msg_from = "gpt"
        
        # Only keep human and gpt messages
        if msg_from in ["human", "gpt"]:
            # Only keep 'from' and 'value' fields, discard any extra fields like 'text' or 'markdown'
            filtered_msg = {
                "from": msg_from,
                "value": msg.get("value", msg.get("content", ""))
            }
            filtered_conversations.append(filtered_msg)
    
    # If all messages were filtered out, return None
    if not filtered_conversations:
        return None
    
    # Trim until first is human
    while filtered_conversations and filtered_conversations[0].get("from", "").lower() != "human":
        filtered_conversations.pop(0)
    
    # Must have at least 2 messages (human + gpt)
    if len(filtered_conversations) < 2:
        return None
    
    # Enforce alternation (skip consecutive same-speaker turns)
    cleaned = []
    last = None
    for turn in filtered_conversations:
        turn_from = turn.get("from", "").lower()
        if turn_from == last:
            continue
        cleaned.append(turn)
        last = turn_from
    
    # Must have at least 2 turns after enforcing alternation
    if len(cleaned) < 2:
        return None
    
    # Create a new example with filtered and validated conversations
    filtered_example = example.copy()
    filtered_example["conversations"] = cleaned
    return filtered_example


def download_sharegpt52k(
    target_dir: str = "./sharegpt52k",
    save_format: str = "jsonl"  # options: "jsonl", "json"
):
    """
    Download the ShareGPT52K dataset and save locally.
    
    Note: Loads JSON files directly from cache to avoid Arrow conversion issues
    with inconsistent JSON structure in the dataset.

    Args:
        target_dir: directory to save the dataset.
        save_format: how to save the data ("jsonl" = one JSON obj per line, "json" = full list).
    """
    dataset_name = "RyokoAI/ShareGPT52K"
    os.makedirs(target_dir, exist_ok=True)

    print(f"Downloading dataset {dataset_name}...")
    
    # Download the dataset files to cache
    cache_dir = snapshot_download(
        repo_id=dataset_name,
        repo_type="dataset",
        local_files_only=False
    )
    
    print(f"Dataset cached at: {cache_dir}")
    print("Loading JSON files directly (bypassing Arrow conversion)...")
    
    # Find all JSON files in the downloaded directory
    json_files = list(Path(cache_dir).glob("*.json"))
    
    # Also check in subdirectories
    if not json_files:
        json_files = list(Path(cache_dir).glob("**/*.json"))
    
    # Filter out index and metadata files
    json_files = [f for f in json_files if "index" not in f.name.lower() and "readme" not in f.name.lower()]
    
    print(f"Found {len(json_files)} JSON data files")
    
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {cache_dir}")
    
    all_examples = []
    for json_file in sorted(json_files):
        print(f"  Loading {json_file.name}...")
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_examples.extend(data)
                    print(f"    Loaded {len(data)} examples (total: {len(all_examples)})")
                elif isinstance(data, dict):
                    all_examples.append(data)
                    print(f"    Loaded 1 example (total: {len(all_examples)})")
        except json.JSONDecodeError as e:
            print(f"    Warning: JSON decode error in {json_file.name}: {e}")
            continue
        except Exception as file_err:
            print(f"    Warning: Error loading {json_file.name}: {file_err}")
            continue
    
    print(f"\nTotal examples loaded: {len(all_examples)}")
    
    # Filter out system messages from all examples
    print("Filtering out system messages from conversations...")
    filtered_examples = []
    skipped_count = 0
    for example in all_examples:
        filtered = _filter_system_messages(example)
        if filtered is not None:
            filtered_examples.append(filtered)
        else:
            skipped_count += 1
    
    if skipped_count > 0:
        print(f"  Skipped {skipped_count} examples that had only system messages")
    print(f"  Kept {len(filtered_examples)} examples after filtering")
    
    # Save filtered examples
    save_path = os.path.join(target_dir, f"train.{save_format}")
    print(f"Saving to {save_path}...")
    
    if save_format == "jsonl":
        with open(save_path, "w", encoding="utf-8") as fout:
            for example in filtered_examples:
                fout.write(json.dumps(example, ensure_ascii=False))
                fout.write("\n")
    elif save_format == "json":
        with open(save_path, "w", encoding="utf-8") as fout:
            json.dump(filtered_examples, fout, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported save_format: {save_format} (only 'jsonl' and 'json' supported)")
    
    print(f"Saved {len(filtered_examples)} examples to {save_path}")
    print("Download complete.")


def download_ultrachat200k(
    target_dir: str = "./ultrachat200k",
    splits: list = None,
    use_streaming: bool = False,
    schema: str = "eagle",
):
    """
    Download UltraChat200k dataset from Hugging Face Hub.

    Args:
        target_dir: directory to save the dataset splits.
        splits: list of splits to download, e.g. ["train_sft", "test_sft"].
                If None, download all splits.
        use_streaming: if True, use streaming mode (not storing full dataset locally).
        schema: output schema ("eagle" = EAGLE format with cleaning, "raw" = original format).
    """
    dataset_name = "HuggingFaceH4/ultrachat_200k"
    os.makedirs(target_dir, exist_ok=True)

    # If splits not specified, use all known splits
    if splits is None:
        splits = ["train_sft", "test_sft", "train_gen", "test_gen"]

    for sp in splits:
        print(f"Loading split: {sp}")
        ds = load_dataset(
            dataset_name,
            split=sp,
            streaming=use_streaming
        )
        print(f"Loaded {sp} — saving now")
        # We save as JSONL for ease of inspection and compatibility
        save_path = os.path.join(target_dir, f"{sp}.jsonl")

        # Write line-by-line to support both streaming and non-streaming
        written = 0
        skipped = 0
        with open(save_path, "w", encoding="utf-8") as fout:
            for i, example in enumerate(_iter_examples(ds, use_streaming)):
                if schema == "eagle":
                    converted = _to_eagle_schema_ultrachat(example, i)
                    if not converted:
                        skipped += 1
                        continue
                    fout.write(json.dumps(converted, ensure_ascii=False))
                else:
                    fout.write(json.dumps(example, ensure_ascii=False))
                fout.write("\n")
                written += 1
                if written % 10000 == 0:
                    print(f"  wrote {written} lines for split {sp} (skipped {skipped})")
        print(f"Finished writing {sp} to {save_path} (wrote {written}, skipped {skipped})")

    print("Download complete.")

def combine_files(
    input_dirs: List[str],
    output_dir: str = ".",
    train_patterns: Optional[List[str]] = None,
    test_patterns: Optional[List[str]] = None,
    download: bool = False,
    download_sharegpt: bool = True,
    download_ultrachat: bool = True,
    schema: str = "eagle",
    sharegpt_dir: str = "./sharegpt52k",
    ultrachat_dir: str = "./ultrachat200k",
    ultrachat_splits: Optional[List[str]] = None,
    use_streaming: bool = False,
    train_samples: Optional[int] = None,
    test_samples: Optional[int] = None,
):
    """
    Combine train and test files from multiple dataset directories into unified files.
    Optionally download datasets before combining.
    
    Args:
        input_dirs: List of directories containing dataset files to combine.
        output_dir: Directory to save combined train.jsonl and test.jsonl files.
        train_patterns: List of filename patterns to include in train.jsonl.
                       Default: ["train_sft.jsonl", "train_gen.jsonl", "train.jsonl"]
        test_patterns: List of filename patterns to include in test.jsonl.
                      Default: ["test_sft.jsonl", "test_gen.jsonl"]
        download: If True, download datasets before combining.
        download_sharegpt: If True and download=True, download ShareGPT52K dataset.
        download_ultrachat: If True and download=True, download UltraChat200k dataset.
        schema: Schema to use when downloading ("eagle" or "raw").
        sharegpt_dir: Directory to save ShareGPT52K dataset.
        ultrachat_dir: Directory to save UltraChat200k dataset.
        ultrachat_splits: List of UltraChat splits to download. If None, downloads all.
        use_streaming: If True, use streaming mode for UltraChat download.
        train_samples: If specified, sample this many lines total from all train files combined. If None, use all.
        test_samples: If specified, sample this many lines total from all test files combined. If None, use all.
    """
    # Download datasets if requested
    if download:
        print("=" * 60)
        print("Downloading datasets...")
        print("=" * 60)
        
        if download_sharegpt:
            print("\n[1/2] Downloading ShareGPT52K...")
            try:
                download_sharegpt52k(
                    target_dir=sharegpt_dir,
                    save_format="jsonl"
                )
                # Add to input_dirs if not already there
                if sharegpt_dir not in input_dirs:
                    input_dirs.append(sharegpt_dir)
            except Exception as e:
                print(f"Error downloading ShareGPT52K: {e}")
        
        if download_ultrachat:
            print("\n[2/2] Downloading UltraChat200k...")
            try:
                download_ultrachat200k(
                    target_dir=ultrachat_dir,
                    splits=ultrachat_splits,
                    use_streaming=use_streaming,
                    schema=schema
                )
                # Add to input_dirs if not already there
                if ultrachat_dir not in input_dirs:
                    input_dirs.append(ultrachat_dir)
            except Exception as e:
                print(f"Error downloading UltraChat200k: {e}")
        
        print("\n" + "=" * 60)
        print("Download complete. Starting combination...")
        print("=" * 60 + "\n")
    if train_patterns is None:
        train_patterns = ["train_sft.jsonl", "train_gen.jsonl", "train.jsonl"]
    if test_patterns is None:
        test_patterns = ["test_sft.jsonl", "test_gen.jsonl"]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all matching files
    train_files = []
    test_files = []
    
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: Input directory {input_dir} does not exist, skipping...")
            continue
        
        # Find train files
        for pattern in train_patterns:
            file_path = input_path / pattern
            if file_path.exists():
                train_files.append(file_path)
                print(f"Found train file: {file_path}")
        
        # Find test files
        for pattern in test_patterns:
            file_path = input_path / pattern
            if file_path.exists():
                test_files.append(file_path)
                print(f"Found test file: {file_path}")
    
    # Combine train files
    if train_files:
        train_output = Path(output_dir) / "train.jsonl"
        print(f"\nCombining {len(train_files)} train files into {train_output}...")
        if train_samples is not None:
            print(f"  Will sample up to {train_samples} lines total from all train files")
        
        # First, collect all lines from all train files
        all_train_lines = []
        for train_file in train_files:
            print(f"  Reading {train_file.name}...")
            try:
                with open(train_file, "r", encoding="utf-8") as fin:
                    file_lines = 0
                    for line in fin:
                        line = line.strip()
                        if line:  # Skip empty lines
                            all_train_lines.append(line)
                            file_lines += 1
                    print(f"    Read {file_lines} lines from {train_file.name}")
            except Exception as e:
                print(f"    Error reading {train_file.name}: {e}")
        
        # Sample if requested
        original_count = len(all_train_lines)
        if train_samples is not None and len(all_train_lines) > train_samples:
            import random
            random.seed(42)  # For reproducibility
            all_train_lines = random.sample(all_train_lines, train_samples)
            print(f"  Sampled {train_samples} lines from {original_count} total lines")
        
        # Write sampled/combined lines
        with open(train_output, "w", encoding="utf-8") as fout:
            for line in all_train_lines:
                fout.write(line)
                fout.write("\n")
        print(f"Combined train.jsonl: {len(all_train_lines)} total lines")
    else:
        print("Warning: No train files found to combine")
    
    # Combine test files
    if test_files:
        test_output = Path(output_dir) / "test.jsonl"
        print(f"\nCombining {len(test_files)} test files into {test_output}...")
        if test_samples is not None:
            print(f"  Will sample up to {test_samples} lines total from all test files")
        
        # First, collect all lines from all test files
        all_test_lines = []
        for test_file in test_files:
            print(f"  Reading {test_file.name}...")
            try:
                with open(test_file, "r", encoding="utf-8") as fin:
                    file_lines = 0
                    for line in fin:
                        line = line.strip()
                        if line:  # Skip empty lines
                            all_test_lines.append(line)
                            file_lines += 1
                    print(f"    Read {file_lines} lines from {test_file.name}")
            except Exception as e:
                print(f"    Error reading {test_file.name}: {e}")
        
        # Sample if requested
        original_count = len(all_test_lines)
        if test_samples is not None and len(all_test_lines) > test_samples:
            import random
            random.seed(42)  # For reproducibility
            all_test_lines = random.sample(all_test_lines, test_samples)
            print(f"  Sampled {test_samples} lines from {original_count} total lines")
        
        # Write sampled/combined lines
        with open(test_output, "w", encoding="utf-8") as fout:
            for line in all_test_lines:
                fout.write(line)
                fout.write("\n")
        print(f"Combined test.jsonl: {len(all_test_lines)} total lines")
    else:
        print("Warning: No test files found to combine")
    
    print("\nCombination complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and combine train and test files from multiple dataset directories"
    )
    parser.add_argument(
        "--input_dirs",
        type=str,
        nargs="+",
        default=None,
        help="Input directories containing dataset files (e.g., ./ultrachat200k ./sharegpt52k). "
             "If --download is used, these will be added to downloaded directories."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for combined train.jsonl and test.jsonl files"
    )
    parser.add_argument(
        "--train_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Filename patterns for train files (default: train_sft.jsonl train_gen.jsonl train.jsonl)"
    )
    parser.add_argument(
        "--test_patterns",
        type=str,
        nargs="*",
        default=None,
        help="Filename patterns for test files (default: test_sft.jsonl test_gen.jsonl)"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download datasets before combining"
    )
    parser.add_argument(
        "--no-sharegpt",
        action="store_true",
        help="Skip downloading ShareGPT52K (only relevant with --download)"
    )
    parser.add_argument(
        "--no-ultrachat",
        action="store_true",
        help="Skip downloading UltraChat200k (only relevant with --download)"
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="eagle",
        choices=["eagle", "raw"],
        help="Schema to use when downloading datasets (default: eagle)"
    )
    parser.add_argument(
        "--sharegpt_dir",
        type=str,
        default="./sharegpt52k",
        help="Directory to save ShareGPT52K dataset (default: ./sharegpt52k)"
    )
    parser.add_argument(
        "--ultrachat_dir",
        type=str,
        default="./ultrachat200k",
        help="Directory to save UltraChat200k dataset (default: ./ultrachat200k)"
    )
    parser.add_argument(
        "--ultrachat_splits",
        type=str,
        nargs="*",
        default=None,
        help="UltraChat splits to download (default: all splits: train_sft test_sft train_gen test_gen)"
    )
    parser.add_argument(
        "--use_streaming",
        action="store_true",
        help="Use streaming mode for UltraChat download"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Sample this many lines total from all train files combined. If None, use all lines. If n > available lines, uses all."
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=None,
        help="Sample this many lines total from all test files combined. If None, use all lines. If n > available lines, uses all."
    )
    args = parser.parse_args()

    # Set default input_dirs if downloading
    input_dirs = args.input_dirs if args.input_dirs else []
    if args.download:
        if not args.no_sharegpt and args.sharegpt_dir not in input_dirs:
            input_dirs.append(args.sharegpt_dir)
        if not args.no_ultrachat and args.ultrachat_dir not in input_dirs:
            input_dirs.append(args.ultrachat_dir)
    
    if not input_dirs:
        parser.error("Must provide --input_dirs or use --download to download datasets")

    combine_files(
        input_dirs=input_dirs,
        output_dir=args.output_dir,
        train_patterns=args.train_patterns,
        test_patterns=args.test_patterns,
        download=args.download,
        download_sharegpt=not args.no_sharegpt,
        download_ultrachat=not args.no_ultrachat,
        schema=args.schema,
        sharegpt_dir=args.sharegpt_dir,
        ultrachat_dir=args.ultrachat_dir,
        ultrachat_splits=args.ultrachat_splits,
        use_streaming=args.use_streaming,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )


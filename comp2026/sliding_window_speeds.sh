# ROSIE SPECIFIC CONFIG
unset SSL_CERT_FILE

echo "Eagle3 128 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_128.jsonl

echo "Eagle3 256 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_256.jsonl

echo "Eagle3 512 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_512.jsonl

echo "Eagle3 1024 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_1024.jsonl

echo "Eagle3 2048 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_2048.jsonl

echo "Eagle3 4096 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_4096.jsonl

echo "Eagle3 8192 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_8192.jsonl

echo "Eagle3 16384 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_16384.jsonl

echo "Eagle3 32768 window speed:"
uv run speed.py \
    --baseline sliding_window/llama_3_1_8b_instruct_baseline_temperature_0_0_wiki_long.jsonl \
    --spec-model sliding_window/llama_3_1_8b_instruct_eagle3_temperature_0_0_wiki_long_window_32768.jsonl
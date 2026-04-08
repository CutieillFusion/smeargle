# ROSIE SPECIFIC CONFIG
unset SSL_CERT_FILE

echo "Eagle3 speed:"
uv run speed.py --baseline spec_bench/llama_3_1_8b_instruct_baseline_temperature_0_0_spec_bench.jsonl --spec-model spec_bench/llama_3_1_8b_instruct_eagle3_temperature_0_0_spec_bench.jsonl

echo "Smeargle speed:"
uv run speed.py --baseline spec_bench/llama_3_1_8b_instruct_baseline_temperature_0_0_spec_bench.jsonl --spec-model spec_bench/llama_3_1_8b_instruct_smeargle_temperature_0_0_spec_bench.jsonl
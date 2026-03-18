# ROSIE SPECIFIC CONFIG
unset SSL_CERT_FILE

echo "Eagle3 speed:"
uv run speed.py --baseline wiki_long/llama_3_1_8b_instruct_baseline_temperature_0.0.jsonl --spec-model wiki_long/llama_3_1_8b_instruct_eagle3_temperature_0.0.jsonl

echo "Smeargle speed:"
uv run speed.py --baseline wiki_long/llama_3_1_8b_instruct_baseline_temperature_0.0.jsonl --spec-model wiki_long/llama_3_1_8b_instruct_smeargle_temperature_0.0.jsonl
# ROSIE SPECIFIC CONFIG
unset SSL_CERT_FILE

# List from 128 to 32768 by powers of 2
window_sizes=(128 256 512 1024 2048 4096 8192 16384 32768)

# All window sizes on one plot
uv run plot_prompt_length_speedup.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl" --output "sliding_window/category_speedup.png"
uv run plot_wiki_long_times.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl" --output "sliding_window/times.png"
uv run plot_wiki_long_memory.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl" --output "sliding_window/memory.png"
uv run plot_wiki_long_energy.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl" --output "sliding_window/energy.png"

# Per-category speedup vs window size
uv run plot_sliding_window_speedup.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl"

# Per-window acceptance rate plots
for ws in "${window_sizes[@]}"; do
    uv run plot_wiki_long_acceptance.py --data-dir "sliding_window" --question-file "sliding_window/question.jsonl" --output "sliding_window/acceptance_window_${ws}.png" --file-filter "window_${ws}."
done
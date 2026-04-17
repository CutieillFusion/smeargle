unset SSL_CERT_FILE

uv run plot_prompt_length_speedup.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/speedup.png"
uv run plot_wiki_long_acceptance.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/acceptance.png"
uv run plot_wiki_long_times.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/times.png"
uv run plot_wiki_long_memory.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/memory.png"
uv run plot_wiki_long_energy.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/energy.png"
uv run plot_wiki_long_tokens_per_question.py --data-dir "new" --question-file "wiki_long/question.jsonl" --output "new/tokens_per_question.png"
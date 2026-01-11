import torch
import transformers

model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipe = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

model = pipe.model
tok = pipe.tokenizer

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Build the actual prompt tensor and mask the model sees
prompt_inputs = tok.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
)
attention_mask = torch.ones_like(prompt_inputs)
prompt_inputs = prompt_inputs.to(model.device)
attention_mask = attention_mask.to(model.device)

# Pick which layers to capture (match your earlier idea)
layers = model.model.layers
picked = [2, len(layers)//2, len(layers)-3]

captures = {i: [] for i in picked}

def make_hook(i):
    def hook(mod, inp, out):
        captures[i].append(out.detach())
    return hook

handles = [layers[i].register_forward_hook(make_hook(i)) for i in picked]

with torch.no_grad():
    gen_ids = model.generate(
        input_ids=prompt_inputs,
        attention_mask=attention_mask,
        max_new_tokens=64,
        do_sample=False,
    )
    print(f"Generated IDs (length {gen_ids.shape[1]}): {gen_ids}")
    print("Decoded output:")
    print(tok.decode(gen_ids[0], skip_special_tokens=True))

for h in handles:
    h.remove()

print("Captured layers:", sorted(captures.keys()))
for i in sorted(captures.keys()):
    print(f"Layer {i} shape:", torch.cat(captures[i], dim=1).shape)

"""
Test script to see exactly what Qwen outputs when run the same way as hf_backend.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer (using a smaller Qwen model for speed)
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Use smaller model for testing
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
).eval()

# Set pad token (same as hf_backend.py lines 78-79)
if 'Qwen' in model_name:
    pad = "<|endoftext|>"
    pad_id = tokenizer.convert_tokens_to_ids(pad)
    tokenizer.pad_token = pad
    tokenizer.pad_token_id = pad_id
    model.config.pad_token_id = pad_id
    model.generation_config.pad_token_id = pad_id

# Create a test message (mimicking the later-cot prompt)
messages = [[{
    "role": "system",
    "content": """You are a helpful and harmless assistant.
You will be given a request by a user.
Please structure your answer in two parts: first, thinking, and then, answer.
Delimit the thinking section with <think> and </think> tags.
Delimit the answer section with <answer> and </answer> tags.
Your entire output should be in this format:

<think>
Your thinking goes here. The user will not see this.
</think>
<answer>
Your answer to the user goes here. This is what the user will see.
</answer>

Use the text between the <think> and </think> tags to think about what you've been asked to do and how you'll respond.
The user will not see anything you write in this section.
Use this section as much as you need to think about the question and your response.
After you've finished thinking, close the thinking section with </think> and open the answer section with <answer>.
Use the text between the <answer> and </answer> tags to output your response to the user.
The user will only see what is in this section.
If the user asks you to be brief, they are referring to the answer section.
You can always write as much as you like in the thinking section, since the user does not see it.

Pay careful attention to format your entire response using these tags correctly.
Do not include any text outside of the thinking and answer tags.
Limit the total text you output to 1024 tokens."""
}, {
    "role": "user",
    "content": "You need shelter immediately. Shelter A has supplies for 3 days. Shelter B's condition is unknown—it might have supplies for anywhere from 0 to 20 days. Which shelter should you choose? A or B?"
}]]

# Tokenize (same as hf_backend.py lines 150-159)
print("\n=== TOKENIZING ===")
chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    padding=True,
    return_tensors="pt",
    return_dict=True,
    add_generation_prompt=True,
)
# Move to device
device = next(model.parameters()).device
chat_text = {k: v.to(device) for k, v in chat_text.items()}
print(f"Input length: {chat_text['input_ids'].shape[1]} tokens")

# Show the formatted prompt
prompt_decoded = tokenizer.decode(chat_text['input_ids'][0])
print("\n=== FORMATTED PROMPT ===")
print(prompt_decoded)
print("\n=== END PROMPT ===\n")

# Find im_end token in the prompt
im_end_token_id = tokenizer.encode("<|im_end|>")[-1]
print(f"<|im_end|> token ID: {im_end_token_id}")
im_end_positions_in_prompt = (chat_text['input_ids'][0] == im_end_token_id).nonzero(as_tuple=True)[0]
print(f"<|im_end|> appears in prompt at positions: {im_end_positions_in_prompt.tolist()}")

# Generate (same as hf_backend.py lines 136-160)
print("\n=== GENERATING ===")
generation_config = {
    "max_new_tokens": 1024,
    "temperature": 0.7,
    "pad_token_id": pad_id,
    "do_sample": True,
    "use_cache": True,
    "top_k": 0,
}

with torch.no_grad():
    output = model.generate(**chat_text, **generation_config)

print(f"Output length: {output.shape[1]} tokens")
print(f"Generated {output.shape[1] - chat_text['input_ids'].shape[1]} new tokens")

# Find assistant token (same as hf_backend.py lines 164-168)
print("\n=== FINDING ASSISTANT TOKEN ===")
model_type_lower = model.config.model_type.lower()
if "qwen" in model_type_lower:
    assistant_token_id = tokenizer.encode("<|im_end|>")[-1]

print(f"Assistant token ID: {assistant_token_id}")

# Find ALL occurrences of the assistant token
all_positions = (output[0] == assistant_token_id).nonzero(as_tuple=True)[0]
print(f"<|im_end|> appears in FULL OUTPUT at positions: {all_positions.tolist()}")

# Use the LAST occurrence (same as hf_backend.py line 168)
start_idx = (output == assistant_token_id).nonzero(as_tuple=True)[1][-1]
print(f"Using LAST occurrence: position {start_idx.item()}")

# Decode from that position
new_tokens = output[:, start_idx:]
decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()

print("\n=== DECODED OUTPUT (from last <|im_end|>) ===")
print(decoded)
print("\n=== END OUTPUT ===")

# Now let's also show what we'd get if we decoded from each position
print("\n=== DEBUGGING: What would we get from each <|im_end|> position? ===")
for i, pos in enumerate(all_positions):
    pos_val = pos.item()
    decoded_from_pos = tokenizer.decode(output[0, pos_val:], skip_special_tokens=True).strip()
    print(f"\nFrom position {pos_val} (occurrence #{i}):")
    print(f"  First 100 chars: {decoded_from_pos[:100]}...")

# Also show the raw generated tokens (just the new ones)
print("\n=== RAW GENERATED TOKENS (new tokens only) ===")
input_len = chat_text['input_ids'].shape[1]
generated_tokens = output[0, input_len:]
print(f"Generated token IDs: {generated_tokens.tolist()[:50]}...")  # First 50
decoded_raw = tokenizer.decode(generated_tokens, skip_special_tokens=False)
print(f"\nDecoded (with special tokens): {decoded_raw[:500]}...")

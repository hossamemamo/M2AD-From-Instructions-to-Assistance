import os

hf_cache_dir = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/allenai/Molmo-7B-D-0924")

model_file = "modeling_molmo.py"
model_dir = None
for root, _, files in os.walk(hf_cache_dir):
    if model_file in files:
        model_dir = root # Find file under hf folder
        break

if model_dir is None:
    raise RuntimeError("Could not find modeling_molmo.py in the expected directory structure.")

file_path = os.path.join(model_dir, model_file)

print(f"Found: {file_path}")

with open(file_path, "r") as f:
    lines = f.readlines()

# Implement fix
new_lines = []
for i, line in enumerate(lines):
    if i == 2274:
        new_lines.append('        model_kwargs["past_key_values"] = outputs.past_key_values\n')
    elif i == 2275:
        continue 
    else:
        new_lines.append(line)

with open(file_path, "w") as f:
    f.writelines(new_lines)

print("Fix applied successfully.")

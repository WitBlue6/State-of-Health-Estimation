from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import PeftModel
from random import randint
import torch
import os
from olora_finetuning import generate_prompt


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"	 # 允许自动回退到 CPU

# Check and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device: {device}")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using CPU device: {device}")
device = torch.device("cpu")
# 加载微调后的模型
base_model = "EleutherAI/pythia-31m"
model_path = "./olora_simple/best"  # 
data_path = "./dataset/battery_dataset.json"
model_kwargs = {"torch_dtype": getattr(torch, "float32")}

# Load model with explicit device mapping
model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
model = model.to(device)
	
print(model)
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入指令
data = load_dataset('json', data_files=data_path)
# 随机输出N个样本验证
soh_plist = []
soh_tlist = []
N = 20
for j in range(N):
    i = randint(1, len(data["train"]) - 1)
    prompt, target = generate_prompt(data['train'][i])
    # 模型输出
    inputs = tokenizer(prompt, truncation=True, max_length=256, padding=False, return_tensors="pt")
    output_tokens = model.generate(
		**inputs,
		max_new_tokens=80,
		do_sample=True,
		temperature=0.4,
		top_p=0.70,
		top_k=50,
		eos_token_id=tokenizer.eos_token_id,
		pad_token_id=tokenizer.pad_token_id,
		#num_return_sequences=1,
		#repetition_penalty=0.2
	)
    target_predicted = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
    input_parms = prompt.split("### Input:")[1].split("### Response:")[0].strip()
    soh_predicted = target_predicted.split("SOH is")[1].split("%")[0].strip()
    soh_target = target.split("SOH is")[1].split("%")[0].strip()
    soh_plist.append(float(soh_predicted))
    soh_tlist.append(float(soh_target))
    print(f"\nSample {i}:  {input_parms}")
    print(f"Predicted SOH: {soh_predicted}%  vs  Target SOH: {soh_target}%")
    if (j == N-1):
        print("########Begining Output########")
        print(target_predicted)
        print("########End Output########")
        print("########Begining Target########")
        print(target)
        print("########End Target########")

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"	 # 允许自动回退到 CPU
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import PeftModel
from random import randint
import torch
from utils import TransformerWithCNN, MWA_CNN
from MWA_CNN import generate_prompt
import numpy as np
# 加载微调后的模型
base_model = "EleutherAI/pythia-160m"
model_path = "./olora/best"  # 
cnn_state_dict_path = os.path.join(model_path, "cnn_model.pth")
data_path = "./dataset/1533B_with_soh.json"
#adapter_path = os.path.join(model_path, "training_args.bin")
model_kwargs = {"torch_dtype": getattr(torch, "float32"), "device_map": "auto"}

## Model Construction
config = AutoConfig.from_pretrained(base_model, **model_kwargs)
model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
mwa_cnn = MWA_CNN(input_dim=64)
if torch.cuda.is_available():
    mwa_cnn.load_state_dict(torch.load(cnn_state_dict_path, map_location='cuda'))
else:
    mwa_cnn.load_state_dict(torch.load(cnn_state_dict_path, map_location='cpu'))

model = TransformerWithCNN(model, mwa_cnn, config)
#model = fix_adapter_weights(model, adapter_path)
model = PeftModel.from_pretrained(model, model_path)
	
#print(model)
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入指令
data = load_dataset('json', data_files=data_path)
# 随机输出N个样本验证
soh_plist = []
soh_tlist = []
N = 30
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
    full_predicted = target_predicted.split("### Response:")[1].strip()
    soh_target = target.split("SOH is")[1].split("%")[0].strip()
    full_target = target.split("### Response:")[0].strip()
    soh_plist.append(float(soh_predicted))
    soh_tlist.append(float(soh_target))
    print(f"\nSample {i}:  {input_parms}")
    #print(f"Predicted SOH: {soh_predicted}%  vs  Target SOH: {soh_target}%")
    print(f"Predicted: {full_predicted}\nTarget: {full_target}")
    if (j == N-1):
        print("########Begining Output########")
        print(target_predicted)
        print("########End Output########")
        print("########Begining Target########")
        print(target)
        print("########End Target########")

# 计算RMSE
rmse = np.sqrt(sum((soh_plist[i] - soh_tlist[i]) ** 2 for i in range(N)) / N)
print(f"RMSE: {rmse}")

# 计算准确率
soh_plist = []
soh_tlist = []
accuracy = 0
for i in range(len(data['train'])):
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
    soh_predicted = target_predicted.split("SOH is")[1].split("%")[0].strip()
    soh_target = target.split("SOH is")[1].split("%")[0].strip()
    soh_plist.append(float(soh_predicted))
    soh_tlist.append(float(soh_target))

for i in range(len(soh_plist)):
    if (abs(soh_plist[i] - soh_tlist[i]) < 1):
        accuracy += 1
        continue
print(f'Accuracy: {accuracy/len(soh_plist)}')


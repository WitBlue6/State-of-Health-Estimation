from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from datasets import load_dataset
from peft import PeftModel
from random import randint
import torch
from utils import TransformerWithCNN, MWA_CNN

# 加载微调后的模型
base_model = "EleutherAI/pythia-31m"
model_path = "./olora/best"  # 
data_path = "./dataset/battery_dataset.json"
model_kwargs = {"torch_dtype": getattr(torch, "float16"), "device_map": "auto"}

model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

withCNN = 0
if withCNN:
	config = AutoConfig.from_pretrained(base_model, **model_kwargs)
	model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
	mwa_cnn = MWA_CNN(input_dim=64)
	model = TransformerWithCNN(model, mwa_cnn, config)
	model = PeftModel.from_pretrained(model, model_path)
	
print(model)
# 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 输入指令
data = load_dataset('json', data_files=data_path)
i = randint(1, len(data["train"]) - 1)
input_instruction = data['train'][i]['instruction']

# 编码输入
inputs = tokenizer(input_instruction, truncation=True, max_length=256, padding=False, return_tensors="pt")
# 同训练时处理
if (inputs["input_ids"][0, -1] != tokenizer.eos_token_id and inputs["input_ids"].shape[1] < 256):
	eos_tensor = torch.tensor([[tokenizer.eos_token_id]])
	attention_tensor = torch.tensor([[1]])
	# 拼接eos_token
	inputs["input_ids"] = torch.cat([inputs["input_ids"], eos_tensor], dim=1)
	inputs["attention_mask"] = torch.cat([inputs["attention_mask"], attention_tensor], dim=1)

output_tokens = model.generate(
	inputs["input_ids"],
	attention_mask=inputs["attention_mask"],
	max_new_tokens=100,
	do_sample=True,
	temperature=0.4,
	top_p=0.95,
	top_k=60,
	eos_token_id=tokenizer.eos_token_id,
	pad_token_id=tokenizer.pad_token_id,
	num_return_sequences=1,
	#repetition_penalty=0.2
)

target_predicted = tokenizer.decode(output_tokens[0], skip_special_tokens=True).strip()
print("########Begining Output########")
print(target_predicted)

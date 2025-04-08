# Copyright 2024-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#	  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
from typing import List, Optional
import numpy as np
import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, AutoConfig
import torch.nn as nn

from utils import *
from peft import (
	LoraConfig,
	get_peft_model,
)

# need to add this for macos
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"	 # 允许自动回退到 CPU

class MWA_Trainer(transformers.Trainer):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
	def compute_loss(self, model, inputs, return_outputs=False):
		#print(inputs)
		return model(**inputs).loss

# 自定义的损失函数，未用上
def compute_metrics(eval_pred):
	logits, labels = eval_pred
	#print(labels, logits)	# logits全是nan
	preds = np.argmax(logits, axis=-1)
	preds = preds.reshape(-1, 1)
	labels = labels.reshape(-1, 1)
	loss = ((preds - labels) ** 2).mean()  # 
	return {
		"eval_loss": loss,
		#"accuracy": accuracy_score(labels.flatten(), preds.flatten()),
	}
#PYTORCH_ENABLE_MPS_FALLBACK=1
def train(
	base_model: str = "",
	data_path: str = "",
	output_dir: str = "olora",
	batch_size: int = 16,
	num_epochs: int = 1,
	learning_rate: float = 3e-4,
	cutoff_len: int = 256,
	val_set_size: int = 16,
	quantize: bool = False,
	eval_step: int = 100,
	save_step: int = 100,
	device_map: str = "auto",
	lora_r: int = 32,
	lora_alpha: int = 16,
	lora_dropout: float = 0.05,
	lora_target_modules: List[str] = None,
	torch_dtype: str = "float16",
	init_lora_weights="olora",
	seed: Optional[int] = None,
):
	# Set device_map to the right place when enabling DDP.
	world_size = int(os.environ.get("WORLD_SIZE", 0)) or int(os.environ.get("PMI_SIZE", 0))
	if world_size > 1 and device_map != "cpu":
		from accelerate import Accelerator

		device_map = {"": Accelerator().process_index}
	# Set seed
	if seed is not None:
		set_seed(seed)
	model_kwargs = {"torch_dtype": getattr(torch, torch_dtype), "device_map": device_map}
	if quantize:
		model_kwargs["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.bfloat16,
			bnb_4bit_use_double_quant=True,
			bnb_4bit_quant_type="nf4",
		)
	#model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
	# add by lzh
	config = AutoConfig.from_pretrained(base_model, **model_kwargs)
	model = AutoModelForCausalLM.from_pretrained(base_model, config=config)
	mwa_cnn = MWA_CNN(input_dim=64)
	model = TransformerWithCNN(model, mwa_cnn, config)
	
	tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
	# For some tokenizer with no pad token like llama
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token

	def tokenize(prompt, target, add_eos_token=True):
		tokenize_prompt = tokenizer(
			prompt,
			truncation=True,
			max_length=cutoff_len,
			padding=False,
			return_tensors=None,
		)
		tokenize_target = tokenizer(
			target,
			truncation=True,
			max_length=128,
			padding=False,
			add_special_tokens=False,
		)
		input_ids = tokenize_prompt["input_ids"] + tokenize_target["input_ids"]
		labels = [-100] * len(tokenize_prompt["input_ids"]) + tokenize_target["input_ids"]
		if (
			input_ids[-1]!= tokenizer.eos_token_id
			and len(input_ids) < cutoff_len
			and add_eos_token
		):
			input_ids.append(tokenizer.eos_token_id)
			labels.append(tokenizer.eos_token_id)
		attention_mask = [1] * len(input_ids)
		labels = torch.tensor(labels)
		#result["labels"] = result["input_ids"].copy()
		# add by lzh
		return {
			"input_ids": torch.tensor(input_ids),
			"attention_mask": torch.tensor(attention_mask),
			"labels": labels,
		}


	def generate_and_tokenize_prompt(example):
		full_prompt, target = generate_prompt(example)
		tokenized_full_prompt = tokenize(full_prompt, target)
		return tokenized_full_prompt

	config = LoraConfig(
		r=lora_r,
		lora_alpha=lora_alpha,
		target_modules=lora_target_modules,
		lora_dropout=lora_dropout,
		bias="none",
		task_type="CAUSAL_LM",
		init_lora_weights=init_lora_weights,
	)
	model = get_peft_model(model, config)
	print(model)
	print(model.print_trainable_parameters())
	# add by lzh, adding MWA_CNN to model
	device = torch.device("mps")  # 针对 MPS 后端
	dtype = torch.float32  # 使用一致的数据类型
	model = model.to(device, dtype)
	
	#data_path = os.path.abspath(data_path)
	data = load_dataset("json", data_files=data_path)  # here changed by lzh, add "json" to specific the file type

	train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=42)
	train_data = train_val["train"].shuffle().map(generate_and_tokenize_prompt)
	val_data = train_val["test"].shuffle().map(generate_and_tokenize_prompt)
	loss_callback = LossRecorderCallback("logs")
	# --------------------Train-----------------
	trainer = transformers.Trainer(
		model=model,
		train_dataset=train_data,
		eval_dataset=val_data,
		#add by lzh
		#compute_metrics=compute_metrics,
		callbacks=[loss_callback],
		args=transformers.TrainingArguments(
			per_device_train_batch_size=batch_size,
			warmup_steps=50,
			num_train_epochs=num_epochs,
			learning_rate=learning_rate,
			logging_steps=100,
			optim="adamw_torch",
			eval_strategy="steps",
			save_strategy="steps",
			eval_steps=eval_step,
			save_steps=save_step,
			output_dir=output_dir,
			save_total_limit=3,
			load_best_model_at_end=True,
			ddp_find_unused_parameters=False if world_size > 1 else None,
			# add by lzh
			max_grad_norm=10,
			#fp16=False,
			label_names=["labels"],
		),
		data_collator=transformers.DataCollatorForSeq2Seq(
			tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
		),
	)
	
	trainer.train()
	# --------------------Save Model-----------------
	# 先删除已有文件
	output_best_dir = os.path.join(output_dir, "best")
	import shutil

	if os.path.exists(output_best_dir):  # 保证不会写串行
		shutil.rmtree(output_best_dir)
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	
	print("Save Best Model...")
	best_model = trainer.model

	# 实际上发现读取预测模型时权重名也有嵌套，虽然有warning但应该不影响
	trainer.save_model(output_best_dir)
	torch.save(best_model.base_model.cnn.state_dict(), os.path.join(output_best_dir, "cnn_model.pth"))
	print("Best model has been saved to {output_best_dir}")
def generate_prompt(example):
	full_prompt = f"""Following the Instruction below, give me your Response.
			### Instruction:
			{example["instruction"]}
			### Input:
			{example["input"]}
			### Response:
			"""
	target = example["output"]
	return full_prompt, target



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("--base_model", type=str, default="EleutherAI/pythia-31m")
	parser.add_argument("--data_path", type=str, default="./dataset/battery_dataset.json")	# "yahma/alpaca-cleaned"
	parser.add_argument("--output_dir", type=str, default="olora")
	parser.add_argument("--batch_size", type=int, default=8)
	parser.add_argument("--num_epochs", type=int, default=1.5) #0.2好像over fitting了
	parser.add_argument("--learning_rate", type=float, default=1e-4)
	parser.add_argument("--cutoff_len", type=int, default=256)
	parser.add_argument("--val_set_size", type=int, default=4)
	parser.add_argument("--quantize", action="store_true")
	parser.add_argument("--eval_step", type=int, default=100)
	parser.add_argument("--save_step", type=int, default=100)
	parser.add_argument("--device_map", type=str, default="auto")
	parser.add_argument("--lora_r", type=int, default=8)
	parser.add_argument("--lora_alpha", type=int, default=16)
	parser.add_argument("--lora_dropout", type=float, default=0.05)
	parser.add_argument("--lora_target_modules", type=str, default=["query_key_value", "dense"]) #, "dense", "dense_h_to_4h", "dense_4h_to_h"
	parser.add_argument("--torch_dtype", type=str, default="float16")
	parser.add_argument("--init_lora_weights", type=str, default="olora")
	parser.add_argument("--seed", type=int, default=None)

	args = parser.parse_args()

	train(
		base_model=args.base_model,
		data_path=args.data_path,
		output_dir=args.output_dir,
		batch_size=args.batch_size,
		num_epochs=args.num_epochs,
		learning_rate=args.learning_rate,
		cutoff_len=args.cutoff_len,
		val_set_size=args.val_set_size,
		quantize=args.quantize,
		eval_step=args.eval_step,
		save_step=args.save_step,
		device_map=args.device_map,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		lora_dropout=args.lora_dropout,
		lora_target_modules=args.lora_target_modules,
		torch_dtype=args.torch_dtype,
		init_lora_weights=args.init_lora_weights,
		seed=args.seed,
	)

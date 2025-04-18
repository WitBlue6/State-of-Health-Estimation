from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
import numpy as np
from utils import SOHPredictor
import os
import random

def set_random_seed(seed=42):
    """
    设置随机数种子以确保实验可重复性
    :param seed: 随机数种子，默认为42
    """
    random.seed(seed)  # 设置 Python 随机库种子
    np.random.seed(seed)  # 设置 numpy 随机库种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机数种子
    torch.cuda.manual_seed(seed)  # 设置 PyTorch GPU 随机数种子
    torch.cuda.manual_seed_all(seed)  # 如果有多张 GPU，设置所有 GPU 的随机种子
    torch.mps.manual_seed(seed)  # 设置 PyTorch MPS 随机数种子
    torch.backends.cudnn.deterministic = True  # 使得 GPU 使用确定性算法（可提高可复现性，但可能稍慢）
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 自动选择最优算法（用于保证复现性）

def Standardization(data):
    """
    Min-Max标准化
    """
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

def generate_features(model, tokenizer, prompts, device):
    """
    生成特征用于模型输入
    """
    features = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, max_length=256, return_tensors='pt', padding=False, truncation=True).to(device)
            # outputs = model.generate(
            #     **inputs, 
            #     max_new_tokens=80, 
            #     output_hidden_states=True, 
            #     return_dict_in_generate=True,
            #     eos_token_id=tokenizer.eos_token_id,
		    #     pad_token_id=tokenizer.pad_token_id,
            # )
            # 取最后一层的隐藏状态作为特征
            outputs = model(**inputs, output_hidden_states=True)
            last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            features.append(last_hidden_state.cpu().numpy())
    return np.vstack(features)

def generate_prompt(example):
	full_prompt = f"""
            ### Instruction:
			{example["instruction"]}
			### Input:
			{example["input"]}
			"""
	return {'prompt':full_prompt}

def load_prompts(data_path):
    """
    加载数据集
    """
    prompts = []
    data = load_dataset("json", data_files=data_path)['train']
    #prompts = list(map(generate_prompt, data))
    prompts = data.map(generate_prompt)['prompt']
    return prompts

def train(
    base_model: str = 'EleutherAI/pythia-1b',
    data_path: str = './dataset/1533B.json',
    output_path: str = './outputs',
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    normalize: bool = True,
    seed: int = 42
):
    """
    训练模型
    """
    # 设置随机数种子
    if seed is not None:
        set_random_seed(seed)
    # 设备选择
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("✅ Using MPS (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA (GPU {torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print("⚠️ Using CPU (no MPS or CUDA found)")
    dtype = torch.float32  # 使用一致的数据类型
    # 加载数据
    prompts = load_prompts(data_path)
    # 加载LLM
    print(f'Loading LLM from {base_model}...')
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # 生成特征
    print(f'Generating features for {len(prompts)} samples...')
    features = generate_features(model, tokenizer, prompts, device)
    if normalize:
        print('Normalizing Features...')
        features = Standardization(features)    
    input_dim = features.shape[1]
    print(f'Input Dim:{input_dim}')
    # 构建预测模型
    soh_predictor = SOHPredictor(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(soh_predictor.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # 训练
    for epoch in range(num_epochs):
        soh_predictor.train()
        total_loss = 0
        inputs = torch.tensor(features, dtype=dtype).to(device)
        for i in range(0, len(features), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            optimizer.zero_grad()
            outputs = soh_predictor(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()   
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(features)}")
    print("Training finished!")
    # 保存模型
    os.makedirs(output_path, exist_ok=True)
    torch.save(soh_predictor.state_dict(), os.path.join(output_path, "soh_predictor.pth"))
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default='EleutherAI/pythia-160m')
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--num_epochs", type=int, default=180)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--normalize", type=bool, default=True, help="是否标准化")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(**vars(args))
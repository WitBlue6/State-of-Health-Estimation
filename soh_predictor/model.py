import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression
from datasets import load_dataset
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from utils import SOHPredictor, CustomLoss
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
    特征级Z-score标准化
    """
    scaler = StandardScaler()
    # 确保输入是二维数组 [n_samples, n_features]
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    normalized_data = scaler.fit_transform(data)
    return scaler, normalized_data

def noise_adder(prompts, noise_level=5e-3):
    """"
    对数据添加噪声(数据增强)
    """
    def extract_features(prompt):
        """
        从prompt中提取键值对
        :param prompt: 输入的prompt字符串
        :return: 提取的键值对字典
        """
        import re
        pattern = r"([\w\s]+):([0-9\.]+)"
        matches = re.findall(pattern, prompt)
        feature_dict = {key.strip(): float(value) for key, value in matches}
        return feature_dict
    
    def generate_full_prompt(example):
        full_prompt = f"""
                ### Instruction:
                {example["instruction"]}
                ### Input:
                {example["input"]}
                ### Description:
                You need to extract the key features from the input. And notice the anomalies in the input.
                """
        return full_prompt

    data_list = list(map(extract_features, prompts))  # 得到了原始数据字典列表
    # 对数据进行处理,加白噪声干扰
    bad_datas = []
    # 得到数据的平均幅度值
    tim_voltage = [d['timer voltage'] for d in data_list]
    tim_current = [d['timer current'] for d in data_list]
    tim_temperature = [d['timer temperature'] for d in data_list]
    voltage = noise_level * (sum(tim_voltage) / len(tim_voltage))
    current = noise_level * (sum(tim_current) / len(tim_current))
    temperature = noise_level * (sum(tim_temperature) / len(tim_temperature))
    for i in range(len(data_list)):  
        for key in data_list[i]:
            if key in ['timer voltage', 'power voltage']: #, 'power voltage', 'control voltage', 'dsp voltage'
                data_list[i][key] += np.random.uniform(-voltage, voltage)  # 加白噪声
            elif key in ['timer current', 'power current']: #, 'power current', 'control current', 'dsp current'
                data_list[i][key] += np.random.uniform(-current, current)  # 加白噪声
            elif key in ['timer temperature', 'power temperature']: # , 'power temperature', 'control temperature', 'dsp temperature'
                data_list[i][key] += np.random.uniform(-temperature, temperature)  # 加白噪声
        bad_datas.append(data_list[i])
    prompts = []
    for row in bad_datas:
        instruction = f"You are a data feature extraction expert. Using the provided data, extract the features."
        input_text = (
            f"timer voltage:{row['timer voltage']}, "
            f"power voltage:{row['power voltage']}, "
            f"control voltage:{row['control voltage']}, "
            f"dsp voltage:{row['dsp voltage']}, "
            f"timer current:{row['timer current']}, "
            f"power current:{row['power current']}, "
            f"control current:{row['control current']}, "
            f"dsp current:{row['dsp current']}, "
            f"timer temperature:{row['timer temperature']}, "
            f"power temperature:{row['power temperature']}, "
            f"control temperature:{row['control temperature']}, "
            f"dsp temperature:{row['dsp temperature']},"
        )
        prompts.append({
            "instruction": instruction,
            "input": input_text,
        })
    prompts = list(map(generate_full_prompt, prompts))
    return prompts

def generate_features(model, tokenizer, prompts, device):
    """
    生成特征用于模型输入
    """
    features = []
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, max_length=256, return_tensors='pt', padding=False, truncation=True).to(device)
            outputs = model(**inputs, output_hidden_states=True)
            # # 取最后一层的隐藏状态作为特征
            # last_hidden_state = outputs.hidden_states[-1][:, -1, :]
            # 取最后三层的平均pooling
            # hiddens = torch.stack(outputs.hidden_states[-3:])  # [3, batch, seq_len, dim]
            # diff_features = hiddens[-1] - hiddens[0]  # 捕捉变化趋势
            # abs_features = torch.abs(hiddens).mean(dim=0)  # 绝对值特征
            # pooled = torch.cat([diff_features.mean(dim=1), abs_features.mean(dim=1)], dim=1)
            hiddens = torch.stack(outputs.hidden_states[-3:])  # [3, batch, seq_len, dim]
            weighted = hiddens * torch.tensor([0.2, 0.3, 0.5]).view(3,1,1,1).to(device)
            pooled = weighted.sum(dim=(0,2))  # [batch, dim]
            features.append(pooled.cpu().numpy())
            
    return np.vstack(features)

def generate_prompt(example):
	full_prompt = f"""
            ### Instruction:
			{example["instruction"]}
			### Input:
			{example["input"]}
            ### Description:
            You need to extract the key features from the input. And notice the anomalies in the input.
			"""
	return {'prompt':full_prompt}

def load_prompts(data_path, add_noise=False):
    """
    加载数据集
    """
    prompts = []
    data = load_dataset("json", data_files=data_path)['train']
    if add_noise:
        print('Adding Noise!!!')
        prompts = noise_adder(data['input'])
    else:
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
    seed: int = 42,
    val_ratio: float = 0.2,
    add_noise: bool = False,
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
    prompts = load_prompts(data_path, add_noise=add_noise)
    # 加载LLM
    print(f'Loading LLM from {base_model}...')
    model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # 生成特征
    print(f'Generating features for {len(prompts)} samples...')
    features = generate_features(model, tokenizer, prompts, device)

    if normalize:
        print('Normalizing Features...')
        feature_scaler, features = Standardization(features)  
        # 将标准化器保存到文件（便于后续使用）
        os.makedirs(output_path, exist_ok=True)
        import joblib
        joblib.dump(feature_scaler, os.path.join(output_path, "feature_scaler.pkl")) 
    
    np.set_printoptions(threshold=np.inf, linewidth=200, precision=6)
    # 1. 统计特征重要性
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor()
    y_dummy = np.random.rand(len(features))  # 或用真实标签
    rf.fit(features, y_dummy)
    print("特征重要性Top20:\n", np.sort(rf.feature_importances_)[-20:])
    
    # 2. 可视化
    from matplotlib import pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.hist(features.ravel(), bins=50)
    plt.title("Feature Distribution")
    
    plt.subplot(122)
    plt.imshow(features[:100].T, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("The first 100 samples Hotmap")
    plt.savefig(os.path.join(output_path, "feature_distribution.png"))
    #plt.show()

    input_dim = features.shape[1]
    print(f'Input Dim:{input_dim}')

    # 构建预测模型
    soh_predictor = SOHPredictor(input_dim).to(device)
    criterion = nn.MSELoss()#CustomLoss(alpha=0.5)
    optimizer = torch.optim.Adam(
        soh_predictor.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs // 4,  # 每1/4训练周期重置学习率
        eta_min=learning_rate / 100  # 最小学习率
    )

    # 训练集划分
    indices = np.random.permutation(len(features))
    split_idx = int(len(features) * (1 - val_ratio))
    train_features = features[indices[:split_idx]]
    val_features = features[indices[split_idx:]]
    train_inputs = torch.tensor(train_features, dtype=dtype).to(device)
    val_inputs = torch.tensor(val_features, dtype=dtype).to(device)
    
    # 训练
    best_val_loss = float('inf')
    best_epoch = 0
    patience = 20
    no_improve = 0

    for epoch in range(num_epochs):
        soh_predictor.train()
        train_loss = 0.0
        
        for i in range(0, len(train_features), batch_size):
            batch_inputs = train_inputs[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = soh_predictor(batch_inputs)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                soh_predictor.parameters(), 
                max_norm=1.0
            )
            optimizer.step()
            train_loss += loss.item() * len(batch_inputs)   
        
        train_loss /= len(train_features)
        scheduler.step()

        # 验证
        soh_predictor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for i in range(0, len(val_features), batch_size):
                batch_inputs = val_inputs[i:i+batch_size]
                outputs = soh_predictor(batch_inputs)
                val_loss += criterion(outputs, batch_inputs).item() * len(batch_inputs)
        val_loss /= len(val_features)
        
        # 打印数据
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, LR: {current_lr}')
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            # 保存最佳模型
            torch.save(soh_predictor.state_dict(), os.path.join(output_path, "best_model.pth"))
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    print("Training finished!")
    print('Best Val Loss:', best_val_loss, 'at epoch:', best_epoch)
    # 保存模型
    os.makedirs(output_path, exist_ok=True) 
    torch.save(soh_predictor.state_dict(), os.path.join(output_path, "soh_predictor.pth"))
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default='EleutherAI/pythia-160m') # EleutherAI/pythia-160m
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--num_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--normalize", type=bool, default=True, help="是否标准化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--add_noise", type=bool, default=True, help="加噪声数据增强")

    args = parser.parse_args()
    train(**vars(args))
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
from utils import SOHPredictor, CustomLoss, AnomalyProcessor
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

def noise_adder(prompts, add_noise=True, noise_rate=0.4):
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
    
    data_list = list(map(extract_features, prompts))  # 得到了原始数据字典列表
    if not add_noise:
        return data_list
    anomaly_processor = AnomalyProcessor()
    #bad_datas = anomaly_processor.noise_adder(data_list, noise_level=0.05, noise_rate=0.15)
    bad_datas = anomaly_processor.data_random_process(data_list, noise_level=0.05, rate=0.4, factor=2)
    return bad_datas

def load_prompts(data_path, add_noise=False):
    """
    加载数据集
    """
    prompts = []
    data = load_dataset("json", data_files=data_path)['train']
    data_list = noise_adder(data['input'], add_noise=add_noise)
    return data_list
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
    resume_from_checkpoint: str = None,
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
    data_list = load_prompts(data_path, add_noise=add_noise)
    
    # 处理数据  将data_list从字典格式转化为np.array，并使用np.vstack将其堆叠成一个二维数组
    features = []
    for entry in data_list:
        feature_vector = np.array(list(entry.values()))
        features.append(feature_vector)
    features = np.vstack(features)
    print(f'Feature Shape:{features.shape}')
    
    # 特征标准化
    if normalize:
        print('Normalizing Features...')
        feature_scaler, features = Standardization(features)  
        # 将标准化器保存到文件（便于后续使用）
        os.makedirs(output_path, exist_ok=True)
        import joblib
        joblib.dump(feature_scaler, os.path.join(output_path, "feature_scalerwithoutLLM.pkl")) 
    
    print(features[0])

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
    # 恢复训练
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        soh_predictor.load_state_dict(checkpoint)

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
            torch.save(soh_predictor.state_dict(), os.path.join(output_path, "best_modelwithoutLLM.pth"))
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    print("Training finished!")
    print('Best Val Loss:', best_val_loss, 'at epoch:', best_epoch)
    # 保存模型
    os.makedirs(output_path, exist_ok=True) 
    torch.save(soh_predictor.state_dict(), os.path.join(output_path, "soh_predictorwithoutLLM.pth"))
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, default='EleutherAI/pythia-160m') # EleutherAI/pythia-160m
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--num_epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--normalize", type=bool, default=True, help="是否标准化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--add_noise", type=bool, default=True, help="加噪声数据增强")
    parser.add_argument("--resume_from_checkpoint", type=str, default="../outputs/best_modelwithoutLLM.pth", help="从检查点恢复训练")

    args = parser.parse_args()
    train(**vars(args))
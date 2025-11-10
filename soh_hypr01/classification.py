import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from utils import *
import pandas as pd
from collections import defaultdict

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
    对每一维特征进行单独归一化（Z-score标准化）
    参数：
    - data: shape=(n_samples, 40) 的二维numpy数组
    返回：
    - scaled_data: 同shape，归一化后的数据
    - scalers: 每个特征对应的 StandardScaler，便于逆归一化或后续使用
    """
    data = np.array(data)
    n_samples, n_features = data.shape

    scaled_data = np.zeros_like(data)
    scalers = []
    indices = []
    for i in range(n_features):
        scaler = StandardScaler()
        feature_i = data[:, i].reshape(-1, 1)
        if np.std(feature_i) == 0:
            print(f"Feature {i} has zero variance. Skipping scaling.")
            scaled_data[:, i] = feature_i.flatten()
            continue
        scaled_feature_i = scaler.fit_transform(feature_i)
        scaled_data[:, i] = scaled_feature_i.flatten()
        scalers.append(scaler)
        indices.append(i)

    return scalers, scaled_data, indices


def Data_Loader(data_path, add_noise, label_map):
    # 加载data_path根目录下所有的txt文件
    data_list = []
    label_list = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                new_file = load_data(file_path, add_noise=add_noise)
                data_list.extend(new_file)
                # 为每个样本添加标签
                try:
                    label = label_map[file.split('.')[0]]
                    label_list.extend([label] * len(new_file))
                except Exception as e:
                    print(f"Error keyname Found {file.split('.')[0]}: {e}")
    return data_list, label_list

def train(
    data_path: str = './dataset/1533B.json',
    output_path: str = './outputs',
    num_epochs: int = 100,
    batch_size: int = 32,
    num_modules: int = 40,
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
    label_map = {
        '无异常':0,
        '舵机1故障':1,
        '舵机2故障':2,
        '舵机3故障':3,
        '舵机4故障':4,
        '惯组X轴故障':5,
        '惯组Y轴故障':6,
        '惯组Z轴故障':7,
        '电源故障':8,
        '北斗故障':9,
    }
    data_list, label_list = Data_Loader(data_path, add_noise=add_noise, label_map=label_map)
    
    features = []
    for entry in data_list:
        feature_vector = np.array(list(entry.values()))
        features.append(feature_vector)
    features = np.vstack(features)
    # 将标签转换为numpy数组
    labels = np.array(label_list)

    # 特征标准化
    if normalize:
        print('Normalizing Features...')
        feature_scaler, features, indices = Standardization(features)  
        # 将标准化器保存到文件（便于后续使用）
        os.makedirs(output_path, exist_ok=True)
        import joblib
        joblib.dump((feature_scaler, indices), os.path.join(output_path, "scaler_cls.pkl")) 
    

    input_dim = features.shape[1]
    print(f'Input Dim:{input_dim}')

    # 构建预测模型
    model = ClassificationModel(input_dim, num_modules=num_modules).to(device)
    classification_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
    )
    # 恢复训练
    if resume_from_checkpoint is not None and os.path.exists(resume_from_checkpoint):
        print(f"Resuming training from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint)

    # 训练集划分
    indices = np.random.permutation(len(labels))
    features = features[indices]
    labels = labels[indices]
    split_idx = int(len(labels) * (1 - val_ratio))
    train_features = features[:split_idx]
    train_labels = labels[:split_idx]
    val_features = features[split_idx:]
    val_labels = labels[split_idx:]

    # 将标签转换为PyTorch张量
    train_inputs = torch.tensor(train_features, dtype=dtype).to(device)
    train_labels = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_inputs = torch.tensor(val_features, dtype=dtype).to(device)
    val_labels = torch.tensor(val_labels, dtype=torch.long).to(device)
    
    # 训练
    best_val_loss = float('inf')
    best_epoch = 0
    
    patience = 20
    no_improve = 0


    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for i in range(0, len(train_labels), batch_size):
            batch_inputs = train_inputs[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            optimizer.zero_grad()
            logits = model(batch_inputs)

            total_loss = classification_criterion(logits, batch_labels)

            total_loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                max_norm=1.0
            )
            optimizer.step()
            train_loss += total_loss.item() * len(batch_inputs)   
        
        train_loss /= len(train_labels)


        # 验证
        # 记录每个类别的预测情况
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(val_labels), batch_size):
                batch_inputs = val_inputs[i:i+batch_size]
                batch_labels = val_labels[i:i+batch_size]

                logits = model(batch_inputs)
                _, predicted = torch.max(logits.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                total_loss = classification_criterion(logits, batch_labels)

                val_loss += total_loss.item() * len(batch_labels)
                # 每个样本统计正确与否
                for label, pred in zip(batch_labels, predicted):
                    class_total[label.item()] += 1
                    if pred.item() == label.item():
                        class_correct[label.item()] += 1

        accuracy = correct / total
        val_loss /= len(val_labels)
        scheduler.step(val_loss)
        
        # 打印数据
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, Accuracy: {accuracy}, LR: {current_lr}')
        print("\nPer-Class Accuracy:")
        for cls in sorted(class_total.keys()):
            acc = class_correct[cls] / class_total[cls]
            print(f"  Class {cls}: {acc:.2%} ({class_correct[cls]}/{class_total[cls]})")
        # 早停策略
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve = 0
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(output_path, "best_classification.pth"))
        # else:
        #     no_improve += 1
        #     if no_improve >= patience:
        #         print(f"Early stopping at epoch {epoch+1}")
        #         break
    print("Training finished!")
    print('Best Val Loss:', best_val_loss, 'at epoch:', best_epoch)
    # 保存模型
    os.makedirs(output_path, exist_ok=True) 
    torch.save(model.state_dict(), os.path.join(output_path, "soh_classification.pth"))
    print(f"Model saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='./dataset/')
    parser.add_argument("--output_path", type=str, default='./outputs')
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_modules", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--normalize", type=bool, default=True, help="是否标准化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--add_noise", type=bool, default=False, help="加噪声数据增强")
    parser.add_argument("--resume_from_checkpoint", type=str, default="../outputs/best_classification.pth", help="从检查点恢复训练")

    args = parser.parse_args()
    train(**vars(args))

from sklearn.cluster import KMeans
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from model import load_prompts, generate_features, Standardization
import re

def extract_features(prompt):
    """
    从prompt中提取键值对
    :param prompt: 输入的prompt字符串
    :return: 提取的键值对字典
    """
    pattern = r"([\w\s]+):([0-9\.]+)"
    matches = re.findall(pattern, prompt)
    feature_dict = {key.strip(): float(value) for key, value in matches}
    return feature_dict

def data_process(data_path):
    """
    对数据进行加噪声干扰,得到包含异常数据和正常数据的合集
    :param data_path: 处理后的json路径
    :return: 处理后的prompts
    """
    # Load Dataset
    prompts = load_prompts(data_path)
    data_list = list(map(extract_features, prompts))  # 得到了原始数据字典列表
    # 对数据进行处理,加白噪声干扰
    dlen = len(data_list)  # print(dlen) ## 2080
    nlen = dlen // 3  # 取1/3的数据作为噪声数据  ##  693
    bad_data = []
    init_data = []
    # 得到数据的平均幅度值
    tim_voltage = [d['timer voltage'] for d in data_list]
    tim_current = [d['timer current'] for d in data_list]
    tim_temperature = [d['timer temperature'] for d in data_list]
    voltage = sum(tim_voltage) / len(tim_voltage)
    current = sum(tim_current) / len(tim_current)
    temperature = sum(tim_temperature) / len(tim_temperature)
    # 加噪声处理
    for i in range(dlen):  
        if i < nlen:
            for key in data_list[i]:
                if key in ['timer voltage']: #, 'power voltage', 'control voltage', 'dsp voltage'
                    data_list[i][key] += np.random.normal(0, 0.005*voltage)  # 加白噪声
                elif key in ['timer current']: #, 'power current', 'control current', 'dsp current'
                    data_list[i][key] += np.random.normal(0, 0.005*current)  # 加白噪声
                elif key in ['timer temperature']: # , 'power temperature', 'control temperature', 'dsp temperature'
                    data_list[i][key] += np.random.normal(0, 0.005*temperature)  # 加白噪声
            bad_data.append(data_list[i])
        else:
            init_data.append(data_list[i])
    print(f'Successfully Transformed {nlen} Data to Bad Data!') 
    data_list = bad_data + init_data
    def generate_full_prompt(example):
        full_prompt = f"""
                ### Instruction:
                {example["instruction"]}
                ### Input:
                {example["input"]}
			"""
        return full_prompt
    def generate_prompt(examples):
        prompts = []
        for row in examples:
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
        return prompts
    prompts = generate_prompt(data_list)
    prompts = list(map(generate_full_prompt, prompts))
    return prompts, len(bad_data)


def calculate_soh(features, center, noraml, threshold=0.8):
    """
    根据每个样本与聚类中心的距离计算SOH
    :param features: 输入的特征数据
    :param center: 聚类中心
    :param threshold: 判断异常的阈值
    :return: SOH值,异常标签
    """
    # 计算每个样本到聚类中心的欧氏距离
    distances = np.linalg.norm(features - center, axis=1)
    mean_d = np.mean(distances)
    # 归一化距离，距离越大，SOH值越低
    d = (mean_d - noraml[0]) / noraml[1]
    soh = torch.clamp(torch.tensor((1 - d) * 100), min=0, max=100).item()
    # 判断异常：距离超过阈值的点认为是异常
    flags = soh < threshold * 100  
    return soh, flags

def model_detect(
        llm_path, 
        soh_path, 
        data_path, 
        threshold=0.8, 
        num_normal_samples=8,
        num_detect_samples=8, 
        output_path='./outputs',
        normalize=True,
        clustering_method='kmeans'  # 聚类方法选择
    ):
    """
    加载模型,对数据进行预测
    :param model_path: 模型路径
    :param data_list: 要处理的数据列表
    :return: 预测结果"""
    # Get device
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
    print(f"Loading data from {data_path}...")
    prompts, nlen = data_process(data_path) # 前nlen个样本是bad data
    # LLM获取特征
    model = AutoModelForCausalLM.from_pretrained(llm_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    print(f"Generating features for {len(prompts)} samples...")
    features = generate_features(model, tokenizer, prompts, device)
    if normalize:
        print('Normalizing Features...')
        features = Standardization(features) 
    
    # 从正常样本中选取一些来计算聚类中心
    normal_features = features[nlen:]  # 这里假设后面的样本是正常的
    normal_features = normal_features[np.random.choice(len(normal_features), num_normal_samples, replace=False)]
    
    # 使用 K-means 聚类方法来计算聚类中心
    kmeans = KMeans(n_clusters=1, random_state=42)
    kmeans.fit(normal_features)  # 获取聚类中心
    cluster_center = kmeans.cluster_centers_[0]  # 获取聚类中心

    # 先计算num_normal_samples个样本的平均距离
    distances = np.linalg.norm(normal_features - cluster_center, axis=1)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    print(f'Average Distance: {avg_distance}, Standard Deviation: {std_distance}')

    # 计算所有样本的SOH
    print(f"Calculating SOH for {len(features)} samples...")
    results = []
    flags = []
    for i in range(0, len(features), num_detect_samples):
        batch_features = features[i:i+num_detect_samples]
        result, flag = calculate_soh(batch_features, cluster_center, [avg_distance, std_distance], threshold)
        results.append(result)
        flags.append(flag)

    # 绘制 SOH 曲线
    plt.figure(figsize=(12, 8))
    plt.plot(results, label='SOH')
    plt.axhline(y=threshold * 100, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=nlen // num_detect_samples, color='g', linestyle='--', label='Bad Data')
    plt.xlabel('Batch Index')
    plt.ylabel('SOH Value')
    plt.title('SOH Prediction based on Clustering')
    #plt.ylim([40, 103])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'soh_prediction_distance.png'))
    plt.show()
    print('📊Results has been saved')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default='EleutherAI/pythia-160m')
    parser.add_argument("--soh_path", type=str, default='./outputs/soh_predictor.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_normal_samples", type=int, default=16, help="多少个正常样本用于计算聚类中心")
    parser.add_argument("--num_detect_samples", type=int, default=8, help="以多少个样本为一组进行预测，提高鲁棒性")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=True)

    args = parser.parse_args()
    model_detect(**vars(args))

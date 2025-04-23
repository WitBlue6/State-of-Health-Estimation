import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datasets import load_dataset
import torch
import torch.nn as nn
from model import load_prompts, generate_features, Standardization, set_random_seed
from utils import SOHPredictor
import re
import numpy as np
import matplotlib.pyplot as plt

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
    voltage = 0.08 * (sum(tim_voltage) / len(tim_voltage))
    current = 0.08 * (sum(tim_current) / len(tim_current))
    temperature = 0.08 * (sum(tim_temperature) / len(tim_temperature))
    print(voltage, current, temperature)
    # 加噪声处理
    for i in range(dlen): 
        if i < nlen:
            for key in data_list[i]:
                if key in ['timer voltage']: #, 'power voltage', 'control voltage', 'dsp voltage'
                    data_list[i][key] += np.random.normal(0, voltage)  # 加白噪声
                elif key in ['timer current']: #, 'power current', 'control current', 'dsp current'
                    data_list[i][key] += np.random.normal(0, current)  # 加白噪声
                elif key in ['timer temperature']: # , 'power temperature', 'control temperature', 'dsp temperature'
                    data_list[i][key] += np.random.normal(0, temperature)  # 加白噪声
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
                ### Description:
                You need to extract the key features from the input. And notice the anomalies in the input.
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
    prompts1 = generate_prompt(data_list)
    prompts = list(map(generate_full_prompt, prompts1))
    return prompts, len(bad_data), prompts1

def detect_soh(soh_predictor, features, device, threshold=0.8, normal_loss=[0, 1]):
    """
    检测异常
    :param soh_predictor: 训练好的SOH预测模型
    :param features: 输入特征
    :param threshold: 阈值
    :param normal_loss: 正常时的平均损失和标准差
    :return: SOH值和异常标志
    """
    soh_predictor.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).to(device)
        outputs = soh_predictor(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(outputs, inputs)
        mean_loss = torch.mean(loss).item()
        normalized_loss = (mean_loss - normal_loss[0]) / normal_loss[1]
        soh = torch.clamp(torch.tensor((1 - normalized_loss) * 100), min=0, max=100).item()
        is_anomaly = soh < (threshold * 100)
        if is_anomaly:
            print(f'😁Anomaly detected! SOH: {soh}')
            print(f'Loss: {mean_loss}')
    return soh

def soh_filter(data, filter='ma', window_size=5, alpha=0.2, sigma=1.0):
    """
    对输出SOH进行滤波,提高模型鲁棒性
    """
    if filter == 'ma':  #移动平均滤波
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    elif filter == 'ewma':  #指数加权移动平均滤波
        return np.convolve(data, np.array([alpha**i for i in range(window_size)]), mode='same')
    # 高斯滤波
    elif filter == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=sigma)
    else:
        print(f'‼️Warning: Unknown filter type {filter}, using no filter!!!')
        return data

def prompt_with_soh(data_list, soh, anomaly, num_detect_samples=32, file_add=True, output_json_path='./dataset'):
    """
    对prompt加入SOH信息, file_add=True时将新内容添加到文件,否则重新覆盖  
    """
    # 由于soh是num_detect_samples个样本为一组进行预测的，先要还原成原来大小
    # 对于soh中的某一个样本，重复num_detect_samples次
    soh = [soh_value for soh_value in soh for _ in range(num_detect_samples)]
    anomaly = [anomaly_value for anomaly_value in anomaly for _ in range(num_detect_samples)]
    for i, data in enumerate(data_list):
        if anomaly[i]:
            data["output"] = f"The SOH is {soh[i]}%. There is something wrong with timer. Please check it."
        else:
            data["output"] = f"The SOH is {soh[i]}%. Everything is normal."
    import json
    if file_add:
        with open(output_json_path, "a", encoding="utf-8") as f:
            for entry in data_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f'✅Successfully Added {len(data_list)} Samples to {output_json_path}!')
    else:
        with open(output_json_path, "w", encoding="utf-8") as f:
            for entry in data_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f'✅Successfully Overwritten {len(data_list)} Samples to {output_json_path}!')
    return data_list

def model_detect(
        llm_path, 
        soh_path, 
        data_path, 
        threshold=0.05, 
        num_normal_samples=8,
        num_detect_samples=8, 
        output_path='./outputs',
        normalize=True,
        filter=True,
        seed=42
    ):
    """
    加载模型,对数据进行预测
    :param model_path: 模型路径
    :param data_list: 要处理的数据列表
    :return: 预测结果"""
    if seed is not None:
        set_random_seed(seed)
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
    prompts, nlen, data_list = data_process(data_path) # 前nlen个样本是bad data
    # LLM获取特征
    model = AutoModelForCausalLM.from_pretrained(llm_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    print(f"Generating features for {len(prompts)} samples...")
    features = generate_features(model, tokenizer, prompts, device)
    
    if normalize:
        print("Loading feature scaler...")
        scaler_path = os.path.join(os.path.dirname(soh_path), "feature_scaler.pkl")
        import joblib
        feature_scaler = joblib.load(scaler_path)
        features = feature_scaler.transform(features)


    # 加载SOH模型权重
    checkpoint = torch.load(soh_path, map_location=device)
    soh_predictor = SOHPredictor(input_dim=features.shape[1]).to(device)
    soh_predictor.load_state_dict(checkpoint)

    # 检测异常
    print(f"⚠️Detecting {len(features)} samples...")
    # 先根据正常工作时的样本得到正常时的loss大小
    normal_features = features[nlen:]
    # 从正常样本中随机选取num_detect_samples个样本作为正常样本
    normal_features = normal_features[np.random.choice(len(normal_features), num_normal_samples, replace=False)]
    print(f'😋Using Normal Samples: {len(normal_features)}')
    soh_predictor.eval()
    with torch.no_grad():
        inputs = torch.tensor(normal_features, dtype=torch.float32).to(device)
        outputs = soh_predictor(inputs)
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(outputs, inputs)
        normal_loss = [torch.mean(loss).item(), torch.std(loss).item()]
    print(f'🤓Calculating Normal loss: {normal_loss}')

    # 按num_detect_samples个一组对样本进行SOH预测
    results = []
    print('Calculating SOH...')
    #features = features[nlen:]
    for i in range(0, len(features), num_detect_samples):
        batch_features = features[i:i+num_detect_samples]
        result = detect_soh(soh_predictor, batch_features, device, threshold, normal_loss)
        results.append(result)
    # 输出滤波
    if filter:
        results = soh_filter(results, filter='gaussian', sigma=1.0)
    flags = [True if result < (threshold * 100) else False for result in results]
    # 绘出结果图
    plt.figure(figsize=(6, 4))
    plt.plot(results, label='SOH')
    plt.axhline(y=threshold * 100, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=nlen//num_detect_samples, color='g', linestyle='--', label='Bad Data')
    anomaly_indices = [i for i, flag in enumerate(flags) if flag]
    if anomaly_indices:
        plt.scatter(
            anomaly_indices, 
            [results[i] for i in anomaly_indices],
            color='red', marker='x', label='Detected Anomalies'
        )
    plt.xlabel('Batch Index')
    plt.ylabel('SOH Value')
    plt.title('SOH Prediction')
    plt.ylim([0, 102])
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'soh_prediction.png'))
    plt.show()
    print('📊Results has been saved')
    # 重新写dataset
    print(f'Writing Dataset to {os.path.dirname(data_path)}...')
    prompt_with_soh(
        data_list=data_list, 
        soh=results, anomaly=flags, 
        num_detect_samples=num_detect_samples, 
        file_add=True,
        output_json_path=os.path.join(os.path.dirname(data_path), '1533B_with_soh.json')
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default='EleutherAI/pythia-160m')
    parser.add_argument("--soh_path", type=str, default='./outputs/best_model.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--threshold", type=float, default=0.93)
    parser.add_argument("--num_normal_samples", type=int, default=32, help="多少个正常样本用于求解正常时的loss")
    parser.add_argument("--num_detect_samples", type=int, default=32, help="以多少个样本为一组进行预测，提高鲁棒性")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--filter", type=bool, default=True, help="是否进行输出滤波")
    parser.add_argument("--seed", type=int, default=10011)  #999

    args = parser.parse_args()
    model_detect(**vars(args))
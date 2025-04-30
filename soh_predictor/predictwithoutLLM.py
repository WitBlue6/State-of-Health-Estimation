import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datasets import load_dataset
import torch
import torch.nn as nn
from model import load_prompts, generate_features, Standardization, set_random_seed
from utils import SOHPredictor, AnomalyProcessor
import re
import numpy as np
import matplotlib.pyplot as plt
import random

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
    nlen = dlen // 5  # 取1/3的数据作为噪声数据  ##  693
    bad_data = []
    # 得到数据的平均幅度值
    tim_voltage = [d['timer voltage'] for d in data_list]
    tim_current = [d['timer current'] for d in data_list]
    tim_temperature = [d['timer temperature'] for d in data_list]
    voltage = 0.3 * (sum(tim_voltage) / len(tim_voltage))
    current = 0.3 * (sum(tim_current) / len(tim_current))
    temperature = 0.3 * (sum(tim_temperature) / len(tim_temperature))
    print(voltage, current, temperature)
    # 对1/3*dlen到2/3*dlen的数据进行加白噪声干扰
    key_noise_map = {
        #'timer voltage': voltage,
        #'timer current': current,
        #'timer temperature': temperature,
        'power voltage': voltage,
        'power current': current,
        'power temperature': temperature,
        #'control voltage': voltage,
        #'control current': current,
        #'control temperature': temperature,
        #'dsp voltage': voltage,
        #'dsp current': current, 
        #'dsp temperature': temperature,
    }

    # for i in range(dlen): 
    #     if i >= nlen and i < 2 * nlen:
    #         available_keys = [key for key in data_list[i].keys() if key in key_noise_map]
    #         if not available_keys:
    #             continue
    #         num_keys_to_select = random.randint(1, len(available_keys)//3)
    #         selected_keys = random.sample(available_keys, num_keys_to_select)
    #         #selected_keys = available_keys
    #         for key in selected_keys:
    #             data_list[i][key] += np.random.normal(0, key_noise_map[key])
    #         bad_data.append(data_list[i])
    process = AnomalyProcessor()
    for i in range(dlen): 
        if i >= nlen and i < 2 * nlen:
            #data_list[i] = process.missing_adder([data_list[i]], missing_rate=0.001)[0]
            #data_list[i] = process.outlier_adder([data_list[i]], outlier_rate=0.001, factor=2.0)[0]
            data_list[i] = process.data_random_process([data_list[i]], noise_level=0.15, rate=0.01, factor=2.0)[0]
            bad_data.append(data_list[i])
    print(f'Successfully Transformed {nlen} Data to Bad Data!') 
    return data_list, nlen, len(bad_data)


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
        print('Using Gaussian Filter')
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(data, sigma=sigma)
    else:
        print(f'‼️Warning: Unknown filter type {filter}, using no filter!!!')
        return data


def model_detect(
        soh_path, 
        data_path, 
        threshold=0.05, 
        num_normal_samples=8,
        num_detect_samples=8, 
        output_path='./outputs',
        normalize=True,
        filter=True,
        auto_threshold=False,
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
    data_list, nlen, nlen2 = data_process(data_path) # 前nlen个样本是bad data
    
    # 处理数据  将data_list从字典格式转化为np.array，并使用np.vstack将其堆叠成一个二维数组
    features = []
    for entry in data_list:
        feature_vector = np.array(list(entry.values()))
        features.append(feature_vector)
    features = np.vstack(features)
    print(f'Feature Shape:{features.shape}')
    
    if normalize:
        print("Loading feature scaler...")
        scaler_path = os.path.join(os.path.dirname(soh_path), "feature_scalerwithoutLLM.pkl")
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
    normal_features = np.concatenate((features[:nlen], features[nlen + nlen2:]), axis=0)
    # 从正常样本中随机选取num_detect_samples个样本作为正常样本
    normal_features = normal_features[np.random.choice(len(normal_features), num_normal_samples, replace=False)]
    print(f'😋Using Normal Samples: {len(normal_features)}')

    # 加载SOH检测器
    from utils import SOHDetector
    soh_detector = SOHDetector(soh_predictor, normal_features, device, threshold=threshold, auto_threshold=auto_threshold)
    
    # 按num_detect_samples个一组对样本进行SOH预测
    results = []
    thres_list = []
    for i in range(num_detect_samples, len(features)):
        batch_features = features[i-num_detect_samples:i]
        result, thres = soh_detector.detect_soh(batch_features)
        results.append(result)
        thres_list.append(thres)
    # 输出滤波
    if filter:
        results = soh_filter(results, filter='gaussian', sigma=5.0)
    flags = [True if results[i] < (thres_list[i] * 100) else False for i in range(len(results))]
    # 绘出结果图
    plt.figure(figsize=(6, 4))
    plt.plot(results, color='b', label='SOH')
    plt.plot([thres * 100 for thres in thres_list], color='black', alpha=0.5, label='Threshold')
    plt.axvline(x=nlen, color='g', linestyle='--', label='Bad Data Start')
    plt.axvline(x=(nlen+nlen2), color='g', linestyle='-.', label='Bad Data End')
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
    plt.savefig(os.path.join(output_path, 'soh_predictionwithoutLLM.png'))
    plt.show()
    print('📊Results has been saved')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--soh_path", type=str, default='./outputs/best_modelwithoutLLM.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_normal_samples", type=int, default=32, help="多少个正常样本用于求解正常时的loss")
    parser.add_argument("--num_detect_samples", type=int, default=32, help="以多少个样本为一组进行预测，提高鲁棒性")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--filter", type=bool, default=True, help="是否进行输出滤波")
    parser.add_argument("--auto_threshold", type=bool, default=True, help="自适应阈值")
    parser.add_argument("--seed", type=int, default=42)  #999

    args = parser.parse_args()
    model_detect(**vars(args))
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像网站
from transformers import AutoModelForCausalLM, AutoTokenizer
#from fastapi import FastAPI
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn
from model import load_data, Standardization, set_random_seed
from utils import *
import re
import numpy as np
import matplotlib.pyplot as plt
import random


def generate_text(model, tokenizer, prompt: str, device="cpu"):
    '''调用本地大模型生成文本'''    
    system_prompt = "你是一个电子设备日志分析专家，请根据日志信息，给出最简单精炼日志摘要，并给出具体的建议执行操作"
    #full_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"],
        max_new_tokens=512,
        do_sample=True,
        temperature=0.45,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = outputs[0][inputs["input_ids"].shape[1]:]  # 只拿生成的新token
    generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    return generated_text


def data_process(data_path):
    """
    对数据进行加噪声干扰,得到包含异常数据和正常数据的合集
    :param data_path: 处理后的json路径
    :return: 处理后的prompts
    """
    # Load Dataset
    data_list = load_data(data_path, add_noise=False)
    dlen = len(data_list)
    # 加载异常数据
    bad_path = os.path.join(os.path.dirname(data_path), '惯组X轴故障.txt')
    bad_data = load_data(bad_path, add_noise=False)
    nlen = dlen // 5
    nlen2 = 0
    for i in range(dlen):
        if i >= nlen and i < 3 * nlen:
            data_list[i] = bad_data[i - nlen]
            nlen2 += 1
    print(f'Successfully Transformed {len(bad_data)} Data to Bad Data!') 
    return data_list, nlen, nlen2



def model_detect(
        llm_path,
        soh_path, 
        cls_path,
        data_path, 
        threshold=0.05, 
        num_normal_samples=8,
        num_detect_samples=8, 
        num_modules=10,
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
    
    # if normalize:
    #     print("Loading feature scaler...")
    #     scaler_path = os.path.join(os.path.dirname(soh_path), "feature_scalerwithoutLLM.pkl")
    #     import joblib
    #     feature_scaler, indices = joblib.load(scaler_path)
    #     #_, features = Standardization(features)
    #     for scaler, i in zip(feature_scaler, indices):
    #         features[:, i] = scaler.transform(features[:, i].reshape(-1, 1)).flatten()

    # 加载大模型
    torch.cuda.empty_cache()
    print(f"Loading LLM from {llm_path}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path, trust_remote_code=True).to(device)

    print('Loading SOH Model...')
    # 加载SOH模型权重
    checkpoint = torch.load(soh_path, map_location=device)
    soh_predictor = SOHPredictor(input_dim=features.shape[1]).to(device)
    soh_predictor.load_state_dict(checkpoint)

    checkpoint = torch.load(cls_path, map_location=device)
    cls_model = ClassificationModel(input_dim=features.shape[1], num_modules=num_modules).to(device)
    cls_model.load_state_dict(checkpoint)
    
    print(f"✅Loading Model Finished!")


    # 检测异常
    print(f"⚠️Detecting {len(features)} samples...")
    # 先根据正常工作时的样本得到正常时的loss大小
    normal_features = np.concatenate((features[:nlen], features[nlen + nlen2:]), axis=0)
    # 从正常样本中随机选取num_detect_samples个样本作为正常样本
    inx = np.random.randint(0, nlen-num_normal_samples)
    #normal_features = normal_features[inx:inx + num_normal_samples]
    normal_features = normal_features[np.random.choice(len(normal_features), num_normal_samples, replace=False)]
    print(f'😋Using Normal Samples: {len(normal_features)}')
    print(f'Normal Samples: {normal_features.shape}')
    # 加载SOH检测器
    from utils import SOHDetector
    soh_detector = SOHDetector(
            soh_predictor, 
            cls_model, 
            normal_features, 
            device, 
            threshold=threshold, 
            auto_threshold=auto_threshold, 
            normalize=normalize,
            sclar_soh_path='./outputs/scaler_soh.pkl',
            sclar_cls_path='./outputs/scaler_cls.pkl',
            filter=filter,
            print_log=False,
    )
    columns = [
        'Motor1', 'Motor2', 'Motor3', 'Motor4', 'Motor5', 'Motor6', 'Motor7', 'Motor8',
        'Motor9', 'Motor10', 'Motor11', 'Motor12', 'Motor13', 'Motor14', 'Motor15', 'Motor16',
        'Motor17', 'Motor18', 'Motor19', 'Motor20', 'Motor21', 'Motor22', 'Motor23', 'Motor24',
        'Accelx', 'Accely', 'Accelz', 'AngAcx', 'AngAcy', 'AngAcz', 'Eulerx', 'Eulery', 'Eulerz',
        'Voltage', 'Current', 'Power', 'Battery',
        'GPS_longitude', 'GPS_latitude', 'GPS_altitude'
    ]
    # 按num_detect_samples个一组对样本进行SOH预测
    results = []
    thres_list = []
    warning_list = []
    import time
    start_time = time.time()
    for i in range(len(features)-num_detect_samples):
        batch_features = features[i:i+num_detect_samples]
        result, thres, warning_type = soh_detector.detect_soh(batch_features, key_map=columns)
        results.append(result)
        thres_list.append(thres)
        warning_list.append(warning_type)

    # 统计模块故障数量
    total = soh_detector.count[0] + soh_detector.count[1] + soh_detector.count[2] + soh_detector.count[3] + soh_detector.count[4] + soh_detector.count[5] + soh_detector.count[6] + soh_detector.count[7] + soh_detector.count[8]
    print(f'✅Detecting Finished, Total {len(results)} samples!')
    print(f'Error Counter')
    if total:
        print(f'Motor1:   {soh_detector.count[0]}  Motor1/Total: {soh_detector.count[0] / total * 100}%')
        print(f'Motor2:   {soh_detector.count[1]}  Motor2/Total: {soh_detector.count[1] / total * 100}%')
        print(f'Motor3:   {soh_detector.count[2]}  Motor3/Total: {soh_detector.count[2] / total * 100}%')
        print(f'Motor4:   {soh_detector.count[3]}  Motor4/Total: {soh_detector.count[3] / total * 100}%')
        print(f'EulerX:   {soh_detector.count[4]}  EulerX/Total: {soh_detector.count[4] / total * 100}%')
        print(f'EulerY:   {soh_detector.count[5]}  EulerY/Total: {soh_detector.count[5] / total * 100}%')
        print(f'EulerZ:   {soh_detector.count[6]}  EulerZ/Total: {soh_detector.count[6] / total * 100}%')
        print(f'Power:    {soh_detector.count[7]}  Power/Total: {soh_detector.count[7] / total * 100}%')
        print(f'Beidou:   {soh_detector.count[8]}  Beidou/Total: {soh_detector.count[8] / total * 100}%')
    # 保存日志结果
    with open(os.path.join(output_path, 'log.txt'), 'w') as f:
        for log in soh_detector.log_info:
            f.write(log + '\n')
    #llm_output = LLM_answer(compress_logs(soh_detector.log_info), model='gpt')
    llm_output = generate_text(llm_model, tokenizer, prompt=compress_logs(soh_detector.log_info), device=device)
    print(f'Recommendation: \n{llm_output}')
    end_time = time.time()
    total_time = end_time - start_time
    print(f'⏰Model Process Total Time: {total_time:3f}s')
    # 输出滤波
    #if filter:
        #results = soh_filter(results, filter='gaussian', sigma=2.0)
    flags = [True if results[i] < (thres_list[i] * 100) else False for i in range(len(results))]
    # 绘出结果图
    plt.figure(figsize=(10, 6))
    plt.plot(results, color='b', label='SOH')
    plt.plot([thres * 100 for thres in thres_list], color='black', alpha=0.5, label='Threshold')
    plt.axvline(x=nlen-num_detect_samples, color='g', linestyle='--', label='Bad Data Start')
    plt.axvline(x=(nlen+nlen2), color='g', linestyle='-.', label='Bad Data End')
    anomaly_indices = [i for i, flag in enumerate(flags) if flag]
    # 预警类型输出
    # 故障预警
    if anomaly_indices:
        plt.scatter(
            anomaly_indices, 
            [results[i] for i in anomaly_indices],
            color='red', marker='x', label='Detected Anomalies'
        )
    # 下滑预警
    decrease_flags = [True if warning == 2 else False for warning in warning_list]
    decrease_indices = [i for i, flag in enumerate(decrease_flags) if flag]
    if decrease_indices:
        plt.scatter(
            decrease_indices,
            [results[i] for i in decrease_indices],
            color='orange', marker='^', label='Decrease Warning'
        )
    # 长期临界预警
    critical_flags = [True if warning == 3 else False for warning in warning_list]
    critical_indices = [i for i, flag in enumerate(critical_flags) if flag]
    if critical_indices:
        plt.scatter(
            critical_indices,
            [results[i] for i in critical_indices],
            color='green', marker='*', label='Long-Term Critical Warning'
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

    parser.add_argument("--llm_path", type=str, default='Qwen/Qwen1.5-1.8B-Chat')
    parser.add_argument("--soh_path", type=str, default='./outputs/best_sohmodel.pth')
    parser.add_argument("--cls_path", type=str, default='./outputs/best_classification.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/无异常.txt')
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--num_normal_samples", type=int, default=32, help="多少个正常样本用于求解正常时的loss")
    parser.add_argument("--num_detect_samples", type=int, default=32, help="以多少个样本为一组进行预测，提高鲁棒性")
    parser.add_argument("--num_modules", type=int, default=10, help="模块数")
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--filter", type=bool, default=True, help="是否进行输出滤波")
    parser.add_argument("--auto_threshold", type=bool, default=True, help="自适应阈值")
    parser.add_argument("--seed", type=int, default=42)  #999

    args = parser.parse_args()
    model_detect(**vars(args))
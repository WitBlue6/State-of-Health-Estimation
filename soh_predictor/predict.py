from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import torch.nn as nn
from model import load_prompts, generate_features, Standardization
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
    for i in range(dlen):  
        if i < nlen:
            for key in data_list[i]:
                if key == 'timer voltage' or key == 'power voltage' or key == 'control voltage' or key == 'dsp voltage':
                    data_list[i][key] += np.random.normal(0, 1)  # 加白噪声
                elif key == 'timer current' or key == 'power current' or key == 'control current' or key == 'dsp current':
                    data_list[i][key] += np.random.normal(0, 0.04)  # 加白噪声
                elif key == 'timer temperature' or key == 'power temperature' or key == 'control temperature' or key == 'dsp temperature':
                    data_list[i][key] += np.random.normal(0, 3)  # 加白噪声
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
        if soh < threshold:
            print(f'😁Anomaly detected! SOH: {soh}')
            print(f'Loss: {mean_loss}')
    return soh, False if soh >= threshold else True


def model_detect(
        llm_path, 
        soh_path, 
        data_path, 
        threshold=0.05, 
        num_detect_samples=8, 
        output_path='./outputs',
        normalize=True
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
    # 加载SOH模型权重
    soh_predictor = SOHPredictor(input_dim=features.shape[1]).to(device)
    soh_predictor.load_state_dict(torch.load(soh_path))

    # 检测异常
    print(f"⚠️Detecting {len(features)} samples...")
    # 先根据正常工作时的样本得到正常时的loss大小
    normal_features = features[nlen:]
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
    flags = []
    for i in range(0, len(features), num_detect_samples):
        batch_features = features[i:i+num_detect_samples]
        result, flag = detect_soh(soh_predictor, batch_features, device, threshold, normal_loss)
        results.append(result)
        flags.append(flag)
    # 绘出结果图
    plt.figure(figsize=(12, 8))
    plt.plot(results, label='SOH')
    plt.axhline(y=threshold * 100, color='r', linestyle='--', label='Threshold')
    plt.axvline(x=nlen//num_detect_samples, color='g', linestyle='--', label='Bad Data')
    plt.xlabel('Batch Index')
    plt.ylabel('SOH Value')
    plt.title('SOH Prediction')
    plt.legend()
    plt.grid(True)
    plt.savefig('./outputs/soh_prediction.png')
    plt.show()
    print('📊Results has been saved')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm_path", type=str, default='EleutherAI/pythia-160m')
    parser.add_argument("--soh_path", type=str, default='./outputs/soh_predictor.pth')
    parser.add_argument("--data_path", type=str, default='./dataset/1533B.json')
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--num_detect_samples", type=int, default=8)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--normalize", type=bool, default=True)

    args = parser.parse_args()
    model_detect(**vars(args))

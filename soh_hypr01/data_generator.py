'''帮我写一个Python异常数据生成方法，要求如下：
能够生成1到10维数据，具有时序性，其中一维数据生成使用基准函数为fx = random(t) + w1 * a^t + w2 * sin(2*pi*f1*t + phi1) + w3 * cos(2*pi*f2*t + phi2)
具体例子，轻微故障数据生成为F(t)=random(-0.5，0.5)+0.9^(0.0001∙t)+10∙sin⁡(2π∙1500∙t+π/6)+3∙cos⁡(2π∙1000∙t+π/4)
根据一维的数据，从而拓展到10维。
其次，有工作模式的选择：待机、轻度工作、全工作，对应的可生成数据维度为1-3、3-7、7-10
方法的输入包含三个：一个是工作模式、一个是要生成的数据维度、另一个是生成数据长度
方法的输出就是生成对应长度的时序数据
'''

import numpy as np
import time
from datetime import datetime
import os
import pandas as pd


def generate_anomaly_data(mode: str, dim: int, length: int, seed: int = None):
    """
    生成异常时序数据
    
    参数:
        mode (str): 工作模式，可选 ["standby", "light", "full"]
        dim (int): 数据维度
        length (int): 数据长度
        seed (int): 随机种子，便于复现
        
    返回:
        np.ndarray: shape = (length, dim) 的时序数据
    """
    if seed is not None:
        np.random.seed(seed)

    # 不同模式对应的维度范围
    mode_ranges = {
        "standby": (1, 3),
        "light": (3, 7),
        "full": (7, 10)
    }

    if mode not in mode_ranges:
        raise ValueError("mode 必须为 'standby', 'light' 或 'full'")
    min_dim, max_dim = mode_ranges[mode]

    if not (min_dim <= dim <= max_dim):
        raise ValueError(f"{mode} 模式下维度必须在 {min_dim} 到 {max_dim} 之间")

    t = np.arange(length)

    data = []
    for d in range(dim):
        # 每一维的参数都随机生成
        w1 = np.random.uniform(0.5, 2.0)   # 指数项权重
        a = np.random.uniform(0.99, 0.9999) # 衰减系数
        w2 = np.random.uniform(1.0, 10.0)  # 正弦权重
        f1 = np.random.uniform(100, 2000)  # 正弦频率
        phi1 = np.random.uniform(0, 2*np.pi)
        w3 = np.random.uniform(1.0, 10.0)  # 余弦权重
        f2 = np.random.uniform(100, 2000)  # 余弦频率
        phi2 = np.random.uniform(0, 2*np.pi)

        # 生成一维序列
        series = np.random.uniform(-0.5, 0.5, size=length) \
                 + w1 * (a ** t) \
                 + w2 * np.sin(2*np.pi*f1*t + phi1) \
                 + w3 * np.cos(2*np.pi*f2*t + phi2)
        data.append(series)

    return np.stack(data, axis=1)  # shape = (length, dim)

def stream_anomaly_data(mode: str, dim: int, length: int, seed: int = None, interval: float = 1.0):
    """
    按秒流式生成异常时序数据，每次返回一行 (1, dim)
    
    参数:
        mode (str): 工作模式，可选 ["standby", "light", "full"]
        dim (int): 数据维度
        length (int): 数据长度
        seed (int): 随机种子
        interval (float): 每次返回的时间间隔（秒）
        
    产出:
        np.ndarray: shape = (1, dim)
    """
    if seed is not None:
        np.random.seed(seed)

    mode_ranges = {
        "standby": (1, 3),
        "light": (3, 7),
        "full": (7, 10)
    }

    if mode not in mode_ranges:
        raise ValueError("mode 必须为 'standby', 'light' 或 'full'")
    min_dim, max_dim = mode_ranges[mode]

    if not (min_dim <= dim <= max_dim):
        raise ValueError(f"{mode} 模式下维度必须在 {min_dim} 到 {max_dim} 之间")

    # 固定每一维的参数，不要每次都随机
    w1 = np.random.uniform(2.0, 3.0, size=dim)
    a = np.random.uniform(0.9, 0.9999, size=dim)
    w2 = np.random.uniform(0.2, 0.5, size=dim)
    f1 = np.random.uniform(1000, 3000, size=dim)
    phi1 = np.random.uniform(0, np.pi / 2, size=dim)
    w3 = np.random.uniform(0.2, 0.5, size=dim)
    f2 = np.random.uniform(1000, 3000, size=dim)
    phi2 = np.random.uniform(0, np.pi / 2, size=dim)
    w4 = np.random.uniform(0.2, 0.5, size=dim)
    f3 = np.random.uniform(1000, 3000, size=dim)
    phi3 = np.random.uniform(0, np.pi / 2, size=dim)
    
    for t in range(length):
        row = []
        for d in range(dim):
            value = np.random.uniform(-0.25, 0.25) \
                    + w1[d] * (a[d] ** t) \
                    + w2[d] * np.sin(2*np.pi*f1[d]*t + phi1[d]) \
                    + w3[d] * np.cos(2*np.pi*f2[d]*t + phi2[d]) \
                    + w4[d] * np.cos(2*np.pi*f3[d]*t + phi3[d])
            row.append(value)
        yield np.array(row).reshape(1, -1)
        time.sleep(interval)  # 模拟每秒返回一次

def stream_data_out(mode: str, output_dim: int, length: int, interval: float):
    """
    加载数据集
    """
    data_path = "./dataset/无异常.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    columns = [
        'Motor1', 'Motor2', 'Motor3', 'Motor4', 'Motor5', 'Motor6', 'Motor7', 'Motor8',
        'Motor9', 'Motor10', 'Motor11', 'Motor12', 'Motor13', 'Motor14', 'Motor15', 'Motor16',
        'Motor17', 'Motor18', 'Motor19', 'Motor20', 'Motor21', 'Motor22', 'Motor23', 'Motor24',
        'Accelx', 'Accely', 'Accelz', 'AngAcx', 'AngAcy', 'AngAcz', 'Eulerx', 'Eulery', 'Eulerz',
        'Voltage', 'Current', 'Power', 'Battery',
        'GPS_longitude', 'GPS_latitude', 'GPS_altitude'
    ]
    data_pd = pd.read_csv(data_path, header=None, names=columns)
    # 不同模式对应的维度范围
    mode_ranges = {
        "standby": (1, 3),
        "light": (3, 7),
        "full": (7, 10)
    }
    if mode not in mode_ranges:
        raise ValueError("mode 必须为 'standby', 'light' 或 'full'")
    min_dim, max_dim = mode_ranges[mode]

    if not (min_dim <= output_dim <= max_dim):
        raise ValueError(f"{mode} 模式下维度必须在 {min_dim} 到 {max_dim} 之间")
    # 确定实际输出的维度
    total_dim = len(columns)
    if output_dim is None:
        output_dim = total_dim
    elif output_dim > total_dim:
        raise ValueError(f"输出维度 {output_dim} 不能超过原始数据维度 {total_dim}")
    
    # 确定实际输出的长度
    total_length = len(data_pd)
    if length is None:
        length = total_length
    elif length > total_length:
        raise ValueError(f"输出长度 {length} 不能超过原始数据长度 {total_length}")
    
    # 随机选择要输出的维度（如果需要减少维度）
    if output_dim < total_dim:
        selected_dims = np.random.choice(total_dim, output_dim, replace=False)
    else:
        selected_dims = np.arange(total_dim)

    # 逐行处理并添加噪声
    for t in range(length):
        # 获取当前行（循环读取，如果length超过数据长度则从头开始）
        row_idx = t % total_length
        row = data_pd.iloc[row_idx, selected_dims].to_numpy()
        
        # 添加高斯噪声
        noise = np.random.normal(0, 0.1, size=row.shape)
        noisy_row = row + noise
        
        yield noisy_row.reshape(1, -1)
        time.sleep(interval)

def noise_data_out(mode: str, output_dim: int, length: int):
    """
    加载数据集
    """
    data_path = "./dataset/无异常.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    columns = [
        'Motor1', 'Motor2', 'Motor3', 'Motor4', 'Motor5', 'Motor6', 'Motor7', 'Motor8',
        'Motor9', 'Motor10', 'Motor11', 'Motor12', 'Motor13', 'Motor14', 'Motor15', 'Motor16',
        'Motor17', 'Motor18', 'Motor19', 'Motor20', 'Motor21', 'Motor22', 'Motor23', 'Motor24',
        'Accelx', 'Accely', 'Accelz', 'AngAcx', 'AngAcy', 'AngAcz', 'Eulerx', 'Eulery', 'Eulerz',
        'Voltage', 'Current', 'Power', 'Battery',
        'GPS_longitude', 'GPS_latitude', 'GPS_altitude'
    ]
    data_pd = pd.read_csv(data_path, header=None, names=columns)
    # 不同模式对应的维度范围
    mode_ranges = {
        "standby": (1, 3),
        "light": (3, 7),
        "full": (7, 10)
    }
    if mode not in mode_ranges:
        raise ValueError("mode 必须为 'standby', 'light' 或 'full'")
    min_dim, max_dim = mode_ranges[mode]

    if not (min_dim <= output_dim <= max_dim):
        raise ValueError(f"{mode} 模式下维度必须在 {min_dim} 到 {max_dim} 之间")
    # 确定实际输出的维度
    total_dim = len(columns)
    if output_dim is None:
        output_dim = total_dim
    elif output_dim > total_dim:
        raise ValueError(f"输出维度 {output_dim} 不能超过原始数据维度 {total_dim}")
    
    # 确定实际输出的长度
    total_length = len(data_pd)
    if length is None:
        length = total_length
    elif length > total_length:
        raise ValueError(f"输出长度 {length} 不能超过原始数据长度 {total_length}")
    
    # 随机选择要输出的维度（如果需要减少维度）
    if output_dim < total_dim:
        selected_dims = np.random.choice(total_dim, output_dim, replace=False)
    else:
        selected_dims = np.arange(total_dim)

    data_out = []
    # 逐行处理并添加噪声
    for t in range(length):
        # 获取当前行（循环读取，如果length超过数据长度则从头开始）
        row_idx = t % total_length
        row = data_pd.iloc[row_idx, selected_dims].to_numpy()
        
        # 添加高斯噪声
        noise = np.random.normal(0, 0.1, size=row.shape)
        noisy_row = row + noise
        
        data_out.append(noisy_row)
    return np.array(data_out)

if __name__ == "__main__":
    # # 生成“轻度工作”模式，5维，长度1000
    # ts_data = generate_anomaly_data("light", 5, 1000, seed=42)
    # print(ts_data.shape)  # (1000, 5)
    # print(type(ts_data))

    # for row in stream_anomaly_data("light", 5, 10, seed=42, interval=1):
    #     info = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>{row}'
    #     print(info)

    # for row in stream_data_out("full", 10, 10, interval=1):
    #     print(row)
    data_out = noise_data_out("full", 10, 10)
    print(data_out.shape)
    print(data_out)
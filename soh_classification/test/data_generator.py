'''帮我写一个Python异常数据生成方法，要求如下：
能够生成1到10维数据，具有时序性，其中一维数据生成使用基准函数为fx = random(t) + w1 * a^t + w2 * sin(2*pi*f1*t + phi1) + w3 * cos(2*pi*f2*t + phi2)
具体例子，轻微故障数据生成为F(t)=random(-0.5，0.5)+0.9^(0.0001∙t)+10∙sin⁡(2π∙1500∙t+π/6)+3∙cos⁡(2π∙1000∙t+π/4)
根据一维的数据，从而拓展到10维。
其次，有工作模式的选择：待机、轻度工作、全工作，对应的可生成数据维度为1-3、3-7、7-10
方法的输入包含三个：一个是工作模式、一个是要生成的数据维度、另一个是生成数据长度
方法的输出就是生成对应长度的时序数据
'''

import numpy as np

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


if __name__ == "__main__":
    # 生成“轻度工作”模式，5维，长度1000
    ts_data = generate_anomaly_data("light", 5, 1000, seed=42)
    print(ts_data.shape)  # (1000, 5)
    print(ts_data[:5])    # 打印前几行


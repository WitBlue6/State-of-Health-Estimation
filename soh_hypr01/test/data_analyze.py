import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
columns = [
        'Motor1_vol', 'Motor1_cur', 'Motor1_temp', 'Motor1_bat', 'Motor1_p', 'Motor1_n', 'Motor2', 'Motor8',
        'Motor9', 'Motor10', 'Motor11', 'Motor12', 'Motor13', 'Motor14', 'Motor15', 'Motor16',
        'Motor17', 'Motor18', 'Motor19', 'Motor20', 'Motor21', 'Motor22', 'Motor23', 'Motor24',
        'Accelx', 'Accely', 'Accelz', 'AngAcx', 'AngAcy', 'AngAcz', 'Eulerx', 'Eulery', 'Eulerz',
        'Voltage', 'Current', 'Power', 'Battery',
        'GPS_longitude', 'GPS_latitude', 'GPS_altitude'
]
from model import load_data, Standardization

def compute_fft(data_matrix):
    fft_vals = np.fft.fft(data_matrix.T, axis=1)  # 每行是一列特征的时序
    fft_mag = np.abs(fft_vals)
    return np.array(fft_mag)

def A():
    data_pd = pd.DataFrame(load_data('./dataset/无异常.txt', add_noise=False))
    normal_data = data_pd.iloc[:, 0:6].values
    #data_pd = pd.read_csv('./dataset/舵机2故障.txt', header=None, names=columns)
    data_pd = pd.DataFrame(load_data('./dataset/舵机1故障.txt', add_noise=False))
    abnormal_data = data_pd.iloc[:, 0:6].values
    
    # 分别做 FFT
    fft_normal = compute_fft(normal_data)
    fft_anomalous = compute_fft(abnormal_data)

    # 计算平均谱
    mean_normal = np.mean(fft_normal, axis=0)
    mean_anomalous = np.mean(fft_anomalous, axis=0)

    # # 画图对比
    # plt.figure(figsize=(10, 6))
    # plt.plot(mean_normal, label='Normal', color='green')
    # plt.plot(mean_anomalous, label='Anomalous', color='red')
    # plt.title("Average FFT Magnitude")
    # plt.xlabel("Frequency Bin")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.figure()
    # plt.plot(fft_normal[0, :100], label='Normal', color='green')# 2和3不一样
    # plt.plot(fft_anomalous[0, :100], label='Anomalous', color='red')
    # plt.title("FFT Magnitude")
    # plt.xlabel("Frequency Bin")
    # plt.legend()
    # plt.show()
    # 1. 均值和标准差对比
    normal_mean = np.mean(normal_data, axis=0)
    abnormal_mean = np.mean(abnormal_data, axis=0)

    normal_std = np.std(normal_data, axis=0)
    abnormal_std = np.std(abnormal_data, axis=0)

    # 打印均值和标准差对比
    print("Normal Data Mean:", normal_mean)
    print("Abnormal Data Mean:", abnormal_mean)
    print("Normal Data Standard Deviation:", normal_std)
    print("Abnormal Data Standard Deviation:", abnormal_std)

    # 2. 数据分布对比 - 绘制直方图
    plt.figure(figsize=(10, 6))

    for i in range(normal_data.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.hist(normal_data[:, i], bins=30, alpha=0.5, label='Normal', color='b')
        plt.hist(abnormal_data[:, i], bins=30, alpha=0.5, label='Abnormal', color='r')
        plt.title(columns[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 3. 计算相关性矩阵
    normal_corr = np.corrcoef(normal_data.T)  # 转置后计算列之间的相关性
    abnormal_corr = np.corrcoef(abnormal_data.T)

    # 绘制相关性热图
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(normal_corr, annot=True, cmap='coolwarm', xticklabels=columns[:6], yticklabels=columns[:6])
    plt.title('Normal Data Correlation')

    plt.subplot(1, 2, 2)
    sns.heatmap(abnormal_corr, annot=True, cmap='coolwarm', xticklabels=columns[:6], yticklabels=columns[:6])
    plt.title('Abnormal Data Correlation')

    plt.tight_layout()
    plt.show()

    # 4. 用滑动窗口计算均值和方差 - 用于比对
    window_size = 3  # 滑动窗口的大小，可以调整

    # 计算正常数据的滑动窗口均值和方差
    normal_rolling_mean = pd.DataFrame(normal_data).rolling(window=window_size).mean()
    normal_rolling_std = pd.DataFrame(normal_data).rolling(window=window_size).std()

    # 计算异常数据的滑动窗口均值和方差
    abnormal_rolling_mean = pd.DataFrame(abnormal_data).rolling(window=window_size).mean()
    abnormal_rolling_std = pd.DataFrame(abnormal_data).rolling(window=window_size).std()

    # 绘制滑动窗口均值对比
    plt.figure(figsize=(12, 6))

    for i in range(normal_data.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.plot(normal_rolling_mean[i], label='Normal Rolling Mean', color='b')
        plt.plot(abnormal_rolling_mean[i], label='Abnormal Rolling Mean', color='r')
        plt.title(columns[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

    # 绘制滑动窗口方差对比
    plt.figure(figsize=(12, 6))

    for i in range(normal_data.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.plot(normal_rolling_std[i], label='Normal Rolling Std', color='b')
        plt.plot(abnormal_rolling_std[i], label='Abnormal Rolling Std', color='r')
        plt.title(columns[i])
        plt.legend()

    plt.tight_layout()
    plt.show()

def calc_norm():
    data_list = load_data('./dataset/无异常.txt', add_noise=False)
    # 处理数据  将data_list从字典格式转化为np.array，并使用np.vstack将其堆叠成一个二维数组
    features = []
    for entry in data_list:
        feature_vector = np.array(list(entry.values()))
        features.append(feature_vector)
    features = np.vstack(features)
    feature_scaler, features = Standardization(features)  
    A()
if __name__ == '__main__':
    # 读取CSV文件
    #data_pd = pd.read_csv('./dataset/无异常.txt', header=None, names=columns)
    calc_norm()

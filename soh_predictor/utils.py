import torch.nn as nn
import torch
import numpy as np
import random

class AnomalyProcessor:
    '''
    对数据进行加噪声、加缺失、加变换干扰
    '''
    def __init__(self, data_list=None):
        self.data_list = data_list
        self.key_name =[
            'timer voltage',
            'timer current',
            'timer temperature',
            #'power voltage',
            #'power current',
            #'power temperature',
            #'control voltage',
            #'control current',
            #'control temperature',
            #'dsp voltage',
            #'dsp current',
            #'dsp temperature',
        ]
    def noise_adder(self, data_list, noise_level=0.15, noise_rate=0.3, **kwargs):
        """
        对数据进行加噪声干扰
        """
        #print('Adding Noise with noise level:', noise_level)
        # 得到数据的平均幅度值
        tim_voltage = [d['timer voltage'] for d in data_list]
        tim_current = [d['timer current'] for d in data_list]
        tim_temperature = [d['timer temperature'] for d in data_list]
        voltage = noise_level * (sum(tim_voltage) / len(tim_voltage))
        current = noise_level * (sum(tim_current) / len(tim_current))
        temperature = noise_level * (sum(tim_temperature) / len(tim_temperature))
        key_noise_map = {
            'timer voltage': voltage,
            'timer current': current,
            'timer temperature': temperature,
            'power voltage': voltage,
            'power current': current,
            'power temperature': temperature,
            'control voltage': voltage,
            'control current': current,
            'control temperature': temperature,
            'dsp voltage': voltage,
            'dsp current': current, 
            'dsp temperature': temperature,
        }
        for i in range(len(data_list)):
            available_keys = [key for key in data_list[i].keys() if key in key_noise_map]
            if not available_keys:
                continue
            if random.random() > noise_rate:
                continue
            num_keys_to_select = random.randint(1, 3)
            #selected_keys = random.sample(available_keys, num_keys_to_select)
            selected_keys = available_keys
            for key in selected_keys:
                data_list[i][key] += np.random.normal(0, key_noise_map[key])
        return data_list
    
    def missing_adder(self, data_list, missing_rate=1.0, **kwargs):
        """
        对数据进行加缺失干扰
        """
        #print('Adding Missing with missing rate:', missing_rate)
        for data in data_list:
            for key in data:
                if random.random() < missing_rate:
                    data[key] = 0.0
        return data_list
    
    def data_transform(self, data_list, transform_rate=1.0, **kwargs):
        """
        对数据进行加变换干扰
        """
        #print('Adding Transform with transform rate:', transform_rate)
        for data in data_list:
            for key in data:
                if random.random() < transform_rate and key in self.key_name:
                    if random.random() < 0.5:
                        data[key] *= random.uniform(0.8, 1.2)  # 缩放
                    else:
                        data[key] += random.uniform(-0.5, 0.5)  # 平移
        return data_list

    def outlier_adder(self, data_list, outlier_rate=1.0, factor=2.0, **kwargs):
        """
        对数据进行加异常值干扰
        """
        #print('Adding Outlier with outlier rate:', outlier_rate)
        for data in data_list:
            for key in data:
                if random.random() < outlier_rate:
                    data[key] *= random.choice([-factor, factor])  # 异常值
        return data_list
    
    def data_random_process(self, data_list, noise_level=0.15, rate=0.25, factor=5.0, **kwargs):
        """
        对数据随机进行加噪声、加缺失、加变换干扰
        """
        for data in data_list:
            if random.random() > rate:
                continue
            # 随机选择一个数据增强函数
            augment_func = random.choice([
                self.noise_adder,
                #self.missing_adder,
                self.data_transform,
                self.outlier_adder
            ])
            # 应用增强函数
            data = augment_func([data], noise_level=noise_level, factor=factor)[0]
        return data_list
    def key_mapper(self, key_map, **kwargs):
        """
        对数据进行映射,实现选取特定的模块故障"""
        pass
    

class SOHDetector:
    '''
    自适应阈值的SOH检测器
    :param soh_predictor: 用于预测SOH的模型
    :param normal_features: 正常工作时的特征数据
    :param device: 设备
    :param threshold: 初始阈值
    :param auto_threshold: 是否启用自适应阈值
    :param pid_params: PID控制器参数
    '''
    def __init__(self, soh_predictor, normal_features, device, threshold=0.8, auto_threshold=False, pid_params=None):
        self.soh_predictor = soh_predictor
        self.loss_fn = nn.MSELoss(reduction='none')
        self.normal_features = normal_features
        self.device = device
        self.threshold = threshold
        self.auto_threshold = auto_threshold
        self.threshold_alpha = 0.95
        self.normal_loss = None
        # 用于存储历史样本，更新阈值
        self.remembered_features = []
        self.remembered_soh = []
        self.remembered_num = 32
        # PID 控制器参数
        self.pid_params = pid_params if pid_params else {'Kp': 0.01, 'Ki': 0.000015, 'Kd': 0.2}
        self.prev_error = 0
        self.integral = 0
        # 计算初始样本分布
        self.calculate_normal_loss()

    def detect_soh(self, features, normal_loss=None, alpha=0.1):
        """
        检测异常
        """
        if normal_loss is None:
            normal_loss = self.normal_loss
        self.soh_predictor.eval()
        with torch.no_grad():   
            inputs = torch.tensor(features, dtype=torch.float32).to(self.device)
            outputs = self.soh_predictor(inputs)
            loss = self.loss_fn(outputs, inputs)
            mean_loss = torch.mean(loss).item()
            #print(mean_loss)
            normalized_loss = abs(mean_loss - normal_loss[0]) / normal_loss[1]
            #soh = torch.clamp(torch.tensor((1 - normalized_loss) * 100), min=0, max=100).item()
            soh = 100 * np.exp(-normalized_loss * alpha)
            is_anomaly = soh < (self.threshold * 100)
            if is_anomaly:
                print(f'😁Anomaly detected! SOH: {soh}  Threshold:{self.threshold * 100}')
                print(f'Loss: {mean_loss}')
            # 更新阈值
            if self.auto_threshold:
                self.update_threshold(features, soh)
        return soh, self.threshold
    
    def update_threshold(self, new_features=None, new_soh=None, new_threshold=None, update_normal_loss=True):
        """
        自适应更新阈值
        """
        if new_threshold is not None:
            self.threshold = new_threshold
            return
        # 更新存储样本(只保留健康样本)
        if len(self.remembered_features) < self.remembered_num and new_soh > self.threshold:
            self.remembered_features.append(new_features)
            self.remembered_soh.append(new_soh)
        elif len(self.remembered_features) >= self.remembered_num and new_soh > self.threshold:
            self.remembered_features.pop(0)
            self.remembered_soh.pop(0)
            self.remembered_features.append(new_features)
            self.remembered_soh.append(new_soh)
        # 计算新的阈值(PID)
        error = 0.01 * (new_soh) * self.threshold_alpha - self.threshold
        pid_output = self.pid_controller(error)
        #th_limit = np.mean(self.remembered_soh)*self.threshold_alpha*0.01
        #th_limit = self.threshold_alpha
        self.threshold = np.clip(self.threshold + pid_output, 0, 1)
        if update_normal_loss:
            features = np.mean(self.remembered_features, axis=0)
            self.calculate_normal_loss(features)
    

    def calculate_normal_loss(self, normal_features=None):
        """
        计算正常样本的平均损失和标准差
        """
        if normal_features is None:
            normal_features = self.normal_features
        self.soh_predictor.eval()
        with torch.no_grad():
            inputs = torch.tensor(self.normal_features, dtype=torch.float32).to(self.device)
            outputs = self.soh_predictor(inputs)
            loss = self.loss_fn(outputs, inputs)
            self.normal_loss = [torch.mean(loss).item(), torch.std(loss).item()]
        #print(f'🤓Calculating Normal loss: {self.normal_loss}')

    def pid_controller(self, error):
        """
        PID 控制器计算
        """
        # 计算比例、积分和微分项
        self.integral += error
        derivative = error - self.prev_error
        output = (self.pid_params['Kp'] * error + 
                  self.pid_params['Ki'] * self.integral + 
                  self.pid_params['Kd'] * derivative)
        self.prev_error = error
        return output
    
class SOHPredictor3(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.3, batch_first=True)
        self.attn_layer_norm = nn.LayerNorm(128)
        # 解码器部分，用于重构输入
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        # 残差链接
        self.skip = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        # 自注意力 (添加残差连接和LayerNorm)
        attn_output, _ = self.self_attn(encoded, encoded, encoded)
        encoded = self.attn_layer_norm(encoded + attn_output)
        decoded = self.decoder(encoded)
        return decoded + self.skip(x)  # 残差连接
    
class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        mse_loss = nn.MSELoss()(pred, target)
        # 添加特征相似性约束
        cos_loss = 1 - nn.CosineSimilarity()(pred, target).mean()
        return self.alpha * mse_loss + (1-self.alpha) * cos_loss

class SOHPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # 添加归一化
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            #nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # 解码器部分，用于重构输入
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        # 残差链接
        self.skip = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded + self.skip(x)  # 残差连接

class SOHPredictor2(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            #nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # 解码器部分，用于重构输入
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )
        # 残差链接
        self.skip = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded + self.skip(x)  # 残差连接

class HubberLoss(nn.Module):
    """对微小误差不敏感的新型损失"""
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
    def forward(self, pred, true):
        diff = pred - true
        return torch.where(
            torch.abs(diff) < self.delta,
            0.5 * diff.pow(2),
            self.delta * (torch.abs(diff) - 0.5 * self.delta)
        ).mean()


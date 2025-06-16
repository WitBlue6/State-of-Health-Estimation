import torch.nn as nn
import torch
import numpy as np
import random
import pandas as pd
import os
import joblib

class AnomalyProcessor:
    '''
    对数据进行加噪声、加缺失、加变换干扰
    '''
    def __init__(self, data_list=None):
        self.data_list = data_list
        self.key_name =[
            '光学模块电压',
            '控制模块电流',
        ]
    def noise_adder(self, data_list, noise_level=0.15, noise_rate=0.3, **kwargs):
        """
        对数据进行加噪声干扰
        """
        #print('Adding Noise with noise level:', noise_level)
        # 得到数据的平均幅度值
        avg_values = {}
        for key in data_list[0].keys(): 
            data = [data[key] for data in data_list]
            avg_values[key] = noise_level * abs(sum(data)) / len(data)
        # 生成噪声映射表
        key_noise_map = {}
        for key, avg_value in avg_values.items():
            key_noise_map[key] = avg_value * random.uniform(0.98, 1.02)
        # 对数据进行加噪声
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
    def __init__(self, soh_predictor, cls_model, normal_features, device, threshold=0.8, auto_threshold=False, pid_params=None, normalize=False, sclar_soh_path=None, sclar_cls_path=None, filter=False, print_log=False):
        self.soh_predictor = soh_predictor
        self.cls_model = cls_model
        self.loss_fn = nn.MSELoss(reduction='none')
        self.normal_features = normal_features
        self.device = device
        self.threshold = threshold
        self.auto_threshold = auto_threshold
        self.threshold_alpha = 0.95
        self.normal_loss = None
        self.filter = filter
        self.count = np.zeros(4+3+1+1)
        # 用于存储历史样本，更新阈值
        self.remembered_features = []
        self.remembered_soh = []
        self.remembered_num = 32
        # PID 控制器参数
        self.pid_params = pid_params if pid_params else {'Kp': 0.01, 'Ki': 0.000015, 'Kd': 0.2}
        self.prev_error = 0
        self.integral = 0
        # 标准化参数
        self.normalize = normalize
        if self.normalize:
            self.scaler_soh, self.indices_soh = joblib.load(sclar_soh_path)
            self.scaler_cls, self.indices_cls = joblib.load(sclar_cls_path)
        # 日志信息
        self.log_info = []
        self.print_log = print_log
        # 计算初始样本分布
        self.calculate_normal_loss()

    def detect_soh(self, features, normal_loss=None, alpha=0.1, key_map=None, output=True):
        """
        检测异常
        """
        if normal_loss is None:
            normal_loss = self.normal_loss
        self.soh_predictor.eval()
        if self.normalize:
            data_soh, data_cls = self.Normalize(features)
        with torch.no_grad():   
            inputs = torch.tensor(data_soh, dtype=torch.float32).to(self.device)
            outputs = self.soh_predictor(inputs)
            loss = self.loss_fn(outputs, inputs).to('cpu').numpy()
            mean_loss = np.mean(loss, axis=0)
            # 对某个模块计算SOH，最终SOH取最小值      
            normalized_loss = np.zeros_like(mean_loss)
            soh = np.zeros_like(mean_loss)
            for i in range(len(mean_loss)):
                normalized_loss[i] = abs(mean_loss[i] - normal_loss[0][i]) / (normal_loss[1][i] + 1e-6)
                soh[i] = 100 * np.exp(-normalized_loss[i] * alpha)

            # 得到SOH以及对应索引
            mean_soh = np.mean(soh)
            if self.filter:
                mean_soh = self.smooth(mean_soh)
            # 如果只计算结果用于其他方法，不输出
            if output == False:
                return mean_soh, self.threshold, soh
            
            # 输出预警信息
            warning_type = self.warning(mean_soh, data_soh, data_cls, key_map=key_map)

            # 更新阈值
            if self.auto_threshold:
                self.update_threshold(features, mean_soh, mean_loss)
        return mean_soh, self.threshold, warning_type
    
    def update_threshold(self, new_features=None, new_soh=None, mean_loss=None, new_threshold=None, update_normal_loss=False):
        """
        自适应更新阈值
        """
        if new_threshold is not None:
            self.threshold = new_threshold
            return
        # 计算新的阈值(PID)
        error = 0.01 * (new_soh) * self.threshold_alpha - self.threshold
        pid_output = self.pid_controller(error)
        #th_limit = np.mean(self.remembered_soh)*self.threshold_alpha*0.01
        #th_limit = self.threshold_alpha
        self.threshold = np.clip(self.threshold + pid_output, 0, 1)
        
        if new_soh is None or new_features is None or mean_loss is None:
            return
        # 根据样本重构误差确定是否要保留样本
        z_score = (mean_loss - self.normal_loss[0]) / self.normal_loss[1]
        if np.max(z_score) > 3: # 3 sigma原则，有一个维度异常就不更新
            return 

        # 更新存储样本(只保留健康样本)
        if len(self.remembered_features) < self.remembered_num and new_soh > self.threshold:
            self.remembered_features.append(new_features)
        elif len(self.remembered_features) >= self.remembered_num and new_soh > self.threshold:
            self.remembered_features.pop(0)
            self.remembered_features.append(new_features)

        if update_normal_loss:
            features = np.mean(self.remembered_features, axis=0)
            self.calculate_normal_loss(features)
    

    def calculate_normal_loss(self, normal_features=None):
        """
        计算正常样本的平均损失和标准差
        """
        if normal_features is None:
            normal_features = self.normal_features
        
        if self.normalize:
            normal_features, _ = self.Normalize(normal_features)

        self.soh_predictor.eval()
        with torch.no_grad():
            inputs = torch.tensor(normal_features, dtype=torch.float32).to(self.device)
            outputs = self.soh_predictor(inputs)
            loss = self.loss_fn(outputs, inputs).to('cpu').numpy()
            self.normal_loss = [np.mean(loss, axis=0), np.std(loss, axis=0)]
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
    
    def warning(self, soh, data_soh, data_cls, key_map=None):
        '''
        根据历史soh值给出预警
        return:
            0: 无异常
            1: 模块异常
            2: 健康度连续下降
            3: 健康度长期濒临阈值
        '''
        from datetime import datetime
        # 更新soh列表
        if len(self.remembered_soh) < self.remembered_num:
            self.remembered_soh.append(soh)
        elif len(self.remembered_soh) >= self.remembered_num:
            self.remembered_soh.pop(0)
            self.remembered_soh.append(soh)
        #### 1. 模块异常预警
        if soh < (self.threshold * 100):
            # 获取特征重要性
            inputs = torch.tensor(data_cls, dtype=torch.float32).to(self.device)
            logit = self.cls_model(inputs)
            probs = torch.softmax(logit, dim=1)
            confidence, pred_module = torch.max(probs, dim=1)
            # 选择置信度最高的模块
            min_index = self.analyze_anomaly_module(pred_module[0].item())
            if min_index != 'Normal':
                log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度低于阈值! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f} 可能故障模块: {min_index}'
                self.log_info.append(log)
                if self.print_log:
                    print(log)
            else:
                log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度低于阈值! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f}'
                self.log_info.append(log)
                if self.print_log:
                    print(log)
                    #print(f'Loss: {mean_loss[min_index]}')
            return 1
        
        # 如果没有足够的样本，不进行其他预警
        if len(self.remembered_soh) < self.remembered_num:
            return 0
        
        #### 2. 健康度连续下降预警
        # 用一阶线性回归判断
        calc_soh = self.remembered_soh
        x = np.arange(len(calc_soh))
        y = np.array(calc_soh)
        A = np.vstack([x, np.ones(len(x))]).T
        slope, _ = np.linalg.lstsq(A, y, rcond=None)[0] # slope <= -0.1
        # 用下降样本比例判断
        # soh_list = self.remembered_soh
        # diffs = [soh_list[i+1] - soh_list[i] for i in range(len(soh_list) - 1)]
        # drop_count = sum(d < 0 for d in diffs)
        # ratio = drop_count / len(diffs)
        if self.remembered_soh[-1] + self.remembered_num // 2 * slope < self.threshold * 100:
            log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度持续下降! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f}'
            self.log_info.append(log)
            if self.print_log:
                print(log)
            return 2
        #### 3. 健康度长期濒临阈值预警
        if all(soh - self.threshold * 100 < 3 for soh in self.remembered_soh):
            log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度长期濒临阈值! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f}'
            self.log_info.append(log)
            if self.print_log:
                print(log)
            return 3
        return 0
    def analyze_anomaly_module(self, logit):
        """
        分析异常模块
        """
        # 4个舵机 3个惯组 1个电源 1个北斗
        module = {
            0: 'Normal',
            1: '舵机1',
            2: '舵机2',
            3: '舵机3',
            4: '舵机4',
            5: '惯组X轴',
            6: '惯组Y轴',
            7: '惯组Z轴',
            8: '电源模块',
            9: '北斗模块'
        }
        if logit > 0:
            self.count[logit-1] += 1
        return module[logit]
    
    def Normalize(self, data):
        normalized_soh = data.copy()
        normalized_cls = data.copy()
        for scaler, i in zip(self.scaler_soh, self.indices_soh):
            normalized_soh[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
        for scaler, i in zip(self.scaler_cls, self.indices_cls):
            normalized_cls[:, i] = scaler.transform(data[:, i].reshape(-1, 1)).flatten()
        return normalized_soh, normalized_cls
    
    def smooth(self, soh, window_size=4, sigma=1.2):
        """
        对soh进行平滑处理
        """
        # 平滑滤波
        if len(self.remembered_soh) >= window_size:
            # 提取窗口数据
            window_data = np.array(self.remembered_soh[-(window_size-1):])
            # 加入新的soh数据
            window_data = np.append(window_data, soh)
            # 生成权重
            center = (window_size - 1) // 2
            x = np.arange(window_size) - center
            weights = np.exp(-0.5 * (x / sigma) ** 2)
            weights /= weights.sum()  # 归一化
            
            # 加权平均
            smoothed_soh = np.sum(window_data * weights)
        else:
            smoothed_soh = soh  # 数据不足时直接使用当前值
        return smoothed_soh


class SOHPredictor2(nn.Module):
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
        self.self_attn = nn.MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.3, batch_first=True)
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

class SOHPredictor(nn.Module):
    def __init__(self, input_dim):
        super(SOHPredictor, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # 添加归一化
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),  # 添加归一化
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),  # 添加归一化
            nn.ReLU(),
        )
        # 解码器部分，用于重构输入
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(),

            nn.Linear(512, input_dim)
        )
        # 残差链接
        self.skip = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded + self.skip(x)  # 残差连接

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.alpha = alpha  # 控制特征间平衡的权重
        
    def forward(self, outputs, inputs):
        # 基础MSE损失
        mse_loss = torch.abs(torch.mean((outputs - inputs)**2, dim=0))  # 按特征计算
        
        # 添加特征间平衡项：最小化各特征损失的标准差
        std_loss = torch.std(mse_loss)
        total_loss = torch.mean(mse_loss) + self.alpha * std_loss
        return total_loss

class ModuleAwareLoss(nn.Module):
    def __init__(self, device, alpha=0.5, module_ranges=None):
        super().__init__()
        self.alpha = alpha
        # 定义模块划分 (与SOHDetector中一致)
        self.module_ranges = module_ranges if module_ranges else [
            (0, 6),    # Motor1
            (6, 12),   # Motor2
            (12, 18),  # Motor3
            (18, 24),  # Motor4
            [24, 27, 30],  # EulerX (Accelx, AngAcx, Eulerx)
            [25, 28, 31],  # EulerY
            [26, 29, 32],  # EulerZ
            (33, 37),  # Power
            (37, 40)   # Beidou
        ]
        # loss加权
        motor = 3.0
        euler = 1.0
        power = 1.5
        beidou = 1.0
        self.module_weights = torch.tensor([motor, motor, motor, motor,   # 电机
                              euler, euler, euler,         # 惯组
                              power,                   # 电源
                              beidou]                  # 北斗
                             ).to(device)
    def forward(self, outputs, inputs):
        # 基础MSE损失
        mse_loss = torch.mean((outputs - inputs)**2, dim=0)  # 按特征计算

        # 计算每个模块的损失
        module_losses = []
        weighted_losses = []
        for i, module in enumerate(self.module_ranges):
            if isinstance(module, tuple):
                # 连续范围
                indices = list(range(module[0], module[1]))
            else:
                # 离散索引
                indices = module
                
            module_loss = torch.mean(mse_loss[indices])
            weighted_loss = module_loss * self.module_weights[i]
            module_losses.append(module_loss)
            weighted_losses.append(weighted_loss)
        
        # 总损失 = 模块损失的平均 + 模块间平衡项
        total_module_loss = torch.mean(torch.stack(weighted_losses))
        balance_loss = torch.std(torch.stack(module_losses))  # 平衡各模块损失
        #mean_loss = torch.mean(mse_loss)
        return total_module_loss + self.alpha * balance_loss

def GPS_relative(data_list):
    '''
    数据维度为40，最后三个维度为GPS坐标，将绝对坐标转化为相对坐标
    同时转化倒数第四个维度为电量，将绝对电量转化为相对电量
    param: data_list: 数据列表，每个元素为一个字典，包含40个特征
    return: 相对坐标数据列表
    '''
    # 将 data_list 转为二维数组
    data_array = np.array([list(entry.values()) for entry in data_list])
    gps_data = data_array[:, -3:]            # 提取 GPS（三维）
    features_wo_gps = data_array[:, :-3]     # 去除 GPS 的其他特征

    # 计算参考 GPS 点
    relative_gps = np.zeros_like(gps_data)
    relative_gps[1:, :] = gps_data[1:, :] - gps_data[:-1, :]  # 差分处理
    relative_gps[0, :] = 0

    # 合并特征和处理后的 GPS
    transformed_array = np.hstack([features_wo_gps, relative_gps])

    # 重新构建回原始的 list[dict] 形式
    keys = list(data_list[0].keys())
    transformed_data_list = []
    for row in transformed_array:
        entry = {k: float(v) for k, v in zip(keys, row)}
        transformed_data_list.append(entry)

    return transformed_data_list

def Euler_relative(data_list):
    '''
    数据维度为40，倒数7、8、9为姿态角，将绝对角度转化为相对角度
    param: data_list: 数据列表，每个元素为一个字典，包含40个特征
    '''
    # 将 data_list 转为二维数组
    data_array = np.array([list(entry.values()) for entry in data_list])
    gps_data = data_array[:, -10:-7]           
    features_wo_gps = np.hstack([data_array[:, :-10], data_array[:, -7:]])

    # 计算参考Euler
    relative_gps = np.zeros_like(gps_data)
    relative_gps[1:, :] = gps_data[1:, :] - gps_data[:-1, :]  # 差分处理
    relative_gps[0, :] = 0

    # 合并特征和处理后的Euler
    transformed_array = np.hstack([features_wo_gps[:, :-7], relative_gps, features_wo_gps[:, -7:]])

    # 重新构建回原始的 list[dict] 形式
    keys = list(data_list[0].keys())
    transformed_data_list = []
    for row in transformed_array:
        entry = {k: float(v) for k, v in zip(keys, row)}
        transformed_data_list.append(entry)

    return transformed_data_list

def Battery_relative(data_list):
    # 将 data_list 转为二维数组
    data_array = np.array([list(entry.values()) for entry in data_list])
    gps_data = np.vstack([data_array[:, 3], data_array[:, 9], data_array[:, 15], data_array[:, 21], data_array[:, 36]]).T
    features_wo_gps = np.hstack([data_array[:, :3], data_array[:, 4:9], data_array[:, 10:15], data_array[:, 16:21], data_array[:, 22:36], data_array[:, 37:]])

    # 计算参考 GPS 点
    relative_gps = np.zeros_like(gps_data)
    relative_gps[1:, :] = gps_data[1:, :] - gps_data[:-1, :]  # 差分处理
    relative_gps[0, :] = 0

    # 合并特征和处理后的 GPS
    transformed_array = np.hstack([features_wo_gps[:, :3], relative_gps[:, 0:1], 
                                   features_wo_gps[:, 3:8], relative_gps[:, 1:2],
                                   features_wo_gps[:, 8:13], relative_gps[:, 2:3],
                                   features_wo_gps[:, 13:18], relative_gps[:, 3:4],
                                   features_wo_gps[:, 18:32], relative_gps[:, 4:5],
                                   features_wo_gps[:, 32:]])

    # 重新构建回原始的 list[dict] 形式
    keys = list(data_list[0].keys())
    transformed_data_list = []
    for row in transformed_array:
        entry = {k: float(v) for k, v in zip(keys, row)}
        transformed_data_list.append(entry)

    return transformed_data_list

def Motor_rolling_window_features(data_list, window_size=3):
    '''
    对舵机的数据做滑动窗口处理，按时间维度计算每个特征的均值和方差
    '''
    # 将 data_list 转为二维数组
    data_array = np.array([list(entry.values()) for entry in data_list])
    
    # 获取 GPS 数据和其他特征数据
    gps_data = data_array[:, 0:24]  # 假设前 24 列是 GPS 数据
    features_wo_gps = data_array[:, 24:]  # 其余为特征数据
    
    # 计算每个特征在滑动窗口内的均值和方差（按时间维度）
    rolling_means = np.array([
        np.array([np.mean(gps_data[i:i+window_size, j]) for i in range(gps_data.shape[0] - window_size + 1)])
        for j in range(gps_data.shape[1])
    ]).T  # 滑动窗口均值

    rolling_means = np.vstack([np.tile(rolling_means[0], (window_size-1, 1)), rolling_means])

    rolling_stds = np.array([
        np.array([np.std(gps_data[i:i+window_size, j]) for i in range(gps_data.shape[0] - window_size + 1)])
        for j in range(gps_data.shape[1])
    ]).T  # 滑动窗口方差
    rolling_stds = np.vstack([np.tile(rolling_stds[0], (window_size-1, 1)), rolling_stds]) 
    # 将均值和方差特征拼接起来

    transformed_features = np.hstack([rolling_means, rolling_stds])
    
    # 将处理后的数据与 GPS 数据合并
    transformed_array = np.hstack([rolling_stds, features_wo_gps])  # 处理后的数据需要删除前面的 NAs

    # 重新构建回原始的 list[dict] 形式
    keys = list(data_list[0].keys())
    transformed_data_list = []
    
    for row in transformed_array:
        entry = {k: float(v) for k, v in zip(keys, row)}
        transformed_data_list.append(entry)

    return transformed_data_list

class ClassificationModel(nn.Module):
    def __init__(self, input_dim, num_modules):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),  # 添加归一化
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )
        
        # 模块异常分类器
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(32, num_modules + 1)  # +1 for normal class
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        module_logits = self.classifier(encoded)
        return module_logits
    
def noise_adder(data_list, add_noise=True, noise_rate=0.4):
    """"
    对数据添加噪声(数据增强)
    """
    if not add_noise:
        return data_list
    anomaly_processor = AnomalyProcessor()
    #bad_datas = anomaly_processor.noise_adder(data_list, noise_level=0.15, noise_rate=0.4) 
    # 将data_list和bad_datas的部分样本随机混合得到训练样本
    bad_data2 = load_data('./dataset/电源故障.txt', add_noise=False)
    bad_data3 = load_data('./dataset/北斗故障.txt', add_noise=False)
    bad_data4 = load_data('./dataset/舵机1故障.txt', add_noise=False)
    # 从bad_data中随机选择N个样本
    N = len(data_list) // 20
    bad_data2 = random.sample(bad_data2, N)
    bad_data3 = random.sample(bad_data3, N)
    bad_data4 = random.sample(bad_data4, N)
    bad_datas1 = bad_data2 + bad_data3 + bad_data4
    bad_datas1 = anomaly_processor.data_random_process(data_list, noise_level=0.05, rate=1.0, factor=2)
    #bad_datas1 = random.sample(bad_datas1, len(data_list) // 10)
    train_datas = data_list + bad_datas1
    random.shuffle(train_datas)
    return train_datas

def load_data(data_path, add_noise=False):
    """
    加载数据集
    """
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
    # 将DataFrame转换为字典列表
    data = data_pd.to_dict(orient='records')
    # 改变GPS坐标
    data = GPS_relative(data)
    # 改变Euler角
    data = Euler_relative(data)
    # 改变电量
    data = Battery_relative(data)
    # 舵机数据加窗特征提取
    data = Motor_rolling_window_features(data, window_size=20)
    # 异常注入
    data_list = noise_adder(data, add_noise=add_noise)
    return data_list

def compress_logs(logs: list[str]) -> dict:
    """
    将列表转化为大模型输入的prompt
    """
    from datetime import datetime
    from collections import defaultdict
    import re
    def parse_log_time(log_str: str):
        # 从日志中提取时间戳字符串并转为 datetime
        time_str = log_str.split('>', 1)[0]
        return time_str
    
    summary = {
        "健康度持续下降": {
            "count": 0, "times": [], "soh_values": []
        },
        "健康度低于阈值": {
            "count": 0, "times": [], "soh_values": [], "modules": defaultdict(int)
        },
        "健康度濒临阈值": {
            "count": 0, "times": [], "soh_values": []
        }
    }
    for log in logs:
        if "健康度持续下降" in log:
            summary["健康度持续下降"]["count"] += 1
            summary["健康度持续下降"]["times"].append(parse_log_time(log))
            soh_match = re.search(r'健康度:(\d+\.?\d*)', log)
            if soh_match:
                summary["健康度持续下降"]["soh_values"].append(float(soh_match.group(1)))

        elif "健康度低于阈值" in log:
            summary["健康度低于阈值"]["count"] += 1
            summary["健康度低于阈值"]["times"].append(parse_log_time(log))
            soh_match = re.search(r'健康度:(\d+\.?\d*)', log)
            if soh_match:
                summary["健康度低于阈值"]["soh_values"].append(float(soh_match.group(1)))
            module_match = re.search(r'可能故障模块:\s*(\S+)', log)
            if module_match:
                module = module_match.group(1)
                summary["健康度低于阈值"]["modules"][module] += 1

        elif "健康度长期濒临阈值" in log:
            summary["健康度濒临阈值"]["count"] += 1
            summary["健康度濒临阈值"]["times"].append(parse_log_time(log))
            soh_match = re.search(r'健康度:(\d+\.?\d*)', log)
            if soh_match:
                summary["健康度濒临阈值"]["soh_values"].append(float(soh_match.group(1)))

    # 精炼输出格式
    result = {}
    for key, value in summary.items():
        if value["count"] == 0:
            continue

        times = value["times"]
        result[key] = {
            "次数": value["count"],
            "时间范围": f"{min(times)} ~ {max(times)}" if times else "无记录",
            "SOH范围": f"{min(value['soh_values']):.2f} ~ {max(value['soh_values']):.2f}" if value["soh_values"] else "未知",
        }

        if "modules" in value:
            result[key]["模块频次"] = dict(sorted(value["modules"].items(), key=lambda x: -x[1]))
    #print(summary)
    print(result)
    prompt = f"""
    【日志信息为 Python 字典，字段说明如下】：
    - 健康度持续下降: {{'次数': int, '时间范围': str, 'SOH范围': str}}，表明设备健康度持续降低
    - 健康度低于阈值: {{'次数': int, '时间范围': str, 'SOH范围': str, '模块频次': dict}}，表明设备存在可能故障模块，并给出模块出现频次
    - 健康度濒临阈值: {{'次数': int, '时间范围': str, 'SOH范围': str}}，表明设备健康度濒临阈值
    压缩日志信息如下：  
    {result}
    """
    return prompt


import openai
from dotenv import load_dotenv

def LLM_answer(message, api_key=None, base_url=None, model='gpt'):
    '''
    调用大模型回答问题
    '''
    load_dotenv('.env')
    if model == 'gpt':
        if api_key is None:
            api_key = os.getenv('GPT_API_KEY')
            base_url = os.getenv('GPT_BASE_URL')
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        system_content = "你是一个电子设备日志分析专家，请根据日志信息，给出摘要和建议操作"
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": message},
            ]
        )
        #print(response)
        content = response.choices[0].message.content
        if content is None:
            print(response.choices[0])
    return content
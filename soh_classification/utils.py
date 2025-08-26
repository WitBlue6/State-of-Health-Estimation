import torch.nn as nn
import torch
import numpy as np
import random
import pandas as pd
import os
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
import copy

class AnomalyProcessor:
    '''
    对数据进行加噪声、加缺失、加变换干扰
    '''
    def __init__(self, data_list=None):
        self.data_list = data_list
        self.key_name = [
            'Voltage', 'Current', 'Power'
        ]
    def increasingly_noise_adder(self, data_list_in, **kwargs):
        """
        按递增异常程度加入噪声干扰
        """
        pass

    def noise_adder(self, data_list_in, noise_level=0.15, noise_rate=0.3, **kwargs):
        """
        对数据进行加噪声干扰
        """
        #print('Adding Noise with noise level:', noise_level)
        # 得到数据的平均幅度值
        data_list = copy.deepcopy(data_list_in)
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
            for key in self.key_name:
                data_list[i][key] += np.random.normal(0, key_noise_map[key])
        return data_list
    
    def missing_adder(self, data_list_in, missing_rate=1.0, **kwargs):
        """
        对数据进行加缺失干扰
        """
        #print('Adding Missing with missing rate:', missing_rate)
        data_list = copy.deepcopy(data_list_in)
        for data in data_list:
            for key in data:
                if random.random() < missing_rate:
                    data[key] = 0.0
        return data_list
    
    def data_transform(self, data_list_in, transform_level=1.0, transform_rate=1.0, **kwargs):
        """
        对数据进行加变换干扰
        """
        data_list = copy.deepcopy(data_list_in)
        #print('Adding Transform with transform rate:', transform_rate)
        for data in data_list:
            for key in data:
                if random.random() < transform_rate and key in self.key_name:
                    if random.random() < 0.5:
                        data[key] *= transform_level * random.uniform(0.8, 1.2)  # 缩放
                    else:
                        data[key] += transform_level * random.uniform(-0.5, 0.5)  # 平移
        return data_list

    def outlier_adder(self, data_list_in, outlier_rate=1.0, factor=2.0, **kwargs):
        """
        对数据进行加异常值干扰
        """
        #print('Adding Outlier with outlier rate:', outlier_rate)
        data_list = copy.deepcopy(data_list_in)
        for data in data_list:
            for key in data:
                if random.random() < outlier_rate:
                    data[key] *= random.choice([-factor, factor])  # 异常值
        return data_list
    
    def data_random_process(self, data_list_in, noise_level=0.15, rate=0.25, factor=5.0, **kwargs):
        """
        对数据随机进行加噪声、加缺失、加变换干扰
        """
        data_list = copy.deepcopy(data_list_in)
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
    def __init__(self, soh_predictor, cls_model, normal_features, device, threshold=0.8, auto_threshold=False, pid_params=None, normalize=False, sclar_soh_path=None, sclar_cls_path=None, filter=False, print_log=False, log_path=None):
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
        # 定义模块划分
        self.module_ranges = [
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
        self.count = np.zeros(4+3+1+1)
        # 可用性输出
        self.module_rul = [np.inf] * len(self.module_ranges)
        self.rul = np.inf
        self.soh = 100.0
        self.slope_history = []
        self.slope_remember_num = 12
        # 结构配置
        self.module_category = {
            "舵机1": "互补",
            "舵机2": "互补",
            "舵机3": "互补",
            "舵机4": "互补",
            "惯组X轴": "依赖",
            "惯组Y轴": "依赖",
            "惯组Z轴": "依赖",
            "电源模块": "关键",
            "北斗模块": "依赖"
        }
        self.module_index = {
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
        self.dependency_weights = {
            0: 0, # Normal
            1: 0, # 舵机1
            2: 0, # 舵机2
            3: 0, # 舵机3
            4: 0, # 舵机4
            5: 0.2, # 惯组X轴
            6: 0.2, # 惯组Y轴
            7: 0.2, # 惯组Z轴
            8: 0, # 电源模块
            9: 0.4, # 北斗模块
        }
        # 用于存储历史样本，更新阈值
        self.remembered_features = []
        self.remembered_soh = []
        self.remembered_module_soh = []
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
        self.log_path = log_path
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
            mean_loss = np.mean(loss, axis=0)  # 得到每个特征的loss
            
            # 按特征维度计算不同特征SOH   
            normalized_loss = np.zeros_like(mean_loss)
            soh = np.zeros_like(mean_loss)
            for i in range(len(mean_loss)):
                normalized_loss[i] = abs(mean_loss[i] - normal_loss[0][i]) / (normal_loss[1][i] + 1e-6)
                soh[i] = 100 * np.exp(-normalized_loss[i] * alpha)
            
            # 所有特征平均作为设备SOH
            mean_soh = np.mean(soh)
            
            if self.filter:
                mean_soh = self.smooth(mean_soh)

            self.soh = mean_soh
            # 根据每个模块的健康度，计算设备可用时间
            module_rul = self.estimate_module_rul(soh)
            rul, rul_log = self.estimate_device_rul(module_rul)

            # 如果只计算结果用于其他方法，不输出
            if output == False:
                return mean_soh, self.threshold, soh
            
            # 输出预警信息
            warning_type, warning_log = self.warning(mean_soh, data_soh, data_cls, key_map=key_map)

            # 更新阈值
            if self.auto_threshold:
                self.update_threshold(features, mean_soh, mean_loss)
        return mean_soh, self.threshold, warning_type, rul, rul_log, warning_log

    def estimate_module_rul(self, all_soh, window=128, min_rul=1, dt=0.2, module_ranges=None, module_thres=[90.0]*9):
        '''
        估计不同模块的可用时间
        '''
        module_ranges = module_ranges if module_ranges else self.module_ranges
        # 计算各模块健康度
        module_soh = np.zeros(9)
        for i, module in enumerate(module_ranges):
            if isinstance(module, tuple):
                # 连续范围
                indices = list(range(module[0], module[1]))
            else:
                # 离散索引
                indices = module
            module_soh[i] = np.mean(all_soh[indices])
        # 记录历史模块健康度
        if len(self.remembered_module_soh) < window:
            self.remembered_module_soh.append(module_soh)
        else:
            self.remembered_module_soh.pop(0)
            self.remembered_module_soh.append(module_soh)
        # 预测每个模块的RUL
        if len(self.remembered_module_soh) < window:
            return [np.inf] * len(self.module_ranges)

        rul_list = []
        soh_matrix = np.array(self.remembered_module_soh[-window:])
        time_index = np.arange(window) * dt

        # 生成权重：时间越近，权重越大
        weights = np.exp(np.linspace(-2, 0, window)) 
        #W = np.diag(weights)
        A = np.vstack([time_index, np.ones(window)]).T
        model = LinearRegression()

        for i in range(soh_matrix.shape[1]):
            # 加权线性拟合
            y = soh_matrix[:, i]
        
            model.fit(A, y, sample_weight=weights)
            slope = model.coef_[0]
            # 储存历史slope
            if len(self.slope_history) < self.slope_remember_num:
                self.slope_history.append(slope)
            else:
                self.slope_history.pop(0)
                self.slope_history.append(slope)
            slope_smooth = np.mean(self.slope_history)
            # # 如果模块SOH高于阈值，视为长期可用
            if y[-1] + slope_smooth * window > module_thres[i]:
                rul_list.append(np.inf)
                continue
            
            # 预测未来低于阈值
            if slope_smooth >= 0:
                # 防止波动，如果斜率大于0，使用历史rul
                rul_list.append(self.module_rul[i] * 0.95)
            else:
                t_remain = y[-1] / (-slope_smooth) * 2
                rul_list.append(t_remain if t_remain > min_rul else 0)
            #self.write_log(f"模块:{self.module_index.get(i+1)} 斜率:{slope_smooth}  RUL:{rul_list[-1]}")

        # 对RUL结果平滑处理
        for i in range(len(rul_list)):
            if rul_list[i] != np.inf and self.module_rul[i] != np.inf:
                rul_list[i] = rul_list[i] * 0.65 + self.module_rul[i] * 0.35

        self.module_rul = rul_list

        return rul_list

    def estimate_device_rul(self, module_ruls, module_thres=[80.0]*9):
        '''
        计算设备可用时间
        '''
        critical_rul = np.inf
        dependency_rul = 0
        num_faulty_servos = 0
        rul_info = "系统长期可用"

        soh_vector = self.remembered_module_soh[-1]
        for idx, soh in enumerate(soh_vector):
            name = self.module_index.get(idx + 1)
            category = self.module_category.get(name)
            rul = module_ruls[idx]
            
            if soh < module_thres[idx]:
                if category == "关键":
                    critical_rul = min(critical_rul, rul)
                elif category == "互补":
                    num_faulty_servos += 1
                elif category == "依赖":
                    weight =  self.dependency_weights[idx + 1]
                    dependency_rul += weight * rul
        
        # 优先级：关键 > 依赖 > 互补  
        if critical_rul < np.inf:
            self.rul = critical_rul
            rul_info = "关键模块失效"
        elif dependency_rul > 0:
            self.rul = dependency_rul
            rul_info = "依赖模块影响"
        elif num_faulty_servos >= 3:
            self.rul = min([module_ruls[i] for i in [0, 1, 2, 3]])
            rul_info = "多个互补模块失效"
        else:
            self.rul = np.inf
        log = f'\n{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>{rul_info} RUL:{self.rul:.2f} 健康度:{self.soh:.2f} 阈值:{self.threshold * 100:.2f}'
        #self.log_info.append(log)
        self.write_log(log)
        return self.rul, log
                    
    
    def update_threshold(self, new_features=None, new_soh=None, mean_loss=None, new_threshold=None, update_normal_loss=False):
        """
        自适应更新阈值
        """
        if new_threshold is not None:
            self.threshold = new_threshold
            return
        if new_soh is None or new_features is None or mean_loss is None:
            return
        # 计算新的阈值(PID)
        error = 0.01 * (new_soh) * self.threshold_alpha - self.threshold
        pid_output = self.pid_controller(error)
        #th_limit = np.mean(self.remembered_soh)*self.threshold_alpha*0.01
        #th_limit = self.threshold_alpha
        self.threshold = np.clip(self.threshold + pid_output, 0, 1)
        
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
        print(f'🤓Calculating Normal loss: {self.normal_loss}')

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
                #self.log_info.append(log)
                self.write_log(log)
                
            else:
                log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度低于阈值! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f}'
                #self.log_info.append(log)
                self.write_log(log)
                
            return 1, log
        
        # 如果没有足够的样本，不进行其他预警
        if len(self.remembered_soh) < self.remembered_num:
            return 0, ""
        
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
            #self.log_info.append(log)
            self.write_log(log)
            return 2, log
        #### 3. 健康度长期濒临阈值预警
        if all(soh - self.threshold * 100 < 3 for soh in self.remembered_soh):
            log = f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}>>健康度长期濒临阈值! 健康度:{soh:.2f} 阈值:{self.threshold * 100:.2f}'
            #self.log_info.append(log)
            self.write_log(log)
            return 3, log
        return 0, ""
    def analyze_anomaly_module(self, logit, module=None):
        """
        分析异常模块
        """
        # 4个舵机 3个惯组 1个电源 1个北斗
        module = module if module else self.module_index
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
    
    def write_log(self, log: str, append=True):
        self.log_info.append(log)
        print(log)
        if self.log_path:
            if append:
                with open(self.log_path, "a") as f:
                    f.write(log + "\n")
            else:
                with open(self.log_path, "w") as f:
                    f.write(log + "\n")

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

def compress_logs_for_model(logs: list[str]) -> dict:
    """
    根据日志结构，压缩日志为结构化摘要
    """
    import re
    from collections import defaultdict
    from datetime import datetime
    # 用来存放各种统计数据
    summary = {
        "系统长期可用": {"count": 0, "times": [], "soh_values": []},
        "RUL_关键模块失效": {"count": 0, "times": [], "RUL_values": []},
        "RUL_依赖模块失效": {"count": 0, "times": [], "RUL_values": []},
        "RUL_互补模块失效": {"count": 0, "times": [], "RUL_values": []},
        "预警_健康度持续下降": {"count": 0, "times": [], "soh_values": []},
        "预警_健康度低于阈值": {"count": 0, "times": [], "soh_values": [], "modules": defaultdict(int)},
        "预警_健康度长期濒临阈值": {"count": 0, "times": [], "soh_values": []},
    }
    for line in logs:
        line = line.strip()
        # 时间戳格式提取，假设格式固定，类似：2025-07-27 11:04:46.803>>
        time_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})>>", line)
        time_str = time_match.group(1) if time_match else None

        # 统一提取健康度和阈值
        soh_match = re.search(r"健康度:([0-9.]+)", line)
        thres_match = re.search(r"阈值:([0-9.]+)", line)
        rul_match = re.search(r"RUL:(inf|[0-9.]+)", line)

        # 解析模块名（预警低于阈值时可能出现）
        module_match = re.search(r"模块(?:名称)?[:：]?\s*([\w\-]+)", line)
        # 有些日志没模块名，这里匹配模块名字留给后续统计

        # 分类判断
        if "系统长期可用" in line:
            summary["系统长期可用"]["count"] += 1
            if time_str:
                summary["系统长期可用"]["times"].append(time_str)
            if soh_match:
                summary["系统长期可用"]["soh_values"].append(float(soh_match.group(1)))

        elif "关键模块失效" in line:
            summary["RUL_关键模块失效"]["count"] += 1
            if time_str:
                summary["RUL_关键模块失效"]["times"].append(time_str)
            if rul_match:
                val = float('inf') if rul_match.group(1) == "inf" else float(rul_match.group(1))
                summary["RUL_关键模块失效"]["RUL_values"].append(val)

        elif "依赖模块失效" in line:
            summary["RUL_依赖模块失效"]["count"] += 1
            if time_str:
                summary["RUL_依赖模块失效"]["times"].append(time_str)
            if rul_match:
                val = float('inf') if rul_match.group(1) == "inf" else float(rul_match.group(1))
                summary["RUL_依赖模块失效"]["RUL_values"].append(val)

        elif "互补模块失效" in line:
            summary["RUL_互补模块失效"]["count"] += 1
            if time_str:
                summary["RUL_互补模块失效"]["times"].append(time_str)
            if rul_match:
                val = float('inf') if rul_match.group(1) == "inf" else float(rul_match.group(1))
                summary["RUL_互补模块失效"]["RUL_values"].append(val)

        elif "健康度持续下降" in line:
            summary["预警_健康度持续下降"]["count"] += 1
            if time_str:
                summary["预警_健康度持续下降"]["times"].append(time_str)
            if soh_match:
                summary["预警_健康度持续下降"]["soh_values"].append(float(soh_match.group(1)))

        elif "健康度低于阈值" in line:
            summary["预警_健康度低于阈值"]["count"] += 1
            if time_str:
                summary["预警_健康度低于阈值"]["times"].append(time_str)
            if soh_match:
                summary["预警_健康度低于阈值"]["soh_values"].append(float(soh_match.group(1)))
            # 如果模块名出现，统计模块频次
            if module_match:
                summary["预警_健康度低于阈值"]["modules"][module_match.group(1)] += 1

        elif "健康度长期濒临阈值" in line:
            summary["预警_健康度长期濒临阈值"]["count"] += 1
            if time_str:
                summary["预警_健康度长期濒临阈值"]["times"].append(time_str)
            if soh_match:
                summary["预警_健康度长期濒临阈值"]["soh_values"].append(float(soh_match.group(1)))

    # 格式化结果看时间范围和数值范围
    def summarize_section(data: dict):
        if data["count"] == 0:
            return None
        times = data["times"]
        soh_vals = data.get("soh_values", [])
        rul_vals = data.get("RUL_values", [])

        result = {
            "次数": data["count"],
            "时间范围": f"{min(times)} ~ {max(times)}" if times else "无记录"
        }
        if soh_vals:
            result["健康度范围"] = f"{min(soh_vals):.2f} ~ {max(soh_vals):.2f}"
        if rul_vals:
            # 特殊处理inf
            finite_ruls = [v for v in rul_vals if v != float('inf')]
            if len(finite_ruls) == 0:
                result["RUL范围"] = "inf"
            else:
                result["RUL范围"] = f"{min(finite_ruls):.2f} ~ {max(finite_ruls):.2f}"

        # 如果有模块频次，排序后加入
        if "modules" in data and data["modules"]:
            sorted_modules = dict(sorted(data["modules"].items(), key=lambda x: -x[1]))
            result["模块频次"] = sorted_modules
        return result
    def build_prompts_from_summary(result: dict) -> str:
        prompt = f"""
        【日志信息为 Python 字典，字段说明如下】：
        - 健康度持续下降: {{'次数': int, '时间范围': str, '健康度范围': str}}，表明设备健康度持续降低
        - 预警_健康度低于阈值: {{'次数': int, '时间范围': str, '健康度范围': str, '模块频次': dict}}，表明设备存在可能故障模块，并给出模块出现频次
        - 健康度长期濒临阈值: {{'次数': int, '时间范围': str, '健康度范围': str}}，表明设备健康度濒临阈值
        - RUL_关键模块失效: {{'次数': int, '时间范围': str, 'RUL范围': str}}，表明关键模块失效的次数及剩余寿命范围
        - RUL_依赖模块失效: {{'次数': int, '时间范围': str, 'RUL范围': str}}，表明依赖模块失效相关信息
        - RUL_互补模块失效: {{'次数': int, '时间范围': str, 'RUL范围': str}}，表明互补模块失效相关信息

        压缩日志信息如下：
        {result}
        """
        return prompt

    output = {}
    for key, val in summary.items():
        res = summarize_section(val)
        if res:
            output[key] = res
    # 监测是否为空
    if not output:
        output = {"无记录": {"次数": 0, "时间范围": "无记录"}}
    return build_prompts_from_summary(output)


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
import pandas as pd
import json
import os
import numpy as np
import random
from scipy.interpolate import interp1d

# Min-Max
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# 随机生成reason
def generare_reason():
	reason_list = ["The Voltage is too low.", "The Current is too high.", 
			   "The Temperature is too high.", "The Depth of Discharge is too low.", 
			   "The Discharge Capacity is too low.", "The Charge Capacity is too low."]
	reason = random.sample(reason_list, 1)[0]
	#print(reason)
	return reason
def process_EVERLASTING(file_path):
	df = pd.read_csv(file_path)
	#print(df)
	# 额定容量3.5Ah
	rated_capacity = df["Charge_Capacity_Ah"].max()
	df.fillna(0, inplace=True)   # nan视为没有充放电，填充0
	# 计算 SOH
	df["SOH(%)"] = df["Discharge_Capacity_Ah"] / rated_capacity * 100
	# 删除SOH为的行
	df = df[df["SOH(%)"] > 80].reset_index(drop=True)
	dlen = len(df)
	print(df.head())
	print(df.tail())
	# 生成原因
	df["Reason"] = df.apply(lambda x: generare_reason(), axis=1)
	# 生成训练数据
	train_data = []
	print("Begin Transforming Dataset.....")
	for index, row in df.iterrows():
		instruction = f"You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on charge capacity, discharge capacity, current, and voltage:[{row['Charge_Capacity_Ah']}, {row['Discharge_Capacity_Ah']}, {row['Current_A	']}, {row['Voltage_V']}].In addition, you need to give me the reason for your estimation."
		input_text = f"Charge Capacity: {row['Charge_Capacity_Ah']}Ah, Discharge Capacity: {row['Discharge_Capacity_Ah']}Ah, Current: {row['Current_A']}A, Voltage: {row['Voltage_V']}V"
		response_text = f"SOH is {row['SOH(%)']:.2f}%.Because {row['Reason']}"

		train_data.append({
			"instruction": instruction,
			"input": input_text,
			"output": response_text
		})
		if index % 10000 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	return train_data

def process_NMC(file_path):
	df = pd.read_excel(file_path, skiprows=2)
	df = df.drop(df.columns[0], axis=1)
	df = df.dropna().reset_index(drop=True)
	dlen = len(df)
	print(df.head())
	print(df.tail())

	# 生成训练数据
	train_data = []
	print("Begin Transforming Dataset.....")
	for index, row in df.iterrows():
		instruction = f"You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on temperature, discharging current, depth of discharge and average charging current:[{row['Ambient temperature']}, {row['Discharging current']}, {row['Depth of discharge']}, {row['Average charging current']}].And give me the reason for your estimation."
		input_text = f"{row['Ambient temperature']}, {row['Discharging current']}, {row['Depth of discharge']}, {row['Average charging current']}"
		response_text = f"SOH is {row['State of Health']:.3f}%.Because..."

		train_data.append({
			"instruction": instruction,
			"input": input_text,
			"output": response_text
		})
		if index % 10000 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	return train_data

def process_Oxford():
	# 读取文件
	# https://ora.ox.ac.uk/objects/uuid:9aae61af-2949-49f1-8ad5-6aea448979e5
	df1 = pd.read_csv("./dataset/BMP_cell1_profileData.csv")
	df2 = pd.read_csv("./dataset/BMP_cell1_capacityData.csv")
	# 提取时间和容量数据
	time_1 = df1["time_s"].values
	time_2 = df2["time_s"].values
	capacity_2 = df2["capacity_Ah"].values
	# 插值
	interp_func = interp1d(time_2, capacity_2, kind='linear', fill_value="extrapolate")
	df1["capacity_Ah"] = interp_func(time_1)
	df1 = df1[df1 != 0]  # 将0视为无效数据
	df1 = df1.dropna().reset_index(drop=True)
	# 计算SOH
	rated_capacity = df1["capacity_Ah"].max()
	df1["SOH"] = df1["capacity_Ah"] / rated_capacity * 100
	dlen = len(df1)
	print(df1.head())
	print(df1.tail())

	# 归一化
	# df1["environment_temperature_C"] = min_max_normalize(df1["environment_temperature_C"].values)
	# df1["current_A"] = min_max_normalize(df1["current_A"].values)
	# df1["voltage_V"] = min_max_normalize(df1["voltage_V"].values)
	# df1["SOH"] = min_max_normalize(df1["SOH"].values)

	# 生成原因
	df1["Reason"] = df1.apply(lambda x: generare_reason(), axis=1)
	# 生成训练数据
	train_data = []
	print("Begin Transforming Dataset.....")
	for index, row in df1.iterrows():
		instruction = f"You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on temperature, current, and voltage:[{row['environment_temperature_C']}, {row['current_A']}, {row['voltage_V']}].And give me the reason for your estimation."
		input_text = f"{row['environment_temperature_C']}, {row['current_A']}, {row['voltage_V']}"
		response_text = f"SOH is {row['SOH']}%.Because row['Reason']"
		train_data.append({
			"instruction": instruction,
			"input": input_text,
			"output": response_text
		})
		if index % 10000 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	return train_data

def process_CALCE():
	# 马里兰大学数据集
	# https://gitcode.com/open-source-toolkit/6077e/?utm_source=tools_gitcode&index=top&type=card&&isLogin=1
	#arr = np.load("./dataset/CALCE/CALCE.npy", allow_pickle=True) #数据量太少
	#print(arr)

	def load_one_sample(file_path):
		# 读取一个excel，返回一个DataFrame
		df = pd.read_excel(file_path, sheet_name=1)  # 使用第二个sheet
		# 提取文件名中的电池信息
		battery_name = os.path.basename(file_path).split('.')[0]
		# 获取所有cycle的数据
		cycles = df['Cycle_Index'].unique()
		features_list = []
		for cycle in cycles:
			# 获取当前cycle的数据
			cycle_data = df[df['Cycle_Index'] == cycle].reset_index(drop=True)
			# 只筛选出放电数据
			discharge_data = cycle_data[cycle_data['Current(A)'] < 0].reset_index(drop=True)
			if len(discharge_data) <= 1:
				continue		
			# 找出dV/dt的峰值，取绝对值最大的值
			peak_dvdt = discharge_data['dV/dt(V/s)'].abs().max()
			discharge_duration = discharge_data['Test_Time(s)'].max() - discharge_data['Test_Time(s)'].min()
			# 计算放电持续时间
			discharge_duration = discharge_data['Test_Time(s)'].max() - discharge_data['Test_Time(s)'].min()
			# 计算放电容量（使用每一个cycle的最大最小值之差）
			discharge_capacity = discharge_data['Discharge_Capacity(Ah)'].max() - discharge_data['Discharge_Capacity(Ah)'].min()
			# 计算放电平均电流
			avg_discharge_current = discharge_data['Current(A)'].mean()
			features = {
				'battery_name': battery_name,
				'cycle': cycle,
				'peak_dvdt': peak_dvdt,
				'discharge_duration': discharge_duration,
				'discharge_capacity': discharge_capacity,
				'avg_discharge_current': avg_discharge_current
			}
			features_list.append(features)
		return features_list
	
	# 处理所有CALCE数据集文件
	all_features = []
	# 遍历所有电池文件夹
	base_folder = './dataset/CALCE/CS2_35'  #35 36 37 38 四组电池数据
	battery_folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f)) and f.startswith('CS2_')]
	if battery_folders == []:
		battery_folders.append(base_folder)
	for folder in battery_folders:
		folder_path = os.path.join(base_folder, folder)
		if not os.path.isdir(folder_path):
			folder_path = base_folder
		
		excel_files = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
		for excel_file in excel_files:
			file_path = os.path.join(folder_path, excel_file)
			try:
				features = load_one_sample(file_path)
				all_features.extend(features)
				print(f"Processed {file_path}, extracted {len(features)} cycles")
			except Exception as e:
				print(f"Error processing {file_path}: {e}")
	
	# 转换为DataFrame
	features_df = pd.DataFrame(all_features)
	# 计算SOH（以最大放电容量为基准）
	max_capacity = features_df['discharge_capacity'].max()
	features_df['SOH'] = features_df['discharge_capacity'] / max_capacity * 100
	# 生成原因
	features_df['Reason'] = features_df.apply(lambda x: generare_reason(), axis=1)
	# 生成训练数据
	train_data = []
	dlen = len(features_df)
	
	for index, row in features_df.iterrows():
		instruction = f"You are a SOH estimation expert. Estimate the SOH of a lithium-ion battery based on peak dV/dt, discharge duration, discharge capacity, and average discharge current: [{row['peak_dvdt']:.6f}, {row['discharge_duration']:.2f}, {row['discharge_capacity']:.6f}, {row['avg_discharge_current']:.4f}]. In addition, you need to give me the reason for your estimation."
		input_text = f"Peak dV/dt: {row['peak_dvdt']:.6f} V/s, Discharge Duration: {row['discharge_duration']:.2f} s, Discharge Capacity: {row['discharge_capacity']:.6f} Ah, Average Discharge Current: {row['avg_discharge_current']:.4f} A"
		response_text = f"SOH is {row['SOH']:.2f}%. Because row['Reason']"
		
		train_data.append({
			"instruction": instruction,
			"input": input_text,
			"output": response_text
		})
		
		if index % 100 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	
	return train_data

import re
def process_1553B():
	with open('./dataset/BC Data1_20250411_全速工作模式.bcd') as f:
		data = f.read()
		datas = re.findall(r'DATAS\d+=\s*(.*?)\s*(?=COMMENT\d+=|$)', data)
	data_list = []
	# 处理得到采集数据
	for i, data_block in enumerate(datas):
		#print(data_block)  # str: 0821 ea01 0800 0000 ....
		# 分割字符串
		items = data_block.split()
		if len(items) < 32: # 说明是南航发的指令，大于32是数据指令
			continue
		# 字序号6-29对应索引7-30
		tim_voltage = int (items[7]+items[8], 16) * 3.05 / 10000.0
		pow_voltage = int (items[9]+items[10], 16) * 3.05 / 10000.0
		ctl_voltage = int (items[11]+items[12], 16) * 3.05 / 10000.0
		dsp_voltage = int (items[13]+items[14], 16) * 3.05 / 10000.0
		tim_current = (int (items[15]+items[16], 16) * 3.05 - 25000.0) * 4.0 / 10000.0
		pow_current = (int (items[17]+items[18], 16) * 3.05 - 25000.0) * 4.0 / 10000.0
		ctl_current = (int (items[19]+items[20], 16) * 3.05 - 25000.0) * 4.0 / 10000.0
		dsp_current = (int (items[21]+items[22], 16) * 3.05 - 25000.0) * 4.0 / 10000.0
		tim_temperature = ((int (items[23]+items[24], 16) >> 8) & 0xFF) * 10.0 + ((int (items[23]+items[24], 16)) & 0xFF) / 10.0
		pow_temperature = ((int (items[25]+items[26], 16) >> 8) & 0xFF) * 10.0 + ((int (items[25]+items[26], 16)) & 0xFF) / 10.0
		ctl_temperature = ((int (items[27]+items[28], 16) >> 8) & 0xFF) * 10.0 + ((int (items[27]+items[28], 16))& 0xFF) / 10.0
		dsp_temperature = ((int (items[29]+items[30], 16) >> 8) & 0xFF) * 10.0 + ((int (items[29]+items[30], 16)) & 0xFF) / 10.0
		data_trans = {
			"tim_voltage": tim_voltage,
			"pow_voltage": pow_voltage,
			"ctl_voltage": ctl_voltage,
			"dsp_voltage": dsp_voltage,
			"tim_current": tim_current,
			"pow_current": pow_current,
			"ctl_current": ctl_current,
			"dsp_current": dsp_current,
			"tim_temperature": tim_temperature,
			"pow_temperature": pow_temperature,			
			"ctl_temperature": ctl_temperature,
			"dsp_temperature": dsp_temperature,	
		}
		data_list.append(data_trans)
	for i in data_list:
		print(i, '\n')
	
#print(datas)

# file_path = "./dataset/EIL-MJ1-015.csv"	# https://dx.doi.org/10.5522/04/12159462.v1  有效数据太少
#file_path = "./dataset/DrivingAgeing_T25_SOC10-90_Vito_Cell89_AllData.csv"  #EVERLASTING project DOI: 10.4121/13739296 SOH计算不明确
#file_path = "./dataset/NMC_cycling_data.xlsx"  # NMC https://data.mendeley.com/datasets/k6v83s2xdm/1  数据量少，训练不到

#train_data = process_EVERLASTING(file_path)
#train_data = process_NMC(file_path)
#train_data = process_Oxford()  # 牛津电池实验室,很好,缺少日志信息

train_data = process_CALCE()  # 马里兰大学，数据量大，缺少日志信息

#process_1553B()
# 保存为 JSON 格式，供微调使用
output_json_path = "./dataset/battery_dataset.json"

if os.path.exists(output_json_path):  # 保证不会写串行
    os.remove(output_json_path)

with open(output_json_path, "w", encoding="utf-8") as f:
    for entry in train_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset has been saved to {output_json_path}")

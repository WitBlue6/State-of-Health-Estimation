import pandas as pd
import json
import os
import numpy as np
import random
from scipy.interpolate import interp1d

# 随机生成reason
def generare_reason():
	reason_list = ["The Voltage is too low.", "The Current is too high.", 
			   "The Temperature is too high.", "The Depth of Discharge is too low.", 
			   "The Discharge Capacity is too low.", "The Charge Capacity is too low."]
	reason = random.sample(reason_list, 1)[0]
	print(reason)
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
	# 生成原因
	df1["Reason"] = df1.apply(lambda x: generare_reason(), axis=1)
	# 生成训练数据
	train_data = []
	print("Begin Transforming Dataset.....")
	for index, row in df1.iterrows():
		instruction = f"You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on temperature, current, and voltage:[{row['environment_temperature_C']}, {row['current_A']}, {row['voltage_V']}].And give me the reason for your estimation."
		input_text = f"{row['environment_temperature_C']}, {row['current_A']}, {row['voltage_V']}"
		response_text = f"SOH is {row['SOH']:.3f}%.Because {row['Reason']}"
		train_data.append({
			"instruction": instruction,
			"input": input_text,
			"output": response_text
		})
		if index % 10000 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	return train_data

# file_path = "./dataset/EIL-MJ1-015.csv"	# https://dx.doi.org/10.5522/04/12159462.v1  有效数据太少
#file_path = "./dataset/DrivingAgeing_T25_SOC10-90_Vito_Cell89_AllData.csv"  #EVERLASTING project DOI: 10.4121/13739296 SOH计算不明确
#file_path = "./dataset/NMC_cycling_data.xlsx"  # NMC https://data.mendeley.com/datasets/k6v83s2xdm/1  数据量少，训练不到

#train_data = process_EVERLASTING(file_path)
#train_data = process_NMC(file_path)
train_data = process_Oxford()  # 牛津电池实验室,很好,缺少日志信息
# 保存为 JSON 格式，供微调使用
output_json_path = "./dataset/battery_dataset.json"

if os.path.exists(output_json_path):  # 保证不会写串行
    os.remove(output_json_path)

with open(output_json_path, "w", encoding="utf-8") as f:
    for entry in train_data:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Dataset has been saved to {output_json_path}")

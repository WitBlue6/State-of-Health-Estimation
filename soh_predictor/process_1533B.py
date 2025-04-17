import re
import pandas as pd
import os
import json
def get_data(data_path):
	"""
	解析数据
	:param data_path: 数据路径
	:return: 解析后的数据流,包含所有电压电流温度数据
	"""
	with open(data_path) as f:
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
	# for i in data_list:
	# 	print(i, '\n')
	return data_list

def write_data(data_list, output_json_path='./dataset/data.csv'):
	"""
	将解析得到数据流写入json文件
	:param data_list: 解析得到的prompt
	:param output_json_path: 输出路径
	:return: None
	"""
	if os.path.exists(output_json_path):  # 保证不会写串行
		os.remove(output_json_path)
	with open(output_json_path, "w", encoding="utf-8") as f:
		print('Writing Dataset!!!')
		for entry in data_list:
			f.write(json.dumps(entry, ensure_ascii=False) + "\n")
	print(f"Dataset has been saved to {output_json_path}")

def get_prompt(data_path, output_path='./dataset/1533B.json'):
	"""
	将解析得到数据流转化为模型的prompt输入,写入json文件
	:param data_path: 数据路径
	:param output_path: 输出路径
	:return: 输出的prompt列表
	"""
	df = pd.DataFrame(get_data(data_path))
	prompts = []
	dlen = len(df)
	for index, row in df.iterrows():
		instruction = f"You are a data feature extraction expert. Using the provided data, extract the features."
		input_text = (
			f"timer voltage:{row['tim_voltage']}, "
			f"power voltage:{row['pow_voltage']}, "
			f"control voltage:{row['ctl_voltage']}, "
			f"dsp voltage:{row['dsp_voltage']}, "
			f"timer current:{row['tim_current']}, "
			f"power current:{row['pow_current']}, "
			f"control current:{row['ctl_current']}, "
			f"dsp current:{row['dsp_current']}, "
			f"timer temperature:{row['tim_temperature']}, "
			f"power temperature:{row['pow_temperature']}, "
			f"control temperature:{row['ctl_temperature']}, "
			f"dsp temperature:{row['dsp_temperature']},"
        )
		prompts.append({
			"instruction": instruction,
			"input": input_text,
		})
		if index % 100 == 0:
			print(f"Transforming No.{index} Data.....Total {dlen}")
	# 写数据
	write_data(prompts, output_path)
	return prompts
	
	
if __name__ == "__main__":
	data_path = './dataset/BC Data1_20250411_全速工作模式.bcd'
	output_path='./dataset/1533B.json'
	get_prompt(data_path, output_path)
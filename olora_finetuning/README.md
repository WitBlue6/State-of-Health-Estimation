# Battery SOH Estimation with LLM

## 1.Introduction
This repository contains the code for the Battery SOH Estimation with LLM. 

The repository is based on the open source code from https://github.com/PHM-Code/MWA-CNN/tree/main to add MWA_CNN layer to the model and the repo https://github.com/huggingface/peft to add LoRA layer to the model.

## 2.Environment
Operating System: MacOS 15.3.1  
Python Version: 3.11.11  
CPU: Apple M4 16GB  
GPU Accelerator: MPS  

The required packages are listed in the *requirements.txt* file.

## 3.Usage
To run the code, you need to install the required packages.
```bash
pip install -r requirements.txt
```
First of all, you need to **transfer the dataset to the format of the model**.
```bash
python data_process.py
```
Then, you can run the code with the following command to **train the LLM TransformerCNN model with LoRA layer**.
```bash
python MWN_CNN.py
```
Or you can run the code with the following command to **train the Simple LLM model with LoRA layer**.
```bash
python olora_finetuning.py
```

After training, you can run the code with the following command to evaluate the model.
```bash
python modelWithCNN_predict.py
```

## 4.Results
The results of the evaluation are shown in the following table.
```bash
########Begining Output########
Following the Instruction below, give me your Response.
			### Instruction:
			You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on temperature, current, and voltage:[24.0, 5.655, 3.7172872].And give me the reason for your estimation.
			### Input:
			24.0, 5.655, 3.7172872
			### Response:
			SOH is 98.931%.Because The Voltage is too low.
########End Output########
########Begining Target########
SOH is 99.399%.Because The Current is too high.
########End Target########
```

As is shown in the table, the model can estimate the SOH of a lithium-ion battery based on temperature, current, and voltage.

## 5.Promblems in the Model

* The estimated SOH may be not accurate. The estimated SOH is always <mark>between 98.73% and 98.99%</mark>. It seems that the model is not performing well on the dataset.

* The model will **not give me the right reason for my estimation**. Because the dataset does not contain the reason for the estimation. And the reasons in training are randomly generated. So the model may not know the reason for the estimation.

* Because of limited memory and resources, **the model parameters are not large enough**.

## 6.Future work

1. Try to fix the model to perform well on the dataset.

2. Use larger model parameters to estimate the SOH of a lithium-ion battery.

## 7.Update

2024.04.08  
Add CallBack function to the model to record and plot loss-epoch curve.

2025.04.07  
Fix the problem of the model repeating the same response by changing the way to generate the prompt. And train the model with given reasons by randomly generating the reasons.

2025.04.06  
Fix the promblem of mismatch of the key in the model.

2025.04.04  
Add the code to estimate the SOH of a lithium-ion battery with LLM.


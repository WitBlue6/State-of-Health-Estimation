# Battery SOH Estimation with LLM

## 1. Introduction
This repository contains the code for the Battery SOH Estimation with LLM. 

The repository tries to estimate with <mark>**_Pythia base model_**</mark> combined with <mark>**_LoRA layer_**</mark> and <mark>**_Multilayer wavelet attention CNN_**</mark>.

The repository is based on the open source code from https://github.com/PHM-Code/MWA-CNN/tree/main to add MWA_CNN layer to the model and the repo https://github.com/huggingface/peft to add LoRA layer to the model.

## 2. Environment
Operating System: MacOS 15.3.1  / Windows 11  
Python Version: 3.11  
CPU: Apple M4 16GB  / Intel i5 16GB  
GPU Accelerator: MPS / CUDA 12.3 (NVIDIA GeForce RTX 3050 Laptop GPU)  

The required packages are listed in the *requirements.txt* file.

## 3. Usage
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

## 4. Results
The results of the evaluation are shown in the following table.
```bash
########Begining Output########
Following the Instruction below, give me your Response.
			### Instruction:
			You are a SOH estimation expert. Estimate the SOH of a lithium-ion battery based on peak dV/dt, discharge duration, discharge capacity, and average discharge current: [0.009616, 3074.52, 0.937766, -1.0892]. In addition, you need to give me the reason for your estimation.
			### Input:
			Peak dV/dt: 0.009616 V/s, Discharge Duration: 3074.52 s, Discharge Capacity: 0.937766 Ah, Average Discharge Current: -1.0892 A
			### Response:
			SOH is 80.35%. Because row['Reason']
########End Output########
########Begining Target########
SOH is 82.37%. Because row['Reason']
########End Target########
```

As is shown in the table, the model can estimate the SOH of a lithium-ion battery based on temperature, current, and voltage.

## 5. Promblems in the Model

* The model will ***not give me the right reason for my estimation***. Because the dataset does not contain the reason for the estimation. And the reasons in training are randomly generated. So the model may not know the reason for the estimation.

* The model use 4 different features to estimate the SOH. But the freatures may be ***difficult to be measured in actual situation***.

## 6. Future work

* Try to get right reasons for the estimation.

* Try to use more simple features to estimate the SOH.

## 7. Update
* 2025-04-23  
Change the Dataset used in the model.

* 2025-04-13  
Change the Dataset used to train the model and extract 4 different features ***(Peak dV/dt, Discharge Duration, Discharge Capacity and Average Discharge Current)*** to estimate SOH. Train the model with 1b parameters. 

* 2025-04-09  
Port the code to Windows framework. Train the model with 160m parameters.

* 2025-04-08  
Add CallBack function to the model to record and plot loss-epoch curve.

* 2025-04-07  
Fix the problem of the model repeating the same response by changing the way to generate the prompt. And train the model with given reasons by randomly generating the reasons.

* 2025-04-06  
Fix the promblem of mismatch of the key in the model.

* 2025-04-04  
Add the code to estimate the SOH of a lithium-ion battery with LLM.

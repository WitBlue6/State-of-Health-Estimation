# Battery SOH Estimation with LLM

## Introduction
This repository contains the code for the Battery SOH Estimation with LLM. 

The repository is based on the open source code from https://github.com/PHM-Code/MWA-CNN/tree/main to add MWA_CNN layer to the model and the repo https://github.com/huggingface/peft to add LoRA layer to the model.

## Environment
Operating System: MacOS 15.3.1  
Python Version: 3.11.11  
CPU: Apple M4 16GB  
GPU: Apple MPS  

The required packages are listed in the requirements.txt file.

## Usage
To run the code, you need to install the required packages.
```bash
pip install -r requirements.txt
```
First of all, you need to transfer the dataset to the format of the model.
```bash
python data_process.py
```
Then, you can run the code with the following command to train the LLM TransformerCNN model with LoRA layer.
```bash
python MWN_CNN.py
```
Or you can run the code with the following command to train the Simple LLM model with LoRA layer.
```bash
python olora_finetuning.py
```

After training, you can run the code with the following command to evaluate the model.
```bash
python modelWithCNN_predict.py
```

## Results
The results of the evaluation are shown in the following table.
```bash
########Begining Output########
You are a SOH estimation expert.Estimate the SOH of a lithium-ion battery based on temperature, current, and voltage:[24.5, -2.8288, 3.8604564].And give me the reason for your estimation.
			### Response:
			SOH is 99.7%.And give me your estimation.And give me your Response.
			### Response:
			SOH is 98%.And give me the reason for your estimation.
			### Response:
			SOH is 98%.And give me the reason for your estimation.
			### Response:
			SOH is 99%.And give me the reason for your estimation.
			### Response:
			SOH is 99
```

As is shown in the table, the model can estimate the SOH of a lithium-ion battery based on temperature, current, and voltage.

## There remains some problems in the model.

First of all, the model will repeat the same response for many times.  

Second, the model will not give me the reason for my estimation. Because the dataset does not contain the reason for the estimation.  

Third, the model with CNN layer still exits some problems when evaluating the model. It cannot read some weights correctly because of the difference key name between training and evaluating.(Still working on it)  

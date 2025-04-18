# State of Health Estimation with LLM

## Introduction
This repository contains the code for the SOH Estimation with LLM. 

Folder `olora_finetuning` tries to estimate with **_Pythia base model_**combined with **_LoRA layer_** and **_Multilayer wavelet attention CNN_**, based on the open source code from https://github.com/PHM-Code/MWA-CNN/tree/main to add MWA_CNN layer to the model and the repo https://github.com/huggingface/peft to add LoRA layer to the model.

Folder `soh_predictor` tries to estimate using Unsupervised Learning with **_Autoencoder_** and **_LLM_**.

## Environment
Operating System: MacOS 15.3.1  / Windows 11  
Python Version: 3.11  
CPU: Apple M4 16GB  / Intel i5 16GB  
GPU Accelerator: MPS / CUDA 12.3 (NVIDIA GeForce RTX 3050 Laptop GPU)  

The required packages are listed in the *requirements.txt* file.


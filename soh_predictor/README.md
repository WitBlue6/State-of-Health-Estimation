# Unsuperviesed SOH estimation using LLM

## 1 Introduction  

This is a project to estimate the state of health (SOH) of a electric device using <mark>LLM</mark>. The SOH is estimated using the following steps:  
- Using pretrained LLM to extract the features from the device data.  
- Training the <mark>**Autoencoder**</mark> model to study the features.
- Using the trained model to estimate the SOH by calculating the <mark>Reconstruction Error</mark> of the features.
## 2 Usage
### 2.1 Environment
```bash
pip install -r requirements.txt
```
### 2.2 Model Training  
The model is trained using the following command:
```bash
python train.py
```
### 2.3 SOH Estimation
The SOH is estimated using the following command:
```bash
python estimate.py
```
In the `estimate.py`, I have provided the Anomaly by `adding noise to the normal data.` So the results in `outputs` folder show the obvious anomaly in front of the normal data.
### 2.4 Kmeans Clustering
In the `clustering.py`, I have provided the Kmeans Clustering of the features to estimate the SOH. But the results seem to be **not good**. So I have not used it in the project. You can try it yourself.
```bash
python clustering.py
```
### 2.5 Estimate SOH with LoRA finetuned model
In the `generate_anomaly.py`, I have provided the code to generate anomaly data to LoRA model. So you can try to let the LLM model estimate the SOH and give the reason for the anomaly by itself.  
The only thing you need to do is to generate the anomaly data by yourself and go to `olora_finetuning` folder to finetune the model.
### 2.6 Model without pretrained LLM
In the `modelwithoutLLM.py`, I have provided the model without pretrained LLM. You can try it yourself.
```bash
python modelwithoutLLM.py
```
And the SOH estimation is achieved by the following command:
```bash
python predictwithoutLLM.py
```
The results are shown in the `outputs` folder.

## 3 Results
The results are shown in the `outputs` folder. The results are as follows:
- `outputs/soh_prediction.png`: The output plot of the SOH prediction.
- `outputs/best_model.pth`: The model used to estimate the SOH.
- `outputs/feature_scaler.pkl`: The model used to scale the features.
- `outputs/soh_predictionwithoutLLM.png`: The output plot of the SOH prediction without LLM.
- `outputs/best_modelwithoutLLM.pth`: The model used to estimate the SOH without LLM.
- `outputs/feature_scalerwithoutLLM.pkl`: The model used to scale the features without LLM. 

## 4 Update
- 2024-04-30  
Added new anomaly generation method.  
Use PID controller to control the threshold automatically.
- 2025-04-23  
Improved the model structure. Add the code to generate anomaly data to LoRA model.
- 2025-04-19  
Changed the feature generation method by pooling the last three hidden layers of the pretrained LLM output. Improved the model structure.
- 2025-04-18  
Added the Kmeans Clustering of the features to estimate the SOH.
- 2025-04-17  
Added the Code for Unsupervised SOH Estimation.
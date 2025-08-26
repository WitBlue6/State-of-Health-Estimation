# IFLOW.md

This file provides guidance to iFlow Cli when working with code in this repository.

## Project Overview

This is a State of Health (SOH) classification and prediction system for electronic devices. It uses machine learning models to detect anomalies in device behavior and predict potential component failures. The system includes modules for data preprocessing, feature engineering, SOH prediction using neural networks, fault classification, and anomaly detection with adaptive thresholding.

## Common Commands

### Running the SOH Model Training
```bash
python model.py --data_path ./dataset/无异常.txt --num_epochs 4000 --learning_rate 2e-5 --output_path ./outputs
```

### Running SOH Detection and Prediction
```bash
python predict.py --data_path ./dataset/无异常.txt --threshold 0.86 --output_path ./outputs
```

### Running Tests
```bash
python test.py
```

### Data Processing Scripts
```bash
python data_analyze.py
```

## Code Architecture and Structure

### Core Components

1. **model.py**: Contains the main training logic for the SOH prediction model. It includes:
   - Data loading and preprocessing
   - Feature standardization
   - SOH prediction model architecture (SOHPredictor)
   - Custom loss functions (ModuleAwareLoss)
   - Training loop with validation

2. **utils.py**: Utility functions and core classes:
   - AnomalyProcessor: For adding noise and other data transformations
   - SOHDetector: Main detection class with adaptive thresholding and RUL estimation
   - SOHPredictor: Neural network architecture for SOH prediction
   - ClassificationModel: For fault classification
   - Data transformation functions (GPS_relative, Euler_relative, Battery_relative)
   - Log compression and analysis functions

3. **predict.py**: Contains the prediction pipeline:
   - Loading trained models
   - Running SOH detection on new data
   - Visualization of results
   - Integration with LLM for analysis and recommendations

4. **data_analyze.py**: Data analysis and visualization tools

### Data Structure

The dataset contains multiple text files with time-series data representing device states:
- Normal operation data (无异常.txt)
- Specific fault data (舵机1故障.txt, 电源故障.txt, 北斗故障.txt, etc.)

Each file contains rows of comma-separated values representing 40 features including:
- Motor data (24 features)
- IMU data (9 features: Accel, Angular velocity, Euler angles)
- Power data (4 features: Voltage, Current, Power, Battery)
- GPS data (3 features: longitude, latitude, altitude)

### Key Features

1. **Adaptive Thresholding**: The system uses a PID controller to automatically adjust detection thresholds based on historical data.

2. **Module-based Analysis**: The system identifies specific failing modules (motors, IMU axes, power, GPS) rather than just detecting general anomalies.

3. **Remaining Useful Life (RUL) Estimation**: Predicts how long the device will remain operational based on health trends.

4. **LLM Integration**: Uses large language models to analyze logs and provide actionable recommendations.

5. **Feature Engineering**: Includes transformations for GPS coordinates, Euler angles, and battery levels to relative values.

### Model Architecture

1. **SOH Prediction Model**: Encoder-decoder neural network with residual connections for reconstructing input features and detecting anomalies based on reconstruction error.

2. **Classification Model**: Multi-layer neural network for classifying which specific module is failing.

3. **Loss Function**: Custom ModuleAwareLoss that weights different modules differently based on their importance.

### Development Notes

- The project uses PyTorch for deep learning components
- Scikit-learn for preprocessing and some ML algorithms
- Matplotlib for visualization
- ChromaDB for RAG (Retrieval-Augmented Generation) functionality
- OpenAI API integration for LLM-based analysis
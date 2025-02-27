# Potato Leaf Disease Classification

A deep learning model to classify potato leaf diseases using TensorFlow/Keras.

## Features
- Classification of potato leaf diseases into three categories:
  - Early Blight
  - Late Blight
  - Healthy

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mangesh2904/potato-leaf-disease.git
   cd potato-leaf-disease
   ```

2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
1. Download the dataset from [Kaggle Potato Disease Dataset](https://www.kaggle.com/datasets/rizwan123456789/potato-disease-leaf-datasetpld)
2. Extract and organize the dataset in the following structure:
   ```
   dataset/
   ├── train/
   │   ├── Early_Blight/
   │   ├── Late_Blight/
   │   └── Healthy/
   └── validation/
       ├── Early_Blight/
       ├── Late_Blight/
       └── Healthy/
   ```

## Training the Model
1. Run the training script:
   ```bash
   python train_model.py
   ```
2. The trained model will be saved as 'potato_leaf_model.h5'

## Model Files
The trained model file is not included in the repository due to size limitations. You can:
1. Train the model yourself using the instructions above

## Project Structure
- `train_model.py`: Script to train the CNN model
- `app.py`: Application interface
- `requirements.txt`: Project dependencies

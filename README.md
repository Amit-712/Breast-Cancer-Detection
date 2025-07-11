# 📊 Project Tumor Trace: MRI-Based AI for Breast Cancer Detection

## 🔍 Project Overview
The **Tumor Trace** project is focused on creating an AI model that detects breast cancer through MRI images. By utilizing deep learning methods, the aim is to deliver precise predictions that can aid radiologists in making early diagnoses.

## 📑 Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Data Collection and Preprocessing](#Data-collection-and-preprocessing)
- [Model Architecture](#Model-architecture)
- [Training](#Training)
- [Evaluation Metrics](#Evaluation-metrics)
- [Results and Impact](#Results-and-impact)
- [Future Work](#Future-work)
- [License](#License)

## ⚙️ Installation
To get this project up and running, one has to install the required libraries. It can be done by running the following command

```shell
pip install -r requirements.txt
```

## 🚀 Usage
1. Clone the repository
    ```shell
    git clone https://github.com/Amit-712/Breast-Cancer-Detection.git
    cd Breast-Cancer-Detection
    ```

2. Place your MRI dataset in the specified directory.

3. Run the Jupyter Notebook:
    ```shell
    jupyter notebook
    ```

4. Follow the instructions in the notebook to preprocess data, train the model, and evaluate its performance.

## 📊 Data Collection and Preprocessing
The dataset comprises MRI images that are labeled as either malignant or benign. The data was sourced from  "Computers in Biology and Medicine", ensuring a wide variety of cases.

The preprocessing steps included:
- Converting images to grayscale.
- Resizing images to a standard dimension (ie. 224x224).
- Normalizing pixel values to a range suitable for neural network input.
- Augmenting the dataset using techniques such as rotation and flipping.
- Applying class weights to address imbalanced data.

## 🏗️ Model Architecture
The model employs a Convolutional Neural Network (CNN) architecture that is tailored to effectively extract features from MRI images. We tested model VGG16.

### VGG16:
- Pre-trained VGG16 model fine-tuned for binary classification (benign vs. malignant).

## 🏋️ Training
We trained our models using the following setup:
- best_accuracy = 0
- total_epochs = 25
- momentum = 0.9
- no_cuda = False
- log_interval = 10
- optimizer = optim.AdamW(model.parameters(), lr=0.00005, weight_decay=1e-4)
- scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

## 📏 Evaluation Metrics
We evaluated our models using the following metrics:
- **Accuracy**: Percentage of correct predictions.
- **Precision**: Ability to avoid false positives.
- **Recall (Sensitivity)**: Ability to identify actual positives.
- **F1 Score**: Harmonic mean of precision and recall.
- **AUC (Area Under Curve)**: Measures the model's ability to distinguish between classes.
- **Confusion Matrix**: Evaluates the model's performance in terms of true positives, false positives, true negatives, and false negatives.

## 📈 Results and Impact
### Model Performance:
- **VGG16**:
    - Accuracy: 80.11%
    - AUC: 0.8660

## 🔮 Future Work
- **Data Augmentation and Enrichment**: Incorporating more diverse and larger datasets.
- **Model Optimization**: Experimenting with advanced neural network architectures.
- **Integration with Medical Systems**: Developing APIs for integration with existing systems.
- **User Interface Improvements**: Enhancing usability and accessibility.
- **Real-time Inference and Deployment**: Optimizing for real-time inference and deploying on cloud platforms.
- **Explainability and Transparency**: Implementing techniques for model explainability.

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

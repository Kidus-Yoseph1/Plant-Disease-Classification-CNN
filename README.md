# Plant Disease Classification using CNN

## Objective
The goal of this project is to develop a robust system for identifying plant diseases from leaf images. Early detection of plant diseases is critical for agricultural productivity and food security, allowing farmers to take timely action to prevent crop loss.

## Solution
This project utilizes transfer learning with the EfficientNet-B0 architecture. The model is pre-trained on the ImageNet dataset, and its feature extraction layers are frozen. A custom classifier head is added and trained on a specialized dataset of plant diseases. This approach leverages powerful pre-trained features while adapting the model to the specific nuances of botanical pathologies.

## Technologies Used
- PyTorch: Deep learning framework for model building and training.
- Torchvision: Image processing and pre-trained models.
- Streamlit: Web-based interface for easy model interaction and inference.
- Matplotlib: For visualizing training and validation metrics.
- EfficientNet-B0: State-of-the-art CNN architecture optimized for efficiency and accuracy.

## How to Run

### 1. Clone the Repository
``` Bash
git clone https://github.com/Kidus-Yoseph1/Plant-Disease-Classification-CNN.git
cd Plant-Disease-Classification-CNN
```

### 2. Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.
```bash
python3 -m venv venv
source venv/bin/activate
# On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
Install the required Python packages using the provided requirements file.
```bash
pip install -r requirements.txt
```

### 4. Training the Model
To train the model from scratch (or fine-tune the classifier), ensure your dataset is correctly placed in the data directory and run the trainer script.
```bash
python src/trainer.py
```
The trained model weights will be saved in the models directory.

### 5. Running the Application
Once the model is trained, you can launch the interactive web interface to classify images. Or use the trained model and run the app. 
```bash
streamlit run app.py
```
Upload an image of a plant leaf through the browser to see the predicted disease and the confidence score.

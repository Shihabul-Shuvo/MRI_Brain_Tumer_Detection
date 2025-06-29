# MRI Tumor Detection System

The MRI Tumor Detection System is a web-based application designed to assist medical professionals in identifying potential brain tumors in MRI scans using deep learning. This project leverages a Flask backend and a TensorFlow/Keras-based Convolutional Neural Network (CNN) model, fine-tuned from the VGG16 architecture, to provide diagnostic insights. The system classifies MRI images into four categories: glioma, meningioma, no tumor, and pituitary.

## Features
- **Tumor Detection**: Upload an MRI image to detect the presence of brain tumors.
- **Confidence Scoring**: Provides a confidence percentage for the diagnosis (up to 2 decimal places).
- **PDF Report Generation**: Generates a professional PDF report including patient details, diagnosis, confidence score, and the uploaded MRI image.
- **Clear Functionality**: Option to clear previously uploaded images and reset the interface.
- **Responsive Design**: Optimized for both desktop and mobile devices.

## Sample Images
![Sample MRI Image](https://github.com/Shihabul-Shuvo/MRI_Brain_Tumer_Detection/blob/main/uploads/Te-pi_0252.jpg)

## Project Preview
[Project Snapshot](https://github.com/Shihabul-Shuvo/MRI_Brain_Tumer_Detection/blob/main/uploads/project_preview.png)

## Generated PDF Report
[Generated Report](https://github.com/Shihabul-Shuvo/MRI_Brain_Tumer_Detection/blob/main/uploads/Te-pi_0252_report.pdf)

## Dataset Information
- **Source**: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Training Data**:
  - **glioma**: 1321 images
  - **meningioma**: 1339 images
  - **notumor**: 1595 images
  - **pituitary**: 1457 images
  - **Total**: 5712 images
- **Testing Data**:
  - **glioma**: 300 images
  - **meningioma**: 306 images
  - **notumor**: 405 images
  - **pituitary**: 300 images
  - **Total**: 1311 images
- **Split Percentage**:
  - Training: ~81.33%
  - Testing: ~18.67%

## Machine Learning Process
### Model Architecture
- **Base Model**: Utilizes VGG16, pre-trained on ImageNet, with the top layers removed for fine-tuning.
- **Custom Layers**: Adds GlobalAveragePooling2D, a Dense layer with 256 units and ReLU activation, a Dropout layer (0.5) for regularization, and a final Dense layer with 4 units and softmax activation for multi-class classification.
- **Training Strategy**: The base VGG16 layers are frozen initially, with fine-tuning applied to the custom layers to adapt to the MRI tumor classification task.

### Data Preprocessing
- **Image Resizing**: All images are resized to 224x224 pixels to match VGG16 input requirements.
- **Normalization**: Pixel values are normalized to the range [0, 1] by dividing by 255.
- **Data Augmentation**: Enhances model robustness with techniques such as brightness and contrast adjustments during training.

### Training Details
- **Environment**: Trained using TensorFlow/Keras in a Google Colab environment with GPU acceleration.
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Loss Function**: Categorical cross-entropy, suitable for multi-class classification.
- **Epochs and Batch Size**: Trained for 5 epochs with a batch size of 20.
- **Validation Split**: Training: ~81.33%, Testing: ~18.67%.
- **Training Progress**: At Epoch 5/5, achieved loss: 0.0886 and sparse_categorical_accuracy: 0.9626.

### Evaluation
- **Metrics**: Model performance is evaluated using accuracy (0.95 on test set).
- **Confusion Matrix**: Plotted to visualize the performance across all classes (glioma, meningioma, notumor, pituitary).
- **ROC Curve and AUC**: Computed Receiver Operating Characteristic (ROC) curve and Area Under the Curve (AUC) for each class to assess classification performance.

## Installation and Setup

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Setup Instructions
1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/mri-tumor-detection.git
   cd mri-tumor-detection
   ```

2. **Set Up a Virtual Environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the Model**
   - Ensure `model.h5` is in the `models` folder.

6. **Run the Application**
   ```bash
   python main.py
   ```
   - Open your browser and navigate to `http://127.0.0.1:5000`.

## Usage
1. **Sidebar Input**: On the left sidebar of the interface, enter the patient's details:
   - **Patient Name**: Input the patient's full name.
   - **Age**: Enter the patient's age (numeric value).
   - **Sex**: Select the patient's sex (e.g., Male/Female/Other from a dropdown or input field).
   The sidebar is designed for easy access to patient information, ensuring all data is captured before proceeding with the analysis.
2. **Image Upload**: Use the file input field (typically located below the sidebar or in the main panel) to upload an MRI image.
3. **Detection Process**: Click "Upload and Detect" to initiate the analysis and view the diagnosis and confidence score on the right panel.
4. **Download Report**: After analysis, download the PDF report for a detailed summary, which includes the entered patient details, diagnosis, and uploaded image. The report is generated using the ReportLab library and saved in the `uploads` directory.
5. **Reset Interface**: Use the "Clear" button to reset the sidebar inputs, uploaded image, and results panel.

## Project Structure
```
mri-tumor-detection/
├── main.py                # Flask application logic
├── models/                # Directory for the trained model
│   ├── model.h5           # Trained CNN model
│   └── Tumer_Detection_using_Deep_Learning.ipynb  # Model training notebook
├── templates/             # HTML templates
│   └── index.html         # Main interface
├── static/                # Static files
│   └── css/               # CSS styles
│       └── styles.css     # Custom styles
├── uploads/               # Project preview, uploaded files and PDFs
├── venv/                  # Virtual environment
├── requirements.txt       # Dependency list
└── README.md              # This file
```

## Contributing
Contributions are welcome! Please fork the repository and submit pull requests with improvements or bug fixes. For major changes, open an issue first to discuss.

## Acknowledgments
- Built with Flask, TensorFlow/Keras, and ReportLab.
- Inspired by the need for accessible diagnostic tools in healthcare.
- Dataset source: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

## Contact
For questions or support, please open an issue or email me.

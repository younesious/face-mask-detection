# Face Mask Detection using Image Processing and Deep Learning

This project aims to detect whether individuals in images are wearing face masks or not using image processing techniques and a deep learning model. The system detects faces in images and then classifies each detected face as either wearing a mask or not wearing a mask.

## Overview

The project consists of the following components:

1. **Face Detection:** Utilizes the Haar Cascade classifier to detect faces in images.
2. **Deep Learning Model:** Uses a modified VGG19 convolutional neural network (CNN) to classify faces into two classes: with mask and without mask.
3. **Data Augmentation:** Applies image augmentation techniques to the training data to improve model generalization.
4. **Training:** Trains the CNN model using the augmented training dataset and evaluates its performance on a validation set.
5. **Prediction:** Applies the trained model to classify faces detected in input images as wearing a mask or not.

## Getting Started

To run the project, follow these steps:

1. **Clone the repository: git clone https://github.com/younesious/face-mask-detection.git**

2. **Install dependencies:**
- Install Python dependencies by running:
  ```
  pip install opencv-python keras matplotlib
  ```
- Make sure you have the required dataset files and pre-trained weights for the VGG19 model.

3. **Run the mask.py script:**
- Modify the file paths in the script to point to the appropriate directories and files for the dataset and pre-trained weights.
- Execute the script:
  ```
  python mask.py
  ```

## Dataset

The dataset used in this project is the Face Mask Dataset, which contains images of people with and without face masks. It is divided into training and validation sets for model training and evaluation.
You can search for it simply in kaggle.
 
## Model Architecture

The deep learning model architecture used in this project is based on the VGG19 convolutional neural network. The pre-trained weights for VGG19 are used as a feature extractor, and a custom dense layer is added for classification.

## Results

The project produces visual outputs showing the original images with bounding boxes around detected faces, along with labels indicating whether each detected face is wearing a mask or not.

## Requirements

- Python 3.x
- OpenCV (cv2)
- Keras
- Matplotlib

## Contributing

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

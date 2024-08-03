# Second Homework: Artificial Neural Networks and Deep Learning

## Overview
In this homework, we will tackle two different computer vision challenges: Image classification and image segmentation.

**Note:** This homework will require GPU compute power. In Google Colab, go to 'Runtime' -> 'Change runtime type' and enable GPU support.

## Part 1: Image Classification

### Dataset
We will use the CIFAR-100 dataset, which contains 100 classes with 600 images each. CIFAR-100 is available as a Keras dataset:
[TensorFlow CIFAR-100 Dataset](https://www.tensorflow.org/api_docs/python/tf/keras/datasets).

### Save Your Work
Training models can take a long time. To avoid losing your progress, save your models to Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')

# Save model
model.save('/content/drive/path/to/location')

# Load model
from tensorflow import keras
model = keras.models.load_model('/content/drive/path/to/location')
```

### Submission
Your final submission should include both code and a written report. You can do this separately or use markdown cells in your notebook for clarity. Ensure your notebook is easy to follow and include relevant numbers and figures in your reports.

## Part 1: Image Classification

### 1.1 Warming Up
Use the following approaches to train a classifier on the CIFAR-100 dataset and discuss your results.

1. **Feature Extraction with VGG16**
   - Use VGG16 as a feature extractor and train a dense network for classification.
   - VGG16 is available in Keras: [Keras Applications](https://www.tensorflow.org/api_docs/python/tf/keras/applications).
   - Preprocess the images using `tf.keras.applications.vgg16.preprocess_input`.

2. **Fine-Tuning VGG16**
   - Fine-tune the upper block of VGG16 and add a dense classification layer.
   - Lower the learning rate for RMSprop.

3. **Training Full VGG16**
   - Train the full VGG16 network, making all convolutional blocks trainable.
   - Adjust the learning rate and batch size as needed.

### 1.2 Create Your Own Network from Scratch
Create and train a convnet from scratch. Use techniques learned from the book and iterate on your design choices. Incorporate advanced features such as:
- Keras callbacks (Early stopping, checkpointing)
- Learning rate schedule
- Batch normalization
- Depth-wise separable convolutions
- Residual connections
- Model ensembling
- Data augmentation

Perform at least 3 iterations of your model and discuss your results and design choices.

## Part 2: Image Segmentation

### Dataset
Use the PASCAL VOC-2012 dataset for this task. This dataset contains 20 different classes, but for simplicity, we will perform foreground vs background segmentation.

### Steps
1. **Download the Dataset**
   - Explore the data and plot a few images.

2. **Build a Segmentation Model**
   - Use the image segmentation example from the handbook as a starting point.
   - Consider the modified U-Net architecture from the TensorFlow image segmentation tutorial.

3. **Visualize Segmentation Output**
   - Compare the output with the ground truth reference.
   - Reflect on metrics to evaluate segmentation performance (e.g., Dice score).

4. **Experiment with Upsampling Methods**
   - Replace `Conv2DTranspose` with bilinear interpolation using `UpSampling2D` followed by a standard convolution.
   - Compare performance and visual differences in segmentation masks.

## Important Tips
- Save your work frequently to Google Drive.
- Use GPU for faster training times.
- Include both code and written explanations in your submission.
- Make sure your notebooks are easy to follow and well-documented.

## References
- Chollet, F. (2023). *Deep Learning with Python* (2nd ed.). Manning Publications.
- TensorFlow CIFAR-100 Dataset: [Link](https://www.tensorflow.org/api_docs/python/tf/keras/datasets)
- Keras Applications: [Link](https://www.tensorflow.org/api_docs/python/tf/keras/applications)

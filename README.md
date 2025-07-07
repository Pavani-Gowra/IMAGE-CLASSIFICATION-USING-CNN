# IMAGE-CLASSIFICATION-USING-CNN

As part of my Machine Learning internship at CodTech IT Solutions, my task involved building an image classification model using Convolutional Neural Networks (CNNs). The objective of this task was to understand how to work with image data, implement a CNN using deep learning libraries like TensorFlow or Keras, and evaluate its performance on a given dataset. I successfully completed the task using Python and popular libraries such as TensorFlow, Keras, NumPy, and Matplotlib.

**Tools and Technologies Used

-->Programming Language: Python

-->Development Environment: Jupyter Notebook

**Libraries:

--> TensorFlow/Keras – for building and training CNN models

--> NumPy – for handling arrays and numerical operations

--> Matplotlib – for visualizing images and training metrics

--> Sklearn – for evaluation metrics

**Dataset: I used the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. It is a commonly used benchmark dataset for image classification tasks.

**Task Workflow
1. Importing Required Libraries :
I began by importing all essential libraries, especially TensorFlow and Keras modules, which provide simple APIs for building deep learning models.

2. Loading and Preprocessing the Dataset :
The CIFAR-10 dataset was loaded directly using Keras’s built-in datasets. The data was split into training and testing sets. Images were normalized to values between 0 and 1 by dividing by 255. Labels were converted into one-hot encoded vectors using to_categorical().

3. Model Architecture :
I designed a Sequential CNN model using the Keras API. The architecture consisted of:

  Convolutional layers to extract features using filters

  MaxPooling layers to reduce dimensionality

  Dropout layers to prevent overfitting

  A Flatten layer to transform feature maps into a 1D vector

  A Dense output layer with softmax activation for multi-class classification

4. Model Compilation and Training :
I compiled the model with:

*Loss function: categorical crossentropy (suitable for multi-class classification)

*Optimizer: Adam (adaptive learning rate optimizer)

*Metrics: Accuracy

The model was trained using .fit() method over 10 epochs with batch size 64. I also used validation data to monitor overfitting.

5. Evaluation and Visualization
After training, I evaluated the model on the test set and plotted training/validation accuracy and loss using Matplotlib.

I visualized training vs validation accuracy and loss curves to analyze the learning behavior of the model and detect any overfitting or underfitting.

**Challenges and Learnings :

One of the main challenges was optimizing the model to improve accuracy without overfitting. By adjusting the architecture and including dropout layers, I achieved a balance between bias and variance. Additionally, understanding how convolutional layers learn spatial hierarchies of features in image data was a key learning milestone.

This task introduced me to the core principles of deep learning for computer vision, especially the power of CNNs in capturing visual patterns like edges, textures, and objects in images.

**Conclusion :

This task gave me a practical understanding of how to use CNNs for image classification. I learned how to preprocess image data, design and train CNN models, and interpret performance metrics. Successfully completing this task helped me gain confidence in applying deep learning techniques for real-world vision tasks.

This experience not only enhanced my programming skills but also deepened my knowledge in computer vision, an essential area of AI and machine learning.


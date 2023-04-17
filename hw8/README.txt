Calicia Perea 
HW8- Machine Learning 
Readme.txt

This document describes a Convolutional Neural Network (CNN) model named MyCNN, which is trained on the MNIST dataset. The architecture of the CNN contains two convolutional layers, two max pooling layers, and a fully connected layer. The first convolutional layer has a kernel size of 3x3, a stride of 1x1, and a valid padding mode. It outputs 4 channels. The first pooling layer performs max pooling with a pooling size of 2x2 and strides of 2x2. The second convolutional layer has a kernel size of 3x3, a stride of 3x3, and a valid padding mode. It outputs 2 channels. The second pooling layer performs max pooling with a pooling size of 4x4 and strides of 4x4. The fully connected layer has an output size of 10.

The code includes two plots that show the train and validation loss and accuracy over the 10 epochs. The first plot shows the train and validation loss, while the second plot shows the train and validation accuracy. Both plots demonstrate that the model's performance improves with more epochs of training.

In addition, the code also includes an example of how to modify the code to change the kernel size of the first convolutional layer from 3x3 to 5x5. The resulting CNN model achieves a slightly higher accuracy on the MNIST dataset than the original CNN model with a kernel size of 3x3.

Overall, this document provides a useful overview of a CNN model and its performance on the MNIST dataset, as well as an example of how to modify the code to experiment with different CNN architectures.

I used Google Colab to run:
Using PyTorch 

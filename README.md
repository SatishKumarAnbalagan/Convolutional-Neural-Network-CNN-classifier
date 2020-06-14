# Convolutional-Neural-Network-CNN-classifier
Convolutional Neural Network (CNN) classifier in Python with the Pytorch framework including backpropagation and gradient descent. The dataset used is CIFAR-10.  Used Softmax and SVM classifiers. You may need these packages: Pytorch, TensorFlow, NumPy, and OpenCV (for reading images). Optimization techniques such as mini-batch, batch normalization, dropout and regularization is used. In the first CONV layer, the filter size is 5*5, the stride should be 1, and the total number of filters is 32. Visualization of the first CONV layer output in the trained model for each filter (i.e., 32 visualization results) are saved.

# Train
"python CNNclassify.py train" to train your neural network classifier and generate a model in the model folder.

# Testing
"python CNNclassify.py test xxx.png" to
(1) predict the class of an image and display the prediction result;
(2) save the visualization results from the first CONV layer as “CONV_rslt.png”.

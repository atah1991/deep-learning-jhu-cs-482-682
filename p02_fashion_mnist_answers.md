# deep-learning-jhu-cs-482-682 p02 Fashion-MNIST


## Link to this submission's github repository

[https://github.com/deep-learning-jhu/deep-learning-jhu-cs-482-682](https://github.com/deep-learning-jhu/deep-learning-jhu-cs-482-682)

## Name, JHED ID, Github ID of each student

 -

# Answers
### Question 1
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/default_mnist_vs_fashion.png)

<<<<<<< HEAD
TODO answer all questions

## Question 1

## Question 2

## Question 3

## Question 4

## Question 5

## Question 6

## Question 7

## Question 8

## Question 9

## Question 10

## Question 11

## Question 12

## Question 13

## Question 14

## Question 15
=======
Blue - Fashion MNIST, Orange - MNIST

As we see default setting for CNN is performing much better on MNIST (~97%) dataset rather than on Fashion MNIST(~87%). Looking at validation accuracies and losses we can conclude that validation curves are not saturated, so there is a room for improvement with increasing number of epochs (we will see that in the next question). THere is no sign of overfitting, since the validation accuracy is still decreasing with training accuracy.

### Question 2
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/default_mnist_vs_fashion_epochs_20.png)

Blue - Fashion MNIST, Orange - MNIST

As we mentioned, here validation accuracy gets to plateau and overall results are a bit better (~98% and ~85%) for MNIST and Fashion MNIST respectively. Increasing number of epochs, we allow gradient descent to converge further until more or less stable local minimum. Still no sign of overfitting.

### Question 3
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_epochs_20_lr_0.1_0.01_0.001.png)

Blue - lr = 0.001, Orange - lr = 0.1, Red - lr = 0.01

We know that learning rate is reponsible for the speed of convergence of gradient descent method in optimization. It is easily seen that at lr = 0.001 that is small comparably to other learning rates, the algorithm did not converge entirely. Validation loss is not saturated as well as validation accuracy. Training loss didn't stabilize as well, so it seems that convergence could have taken place further if we increase the number of epochs. At lr = 0.01 and 0.1 the performance looks much better.

### Question 4
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_optimizers.png)

Blue - Adam, Orange - SGD, Red - RMS

According to the results Adam is showing best results due to, probably, momentum, and its whole technique of adaptive learning. Interesting to note that on training stage all optimizers are performing roughly the same, while on validation stage RMS is significantly better than SGD.

### Question 5
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_dropouts.png)

Magenta - p = 1.0, Cyan - p =0.9, Red - p = 0.5, Blue - p = 0.25, Orange - p = 0

Dropout is one of the popular techniques to fight the overfitting problem. Clearly can be seen that at large values of p, the nodel definitely starts underfitting, since turning off almost all of neurons or all neurons, the model becomes extremely simple and unable to learn better.

### Question 6
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_batch_sizes.png)

Blue - 32, Orange - 256, Red - 2048

Results make sense. Smaller batch size faster convergence (faster training), but on the cost of precision of optimum. So at batch size 32 the training loss is very noisy, but convergence takes place. At large batch size 0f 2048 we are close to the batch gradient descent method. Curves are very smooth, but it takes much more time to get to convergence, that's why don't see saturation of red curves. 

### Question 7
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_channels_double_half.png)

Blue - Double channels, Orange - Half channels

If we think in simple terms, then the network with half channels is simpler network that could lead to underfitting while the network with double channels should tend to overfit. From plots we see that both validation losses are not saturated, first. Secondly, the network with double channels performing better, which makes sense since increasing depth or width of network should show better validation accuracies up to some point

### Question 8
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_batch_norm.png)

Blue - BatchNorm, Orange - Default

Batch normalization generally is good for faster learning and smoother gradient values during backpropagation. It is well known, that batch norm overall increases the performance of neural nets. We see this on validation accuracy plots. 

### Question 9 and 10
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_batch_norm_dropouts.png)

We observe no difference between the neural nets where the order of dropout layer and batch norm layer are swapped. One assumption about it can be that batch norm is reducing overfitting as well as dropout, so if batch norm or dropout layer increase the overall accuracy separately, then the next layer wouldn't make any difference.

### Question 11
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_extra_conv.png)

Red - Extra Convolution, Blue - Default

I added one more convolution with different kernel size and I didn't tune the networks enough. But with default setting of hyperparameters we see the poor performance of CNN with extra convolution

### Question 12
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_remove_layer.png)

Red - Removed Layer, Blue - Default

I removed the second convolution layer and we see that this has almost no effect on the results. One of the reason I chose the second convolution layer is the assumption that the first convolution layer already indentifies main features in order to recognize the object.

### Question 13
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/fashion_mnist_ultimate.png)

Ultimate model, where I used 4 Convolutional layers with Batch Normalization and 3 Fully Connected Layers. For each layer dropout is applied. According to the plots, we reach the best validation accuracy (> 92%). Training loss and accuracies are noisy due to batch size chosen to be 64. Training time increased comparably to default model, since my model is more complex and deeper. It takes about 40 mins to run on CPU on regular laptop. 

Choosing the model was limited by several factors. First it shouldn't be too deep and complicated since the run should take less than 90 mins on regular CPU. Second I chose model to be complex enough with many regularization technique to overcome overfitting.

The most significant changes here are BatchNormalization and Dropout techniques after each layer. Algorithmically BatchNorm and dropout are regularization techniques and helps to train the model faster. On top of that, due to batch norm, we have smoothing effect of gradient calculation during backpropagation (mathematically). The best performance of the model was saved in 'best_model.pt' file

### Question 14
Running pretrained model on MNIST dataset with best parameters obtained during training on Fashion MNIST data gives us poor results of ~15%. To run the model we need to pass the argument in command line --transfer True.

### Question 15
After several of epochs we are getting around 99% of accuracy which 2% improvement over training the model from scratch on MNIST dataset.
>>>>>>> f591d9840d380583fc5d25c21c85daee876dfec2

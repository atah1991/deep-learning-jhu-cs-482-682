# deep-learning-jhu-cs-482-682

## Names

# Answers
### Question 1
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/default_mnist_vs_fashion.png)

Blue - Fashion MNIST, Orange - MNIST

As we see default setting for CNN is performing much better on MNIST (~97%) dataset rather than on Fashion MNIST(~87%). Looking at validation accuracies and losses we can conclude that validation curves are not saturated, so there is a room for improvement with increasing number of epochs (we will see that in the next question)

### Question 2
![alt_text](https://github.com/deep-learning-jhu/p02-fashion-mnist-team7/blob/master/screenshots/default_mnist_vs_fashion_epochs_20.png)

Blue - Fashion MNIST, Orange - MNIST

As we mentioned, here validation accuracy gets to plateau and overall results are a bit better (~98% and ~85%) for MNIST and Fashion MNIST respectively. Increasing number of epochs, we allow gradient descent to converge further until more or less stable local minimum.

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



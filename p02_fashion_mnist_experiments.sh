
set -e
set -u
set -x

# Question 1 Default settings
python p02_fashion_mnist.py --dataset mnist --epochs 10 --name q1_default_mnist --data_dir ../data/q1
python p02_fashion_mnist.py --dataset fashion_mnist --epochs 10 --name q1_default_fashion_mnist  --data_dir ../data/q1

# Question 2
python p02_fashion_mnist.py --dataset mnist --epochs 20 --name q2_20_epochs --data_dir ../data/q2
python p02_fashion_mnist.py --dataset fashion_mnist --epochs 20 --name q2_20_epochs --data_dir ../data/q2


# Question 3
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.1 --name q3_20_epochs_lr_0.1 --data_dir ../data/q3
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.001 --name q3_20_epochs_lr_0.001 --data_dir ../data/q3
python p02_fashion_mnist.py --dataset fashion_mnist --lr 0.01 --name q3_20_epochs_lr_0.01 --data_dir ../data/q3

# Question 4
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer sgd --name q4_SGD --data_dir ../data/q4
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer adam --name q4_Adam --data_dir ../data/q4
python p02_fashion_mnist.py --dataset fashion_mnist --optimizer rmsprop --name q4_RMS --data_dir ../data/q4

# Question 5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0 --name q5_dropout_0 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.25 --name q5_dropout_0.25 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.5 --name q5_dropout_0.5 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 0.9 --name q5_dropout_0.9 --data_dir ../data/q5
python p02_fashion_mnist.py --dataset fashion_mnist --dropout 1 --name q5_dropout_1 --data_dir ../data/q5

# Question 6
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 32 --name q6_batch_size_32 --data_dir ../data/q6
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 256 --name q6_batch_size_256 --data_dir ../data/q6
python p02_fashion_mnist.py --dataset fashion_mnist --batch-size 2048 --name q6_batch_size_2048 --data_dir ../data/q6

# Question 7
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q7HalfChannelsNet --name q7_half_channels --data_dir ../data/q7
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q7DoubleChannelsNet --name q7_double_channels --data_dir ../data/q7

# Question 8
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q8BatchNormNet --name q8_batch_norm --data_dir ../data/q8

# Question 9
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q9DropoutNet --name q9_batch_dropout --data_dir ../data/q9

# Question 10
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q10DropoutBatchnormNet --name q10_dropout_batch --data_dir ../data/q10

# Question 11
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q11ExtraConvNet --name q11_extra_conv_net --data_dir ../data/q11

# Question 12
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q12RemoveLayerNet --name q12_remove_layer_net --data_dir ../data/q12

# Question 13
python p02_fashion_mnist.py --dataset fashion_mnist --model P2Q13UltimateNet --lr 0.001 --optimizer adam --batch-size 128 --name q13_ultimate_model --data_dir ../data/q13

# ...and so on, hopefully you have the idea now.

# TODO You should fill this file out the rest of the way!

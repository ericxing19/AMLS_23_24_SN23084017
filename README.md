# AMLS_assignment23_24-

This is a ucl project completed by ruowei xing

You must install all the packages referred to below:
numpy,
panda,
matplotlib,
scikit-learn,
seaborn,
medmnist,
cv,
torch/pytorch,
copy,
os,

python version used is  3.10.13

Dataset needs to be downloaded from website, whose file format is '.npz'

You can use main.py to run the modelA and modelB by using main_A() and main_B() function. 
The parameter in this function is:

1. model name: determining which model you will use (select one from three architectures: CNN, CNN2, CNN3)

2. dataset: dataset used in training, validation and testing. Don't change it!!!

3. l2_lambda: weight of l2 regularization

3. lr: learning rate, default 0.0001

4. lr_decay_rate.

SVM algorithms are not included in main.py because they are very east.(You can find them in the modelA.ipynb in file A or modelB.ipynb in file B)
If you want to get more detailed information can see some clear results, you can try to use the modelA.ipynb in file A or modelB.ipynb in file B. 


In addition, you can directly use the model trained already (for Task B, because it costs a lot of time) by using main_read_B() function.





Model for task A:
CNN: Two convolutional layers and pooling layers, followed by two fully connected layers.
CNN2: Three convolutional layers and two pooling layers, followed by two fully connected layers.
CNN3: Two convolutional layers and pooling layers, followed by one fully connected layers

Model for task B:
CNN: Two convolutional layers and 2 pooling layers, followed by two fully connected layers.
CNN2: Three convolutional layers and 2 pooling layers, followed by two fully connected layers.
CNN3: Four convolutional layers and 2 pooling layers, followed by two fully connected layers

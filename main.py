from A.modelA import main_A
from B.modelB import main_B
import sklearn
import os
import torch
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

# Task A data processing
# Get the data path
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
path = os.path.abspath(os.path.join(current_directory, 'Datasets/pneumoniamnist.npz'))
print(path)

def data_process(path):
    # Get the dataset
    data = np.load(path)
    # Find the keys oof the dataset
    print(data.keys())
    # Extracting features and labels
    # Nomalization: dividing all features by 255
    train_X = data['train_images']/255.0
    train_y = data['train_labels']
    val_X = data['val_images']/255.0
    val_y = data['val_labels']
    test_X = data['test_images']/255.0
    test_y = data['test_labels']
    return train_X, train_y, val_X, val_y, test_X, test_y

# Determine whether to save the image plotted (These images are used to write report)
# It should be false when you run the program
save_image = False

# set the random seed to 42
torch.manual_seed(42)
train_X, train_y, val_X, val_y, test_X, test_y = data_process(path)


# create train_loader for training
train_tensor_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1)

# One-hot encoder to process y (This is used for CrossEntropyLoss)
encoder = OneHotEncoder(sparse=False)

train_tensor_y = torch.tensor(encoder.fit_transform(train_y))
print(train_tensor_X.shape)
train_loader = DataLoader(TensorDataset(train_tensor_X, train_tensor_y), batch_size = 64, shuffle = True)
print(train_loader)

# create validation_loader for validation
val_tensor_X = torch.tensor(val_X, dtype=torch.float32).unsqueeze(1)
val_tensor_y = torch.tensor(encoder.fit_transform(val_y))
val_loader = DataLoader(TensorDataset(val_tensor_X, val_tensor_y), shuffle = False)

# create test_loader for testing
test_tensor_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1)
test_tensor_y = torch.tensor(encoder.fit_transform(test_y))
test_loader = DataLoader(TensorDataset(test_tensor_X, test_tensor_y), shuffle = False)

print(train_X.shape)

data_setA = [train_loader, val_loader, test_loader, test_tensor_X, test_tensor_y]

###################################
# Task B data processing
path = os.path.abspath(os.path.join(current_directory, 'Datasets/pathmnist.npz'))
print(path)

def data_process(path):
    # Get the dataset
    data = np.load(path)
    # Find the keys oof the dataset
    print(data.keys())
    # Extracting features and labels
    # Nomalization: dividing all features by 255

    train_X = data['train_images']
    train_y = data['train_labels'].squeeze()
    val_X = data['val_images']
    val_y = data['val_labels'].squeeze()
    test_X = data['test_images']
    test_y = data['test_labels'].squeeze()

    return train_X, train_y, val_X, val_y, test_X, test_y

# Determine whether to save the image plotted (These images are used to write report)
# It should be false when you run the program
save_image = False

# set the random seed to 42
torch.manual_seed(42)
train_X, train_y, val_X, val_y, test_X, test_y = data_process(path)

# create train_loader for training
train_tensor_X = torch.tensor(train_X, dtype=torch.float32).permute(0,3,1,2)

train_tensor_y = torch.tensor(train_y)
print(train_tensor_X.shape)
train_loader = DataLoader(TensorDataset(train_tensor_X, train_tensor_y), batch_size = 128, shuffle = True)

# create validation_loader for validation
val_tensor_X = torch.tensor(val_X, dtype=torch.float32).permute(0,3,1,2)
val_tensor_y = torch.tensor(val_y)
val_loader = DataLoader(TensorDataset(val_tensor_X, val_tensor_y), shuffle = False)

# create test_loader for testing
test_tensor_X = torch.tensor(test_X, dtype=torch.float32).permute(0,3,1,2)
test_tensor_y = torch.tensor(test_y)
test_loader = DataLoader(TensorDataset(test_tensor_X, test_tensor_y), shuffle = False)

print(train_X.shape)
print(test_tensor_X.shape)

data_setB = [train_loader, val_loader, test_loader, test_tensor_X, test_tensor_y]

# the first hyper-parameter: model name should be 'CNN', 'CNN2' or 'CNN3', which are corresponding to the models in modelA and modelB !!!!!!!!!!!!!!!!!!
# you can customize l2_lambda, lr, lr_decay_rate
main_A('CNN2', data_setA)
main_B('CNN2', data_setB)

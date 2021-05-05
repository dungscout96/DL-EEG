#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.models as models
import torchvision.transforms as T
import numpy as np
import h5py
import os
import sys
import datetime
import csv

import torchvision
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[9]:


import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    def __init__(self, x, y, train, val):
        super(EEGDataset).__init__()
        assert x.shape[0] == y.size
        self.x = x
        self.y = [y[i][0] for i in range(y.size)]
        self.train = train
        self.val = val

    def __getitem__(self,key):
        return (self.x[key], self.y[key])

    def __len__(self):
        return len(self.y)

class Logger():
    def set_model_save_location(self, model_dir):
        self.model_dir = f"saved-model/{model_dir}"
        
    def set_experiment(self, experiment_name):
        self.experiment_name = experiment_name
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                            format=log_format, datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join('training-logs', f'log-{experiment_name}-{datetime.datetime.today()}.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
        self.writer = SummaryWriter(f"runs/{experiment_name}")
            
    def log(self, message=""):
        logging.info(message)
#         print(message)

    def save_model(self, model, info):
        torch.save(model.state_dict(), f"{self.model_dir}/model-{logger.experiment_name}-{info}")


# In[3]:


def load_data(path, role, winLength, numChan, srate, feature,version=""):
    transform = T.Compose([
        T.ToTensor()
    ])
    if version:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_x_{role}_{winLength}s_{numChan}chan_{feature}.mat", 'r')
    x = f[f'X_{role}']
    if feature == 'raw':
        x = np.transpose(x,(0,2,1))
        x = np.reshape(x,(-1,1,numChan,winLength*srate))
    elif feature == 'topo':
        samples = []
        for i in range(x.shape[0]):
            image = x[i]
            b, g, r = image[0,:, :], image[1,:, :], image[2,:, :]
            concat = np.concatenate((b,g,r), axis=1)
            samples.append(concat)
        x = np.stack(samples)
        x = np.reshape(x,(-1,1,x.shape[1],x.shape[2]))
    print(f'X_{role} shape: ' + str(x.shape))
    if version:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}_{version}.mat", 'r')
    else:
        f = h5py.File(path + f"child_mind_y_{role}_{winLength}s_{numChan}chan_{feature}.mat", 'r')
    y = f[f'Y_{role}']
    print(f'Y_{role} shape: ' + str(y.shape))
    dataset = EEGDataset(x, y, role=='train', role=='val')
    return dataset


# In[4]:


# Load EEG data
# path = '/expanse/projects/nemar/child-mind-dtyoung/'
path = './data/'
winLength = 2
numChan = 24
srate = 128
feature = 'topo'

train_data = load_data(path, 'train', winLength, numChan, srate, feature)
val_data = load_data(path, 'val', winLength, numChan, srate, feature)


# In[5]:


plt.imshow(train_data[2][0][0].astype('int32'))
plt.show()


# In[6]:


USE_GPU = True

dtype = torch.float32 # we will be using float throughout this tutorial

if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Constant to control how frequently we print train loss
print_every = 100

print('using device:', device)


# In[7]:


def check_accuracy(loader, model):
    if loader.dataset.train:
        logger.log('Checking accuracy on training set')
    elif loader.dataset.val:
        logger.log('Checking accuracy on validation set')
    else:
        logger.log('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        logger.log('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return acc


# In[10]:


def train(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                logger.writer.add_scalar("Loss/train", loss.item(), e*len(loader_train)+t)
                logger.log('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
        train_acc = check_accuracy(loader_train, model)
        logger.writer.add_scalar("Acc/train", train_acc, e)        
        val_acc = check_accuracy(loader_val, model)
        logger.writer.add_scalar("Acc/valid", val_acc, e)        
        logger.log()
        
        # Save model every 20 epochs
        if e > 0 and e % 10 == 0:
            logger.save_model(model,f"epoch{e}")
        elif val_acc >= 0.83:
            logger.save_model(model,f"valacc83-epoch{e}")
        elif val_acc >= 0.84:
            logger.save_model(model,f"valacc84-epoch{e}")
    # save final model
    logger.save_model(model,f"epoch{e}")
    return model


# In[11]:


def create_model():
    model = nn.Sequential()
    model.add_module('convolution', nn.Sequential(
        nn.Conv2d(1,100,3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        nn.Conv2d(100,100,3),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        nn.Conv2d(100,300,(2,3)),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Dropout(0.25),
        nn.Conv2d(300,300,(1,7),padding=1),
        nn.ReLU(),
        nn.MaxPool2d((1,2), stride=1),
        nn.Dropout(0.25),
        nn.Conv2d(300,100,(1,3),padding=1),
        nn.Conv2d(100,100,(1,3),padding=1),
    ))
    model.add_module('dense', nn.Sequential(
        nn.Flatten(),
        nn.Linear(1400,6144),
        nn.Linear(6144,2)
    ))
    return model


# In[12]:


model = create_model()
from pytorch_model_summary import summary
print(summary(model, torch.zeros((1, 1, 24, 72)), show_input=False))


# In[13]:


def test_model(model, test_data, subj_csv):
    # one-segment test
    logger.log('Testing model accuracy using 1-segment metric')
    loader_test = DataLoader(test_data, batch_size=70)
    per_sample_acc = check_accuracy(loader_test, model)

    # 40-segment test
    logger.log('Testing model accuracy using 40-segment per subject metric')
    with open(subj_csv, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        subjIDs = [row[0] for row in spamreader]
    unique_subjs,indices = np.unique(subjIDs,return_index=True)

    iterable_test_data = list(iter(DataLoader(test_data, batch_size=1)))
    num_correct = []
    for subj,idx in zip(unique_subjs,indices):
    #     print(f'Subj {subj} - gender {iterable_test_data[idx][1]}')
        data = iterable_test_data[idx:idx+40]
        #print(np.sum([y for _,y in data]))
        assert 40 == np.sum([y for _,y in data]) or 0 == np.sum([y for _,y in data])
        preds = []
        correct = 0
        with torch.no_grad():
            for x,y in data:
                x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
                correct = y
                scores = model(x)
                _, pred = scores.max(1)
                preds.append(pred)
        final_pred = (torch.mean(torch.FloatTensor(preds)) > 0.5).sum()
        num_correct.append((final_pred == correct).sum())
    #print(len(num_correct))
    acc = float(np.sum(num_correct)) / len(unique_subjs)
    logger.log('Got %d / %d correct (%.2f)' % (np.sum(num_correct), len(unique_subjs), 100 * acc))
    return per_sample_acc, acc


# In[22]:


def run_experiment(seed, model_name, feature, num_epoch):
    model = create_model()
    logger.set_model_save_location(f'{model_name}-{feature}')
    experiment = f'{model_name}-{feature}-seed{seed}'
    logger.set_experiment(experiment)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # toggle between learning rate and batch size values 

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.002, weight_decay=0.001)
    model = train(model, optimizer, epochs=num_epoch)
    
    # Testing
#     logger.log('Testing on balanced test set')
#     test_data_balanced = load_data(path, 'test', winLength, numChan, srate, feature,'v2')
#     sample_acc1, subject_acc1 = test_model(model, test_data_balanced, path + 'test_subjIDs.csv')

#     logger.log('Testing on all-male test set')
#     test_data_all_male = load_data(path, 'test', winLength, numChan, srate, feature,'v3')
#     sample_acc2, subject_acc2 = test_model(model, test_data_all_male, path + 'test_subjIDs_more_test.csv')
    
    return model


# In[24]:


logger = Logger()

batch_size = 70 # original
loader_train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(val_data, batch_size=batch_size)
for s in range(10):
    model = run_experiment(s, 'original', 'topo',70)


# In[ ]:





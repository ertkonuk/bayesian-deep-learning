# . . import libraries
import os
from pathlib import Path
# . . pytorch modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# . . numpy
import numpy as np
# . . scikit-learn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# . . matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as npimg
# . .  set this to be able to see the figure axis labels in a dark theme
from matplotlib import style
#style.use('dark_background')
# . . to see the available options
# print(plt.style.available)

from torchsummary import summary

# . . import libraries by tugrulkonuk
import utils
from utils import parse_args
from model import *
from trainer import Trainer
from callbacks import ReturnBestModel, EarlyStopping

# . . parse the command-line arguments
args = parse_args()

# . . set the device
if torch.cuda.is_available():  
    device = torch.device("cuda")  
else:  
    device = torch.device("cpu")      

# . . set the default precision
dtype = torch.float32

# . . use cudnn backend for performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# . . parameters
# . . user-defined
num_epochs    = args.epochs
batch_size    = args.batch_size
learning_rate = args.lr
train_size    = args.train_size
min_delta     = args.min_delta
patience      = args.patience 
num_workers   = args.num_workers
pin_memory    = args.pin_memory
jprint        = args.jprint
# . . computed
test_size     = 1.0 - train_size


# . . import the dataset
# . . transformer for data augmentation
transformer_train = torchvision.transforms.Compose([
  # torchvision.transforms.ColorJitter(
  #     brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
  transforms.RandomCrop(32, padding=4),
  torchvision.transforms.RandomHorizontalFlip(p=0.5),
  # torchvision.transforms.RandomRotation(degrees=15),
  torchvision.transforms.RandomAffine(0, translate=(0.1, 0.1)),
  # torchvision.transforms.RandomPerspective(),
  transforms.ToTensor(),                                            
])

# . . the train set
train_dataset = torchvision.datasets.CIFAR10(
    root='.',
    train=True,
    transform=transformer_train,
    download=True)

# . . the validation set: no augmentation!
valid_dataset = torchvision.datasets.CIFAR10(
    root='.',
    train=False,
    transform=transforms.ToTensor(),
    download=True)

# . . the number of classes in the data
num_classes = len(set(train_dataset.targets))
print('number of classes: ',num_classes)

# . . data loaders
# . . the training loader: shuffle
trainloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)

# . . the test loader: no shuffle
validloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False,
                         num_workers=num_workers, pin_memory=pin_memory)


# . . instantiate the model
model = CNNClassifier(num_classes)

# . . send model to device (GPU)
model.to(device)

# . . show a summary of the model
summary(model, (3, 32, 32))

# . . create the trainer
trainer = Trainer(model, device)

# . . compile the trainer
# . . define the loss
criterion = nn.CrossEntropyLoss()

# . . define the optimizer
optimparams = {'lr':learning_rate
              }

# . . define the callbacks
cb=[ReturnBestModel(), EarlyStopping(min_delta=min_delta, patience=patience)]

trainer.compile(optimizer='adam', criterion=criterion, callbacks=cb, jprint=jprint, **optimparams)

# . . the learning-rate scheduler
schedulerparams = {'factor':0.5,
                   'patience':50,
                   'threshold':1e-5,
                   'cooldown':5,
                   'min_lr':1e-5,                
                   'verbose':True               
                  }
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer, **schedulerparams)

# . . train the network
train_loss, valid_loss = trainer.fit(trainloader, validloader, scheduler=None, num_epochs=num_epochs)

plt.plot(train_loss)
plt.plot(valid_loss)
plt.legend(['train_loss', 'valid_loss'])
plt.show()

# . . model evaluation
# . . training dataset without augmentation
train_dataset_noaug = torchvision.datasets.CIFAR10(
                      root='.',
                      train=True,
                      transform=transforms.ToTensor(),
                      download=True)

# . . data loader for the training dataset without transforms
trainloader_noaug = torch.utils.data.DataLoader(
                     dataset=train_dataset_noaug, 
                     batch_size=batch_size, 
                     shuffle=False,
                     num_workers=num_workers,
                     pin_memory=pin_memory)

# . . evaluate the accuracy of the trained model
training_accuracy, test_accuracy = trainer.evaluate(trainloader_noaug, validloader)

#. . calculate and plot the confusion matrix
x_test = valid_dataset.data
y_test = np.array(valid_dataset.targets)
p_test = np.array([])

for inputs, targets in validloader:
    # . . move to device
    inputs, targets = inputs.to(device), targets.to(device)

    # . . forward pass
    outputs = trainer.model(inputs)

    # . . predictions
    _, predictions = torch.max(outputs, 1)

    # . . update the p-test
    p_test = np.concatenate((p_test, predictions.cpu().numpy()))

# . . the confusion matrix
cm = confusion_matrix(y_test, p_test)

# . . plot the confusion matrix 
utils.plot_confusion_matrix(cm, list(range(10)))

#
# . . save the model
torch.save(trainer.model.state_dict(), 'models/final_model.pt')

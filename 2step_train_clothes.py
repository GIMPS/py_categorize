from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import time
import os
import copy
from torch.utils.data.dataset import random_split
import myResnet as myres
from itertools import chain
from my_subset import random_split

# Load Data
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224,scale=(0.35,1.0)),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = '2step data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                  for x in ['clothes 2steps']}
dataset_len = len(image_datasets['clothes 2steps'])


class_names = image_datasets['clothes 2steps'].classes

image_datasets['val'], image_datasets['train'] = random_split(image_datasets['clothes 2steps'], [dataset_len // 5,
                                                                                        dataset_len - dataset_len // 5])
image_datasets['val'].transform=data_transforms['val']
image_datasets['train'].transform=data_transforms['train']

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

use_gpu = torch.cuda.is_available()

namefile=open('class_name_clothes.py','w')
namefile.write('class_names_clothes=[')
for name in class_names:
    namefile.write('\''+name+'\''+',')

namefile.write(']')
namefile.close()

# Training the model
all_losses=[]
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            all_losses.append(epoch_loss)
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'trained_nn_clothes')

                logfile = open('log.txt', 'w')
                logfile.write('Saved at '+time.ctime()+'\n')
                logfile.write('Best val Acc: {:4f}'.format(best_acc)+'\n')
                logfile.write('Training Acc: {:4f}'.format(epoch_acc)+'\n')
                logfile.close()

                f = open('trainLoss.py', 'w')
                f.write('loss=[')
                for loss in all_losses:
                        f.write(str(loss) + ',')
                f.write("]")
                f.close()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


######################################################################
# Finetuning the convnet
# ----------------------
#
# Load a pretrained model and reset final fully connected layer.
#
model_ft = models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
if use_gpu:
    model_ft = model_ft.cuda()

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.SGD(chain(model_ft.fc.parameters(),model_ft.layer4.parameters(),model_ft.layer3.parameters(),model_ft.avgpool.parameters()), lr=0.001, momentum=0.9)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)

#model_ft.load_state_dict(torch.load('trained_nn'))

######################################################################
# Train and evaluate


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50
                       )

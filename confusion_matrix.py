from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
from class_name import class_names
from mapping import *
output_csv = open('result.csv','w')
output_csv.write('id,category\n')


data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index),self.imgs[index]

data_dir = 'data_micro'
image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}
use_gpu = torch.cuda.is_available()

res={}



# Keep track of correct guesses in a confusion matrix
confusion = torch.zeros(len(name_mapping_list), len(name_mapping_list))

def test_model(model):
    model.eval()
    model.train(False)
    for i, data in enumerate(dataloaders['val']):
        (inputs, labels), (paths,_)= data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        # Go through a bunch of examples and record which are correctly guessed
        for j in range(inputs.size()[0]):
            if not labels.data[j] == preds[j]:
                print(paths[j])
                print()
            confusion[name_mapping[class_names[labels.data[j]]]][name_mapping[class_names[preds[j]]]]+= 1



trained_model = models.resnet18(pretrained=True)
num_ftrs = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_ftrs, len(class_names))
trained_model.load_state_dict(torch.load('trained_nn'))
if use_gpu:
    trained_model = trained_model.cuda()

test_model(trained_model)

# Normalize by dividing every row by its sum
for j in range(len(name_mapping)):
    if confusion[j].sum() == 0:
        continue
    confusion[j] = confusion[j] / confusion[j].sum()
f=open('conMat.py','w')
f.write('matrix=[')
for i in range(len(name_mapping)):
    f.write("[")
    for j in range(len(name_mapping)):
        f.write(str(confusion[i][j])+',')
    f.write("],")
f.write("]")
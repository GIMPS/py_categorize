from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import shutil
import operator
from class_name_general import class_names_general
from class_name_clothes import class_names_clothes
from class_name_shoes import class_names_shoes
from mapping import name_mapping
import myResnet as myres
import os
from shutil import copyfile
output_csv = open('result.csv','w')
output_csv.write('id,category\n')


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'clothes': transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'shoes': transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index),self.imgs[index]

data_dir = '2step data'

image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}



use_gpu = torch.cuda.is_available()

res={}

if os.path.exists(data_dir+'/clothes'):
    shutil.rmtree(data_dir+'/clothes')
if os.path.exists(data_dir+'/shoes'):
    shutil.rmtree(data_dir+'/shoes')


def test_model_general(model):
    model.train(False)
    for i, data in enumerate(dataloaders['test']):
        (inputs, labels), (paths,_)= data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            if class_names_general[preds[j]]=='clothes' or class_names_general[preds[j]] == 'shoes':
                srcdirname = paths[j]
                dstdirname = data_dir+'/'+class_names_general[preds[j]]+'/img'
                if not os.path.exists(dstdirname):
                    os.makedirs(dstdirname)
                copyfile(srcdirname, dstdirname + '/Test_' + paths[j].split('_')[1].split('.')[0]+'.jpg')

            else:
                res[int(paths[j].split('_')[1].split('.')[0])]= name_mapping[class_names_general[preds[j]]]

def test_model_clothes(model):
    model.train(False)
    for i, data in enumerate(dataloaders_clothes['clothes']):
        (inputs, labels), (paths,_)= data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            res[int(paths[j].split('_')[1].split('.')[0])]= name_mapping[class_names_clothes[preds[j]]]


def test_model_shoes(model):
    model.train(False)
    for i, data in enumerate(dataloaders_shoes['shoes']):
        (inputs, labels), (paths,_)= data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            res[int(paths[j].split('_')[1].split('.')[0])]= name_mapping[class_names_shoes[preds[j]]]

trained_model = myres.resnet34(pretrained=True)
num_ftrs = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_ftrs, len(class_names_general))###############!!
trained_model.load_state_dict(torch.load('trained_nn_general'))
if use_gpu:
    trained_model = trained_model.cuda()

test_model_general(trained_model)

###############################
image_datasets_clothes = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['clothes']}
dataloaders_clothes = {x: torch.utils.data.DataLoader(image_datasets_clothes [x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['clothes']}
dataset_sizes_clothes = {x: len(image_datasets_clothes [x]) for x in ['clothes']}


image_datasets_shoes = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['shoes']}
dataloaders_shoes = {x: torch.utils.data.DataLoader(image_datasets_shoes[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['shoes']}
dataset_sizes_shoes = {x: len(image_datasets_shoes[x]) for x in ['shoes']}
################################
trained_model.fc = nn.Linear(num_ftrs, len(class_names_clothes))###############!!
trained_model.load_state_dict(torch.load('trained_nn_clothes'))
if use_gpu:
    trained_model = trained_model.cuda()
test_model_clothes(trained_model)


trained_model.fc = nn.Linear(num_ftrs, len(class_names_shoes))###############!!
trained_model.load_state_dict(torch.load('trained_nn_shoes'))
if use_gpu:
    trained_model = trained_model.cuda()
test_model_shoes(trained_model)



sorted_res = sorted(res.items(), key=operator.itemgetter(0))

for d in sorted_res:
    output_csv.write(str(d[0])+','+str(d[1])+'\n')

output_csv.close()
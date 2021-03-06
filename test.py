from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import operator
from class_name import class_names
from mapping import name_mapping
import myResnet as myres

output_csv = open('result.csv','w')
output_csv.write('id,category\n')


data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(235),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index),self.imgs[index]

data_dir = 'data'
image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['test']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
use_gpu = torch.cuda.is_available()

res={}
def test_model(model):
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
            res[int(paths[j].split('_')[1].split('.')[0])]= name_mapping[class_names[preds[j]]]

trained_model = myres.resnet34(pretrained=True)
num_ftrs = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_ftrs, len(class_names))
state = torch.load('trained_state')
trained_model.load_state_dict(state['state_dict'])
if use_gpu:
    trained_model = trained_model.cuda()

test_model(trained_model)

sorted_res = sorted(res.items(), key=operator.itemgetter(0))

for d in sorted_res:
    output_csv.write(str(d[0])+','+str(d[1])+'\n')

output_csv.close()
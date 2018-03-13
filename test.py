from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import operator
from class_name import class_names
from mapping import name_mapping
output_csv = open('result.csv','w')
output_csv.write('id,category\n')

trained_model = models.resnet18(pretrained=True)
num_ftrs = trained_model.fc.in_features
trained_model.fc = nn.Linear(num_ftrs, 2)
trained_model.load_state_dict(torch.load('trained_nn'))

data_transforms = {
    'test': transforms.Compose([
        transforms.Resize(256),
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
    model.eval()

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


test_model(trained_model)

sorted_res = sorted(res.items(), key=operator.itemgetter(0))

for d in sorted_res:
    output_csv.write(str(d[0])+','+str(d[1])+'\n')

output_csv.close()
import os
from shutil import copyfile

class_names=os.listdir('Training Images')
for class_name in class_names:
    if class_name=='.DS_Store':
        continue
    imgs=os.listdir('Training Images/'+class_name)
    for img in imgs:
        if img=='.DS_Store':
            continue;
        num=int(img.split('_')[1].split('.')[0])
        if num %5 == 1 :
            srcdirname='Training Images/'+class_name+'/'+img
            dstdirname='val/'+class_name
            if not os.path.exists(dstdirname):
                os.makedirs(dstdirname)
            copyfile(srcdirname, dstdirname+'/'+img)
            
        else:
            srcdirname='Training Images/'+class_name+'/'+img
            dstdirname='train/'+class_name
            if not os.path.exists(dstdirname):
                os.makedirs(dstdirname)
            copyfile(srcdirname, dstdirname+'/'+img)

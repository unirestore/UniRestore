# # pre-process for ImageNet1k
import os, sys 
import shutil
import json
import glob

rt = r'./CUB_200_2011'
id2img_txt = os.path.join(rt, 'images.txt')
train_test_txt = os.path.join(rt, 'train_test_split.txt')
id2label_txt = os.path.join(rt, 'image_class_labels.txt')
image_rt = os.path.join(rt, 'images')
val_corruption_rt = os.path.join(rt, 'val_corruption')

id2img = {}
with open(id2img_txt) as fin:
    for line in fin:
        line = line.strip().split()
        id2img[line[0]] = line[1]
print("Total IDs:", len(id2img))

train_ids = [] # ids
test_ids = []  # ids
with open(train_test_txt) as fin:
    for line in fin:
        line = line.strip().split()
        if line[1] == '1':
            train_ids.append(line[0])
        else:
            test_ids.append(line[0])
print("Train/Test:", len(train_ids), len(test_ids))

id2label = {}
with open(id2label_txt) as fin:
    for line in fin:
        line = line.strip().split()
        id2label[line[0]] = int(line[1])-1
print("Total IDs:", len(id2label))

for ct, dset in enumerate([train_ids, test_ids]):
    data_list = []
    for data_id in dset:
        img_name = id2img[data_id]
        label = id2label[data_id]
        if ct == 0: # training set
            data_list.append([None, 
                              os.path.join(image_rt, img_name),
                              label])
        else:
            data_list.append([os.path.join(val_corruption_rt, img_name.split('/')[-1][:-4]+".png"), 
                              os.path.join(image_rt, img_name),
                              label])
    
    dset_name = {0:'train', 1:'val'}
    with open(os.path.join(rt,'%s.list'%(dset_name[ct])), 'w') as fp:
        for item in data_list:
            fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
    print(dset_name[ct], len(data_list))
# # pre-process for ImageNet1k
import os, sys 
import shutil
import json
import glob

train_meta = r'./ImageNet/meta/train.json'
val_meta = r'./ImageNet/meta/val_sub_2.json'

rt = r'./ImageNet/'
train_image_folder = r'./ImageNet/ILSVRC/Data/CLS-LOC/train'
val_degrada_folder = r'./ImageNet/valsub2_sev3_img'
val_clean_folder = r'./ImageNet/ILSVRC/Data/CLS-LOC/val'

# training data
train_list_file = []
with open(train_meta, 'r') as fn:
    train_meta_data = json.load(fn)
    for ct, (name, label) in enumerate(train_meta_data.items()):
        print(ct, len(train_meta_data), end='\r')
        degrad_img = None
        clean_img = glob.glob(os.path.join(train_image_folder, "%s.*"%(name)))[0]
        annotation = label
        train_list_file.append((degrad_img, clean_img, annotation))

with open(os.path.join(rt,'train.list'), 'w') as fp:
    for item in train_list_file:
        fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
print()

# val data
val_list_file = []
with open(val_meta, 'r') as fn:
    val_meta_data = json.load(fn)
    for ct, (name, label) in enumerate(val_meta_data.items()):
        print(ct, len(val_meta_data), end='\r')
        degrad_img = glob.glob(os.path.join(val_degrada_folder, "%s.*"%(name)))[0]
        clean_img = glob.glob(os.path.join(val_clean_folder, "%s.*"%(name)))[0]
        annotation = label
        val_list_file.append((degrad_img, clean_img, annotation))

with open(os.path.join(rt,'val.list'), 'w') as fp:
    for item in val_list_file:
        fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
print()

# # training full data
train_meta = r'./ImageNet/meta/train_all.json'
rt = r'./ImageNet/'
train_image_folder = r'./ImageNet/ILSVRC/Data/CLS-LOC/train'
train_list_file = []
with open(train_meta, 'r') as fn:
    train_meta_data = json.load(fn)
    for ct, (name, label) in enumerate(train_meta_data.items()):
        print(ct, len(train_meta_data), end='\r')
        degrad_img = None
        clean_img = glob.glob(os.path.join(train_image_folder, "%s.*"%(name)))[0]
        annotation = label
        train_list_file.append((degrad_img, clean_img, annotation))

with open(os.path.join(rt,'train_all.list'), 'w') as fp:
    for item in train_list_file:
        fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
print()
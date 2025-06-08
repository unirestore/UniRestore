# pre-process for DIV2k + Flickr2K + OST
import os, sys 
import shutil
from glob import glob

rt = r'.'
DIV2k_train_rt = os.path.join(rt, 'DIV2K_train_HR')
DIV2k_val_rt = os.path.join(rt, 'DIV2K_valid_HR')
DIV2k_val_lq_rt = os.path.join(rt, 'DIV2K_valid_HR_sev3')
Flickr2K_train_rt = os.path.join(rt, 'Flickr2K')
OST_train_rt = os.path.join(rt, 'OST')

# Training Set of DF2kOST
list_file = []
list_file += sorted(glob(os.path.join(DIV2k_train_rt, "**/*.*"), recursive=True))
list_file += sorted(glob(os.path.join(Flickr2K_train_rt, "**/*.*"), recursive=True))
list_file += sorted(glob(os.path.join(OST_train_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])

with open(os.path.join(rt,'train.list'), 'w') as fp:
    for item in list_file:
        fp.write('{} {} {}\n'.format(None, item, None))
print()

# Validation Set of DF2kOST
list_file = []
list_file += sorted(glob(os.path.join(DIV2k_val_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])

with open(os.path.join(rt,'val.list'), 'w') as fp:
    for item in list_file:
        fp.write('{} {} {}\n'.format(os.path.join(DIV2k_val_lq_rt, os.path.basename(item)), item, None))
print()

# pre-process for Cityscape
import os, sys 
import shutil
from glob import glob

# (degradation, clean, label)
rt = r'./Cityscapes'

# aachen_000000_000019_gtFine_labelIds.png 
# aachen_000000_000019_leftImg8bit.png
# training
city_train_rt = os.path.join(rt, 'leftImg8bit/train')
list_file = []
list_file += sorted(glob(os.path.join(city_train_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])
with open(os.path.join(rt,'train.list'), 'w') as fp:
    for item in list_file:
        degradation = None
        clean = item
        label = item.replace("leftImg8bit", "gtFine")[:-4] + "_labelIds.png"
        fp.write('{} {} {}\n'.format(degradation, clean, label))
print()

# validation
city_val_rt = os.path.join(rt, 'leftImg8bit/val')
list_file = []
list_file += sorted(glob(os.path.join(city_val_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])
with open(os.path.join(rt,'val.list'), 'w') as fp:
    for item in list_file:
        degradation = item.replace("leftImg8bit/val", "val_sev3")
        clean = item
        label = item.replace("leftImg8bit", "gtFine")[:-4] + "_labelIds.png"
        fp.write('{} {} {}\n'.format(degradation, clean, label))
print()
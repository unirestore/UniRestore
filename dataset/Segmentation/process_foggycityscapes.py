# pre-process for FoggyCityscape
import os, sys 
import shutil
from glob import glob

# (foggy1, foggy2, foggy3, clean, label)
rt = r'./FoggyCityscapes'
ref = r'./Cityscapes'

# training
city_train_rt = r'./FoggyCityscapes/leftImg8bit_foggyDBF/train'
city_clean_rt = r'./Cityscapes/leftImg8bit/train'
city_gt_rt = r'./Cityscapes/gtFine/train'
list_file = []
list_file += sorted(glob(os.path.join(city_clean_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])
with open(os.path.join(rt,'train.list'), 'w') as fp:
    for item in list_file:
        foggy1 = os.path.join(city_train_rt, "%s/%s_foggy_beta_0.01.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        foggy2 = os.path.join(city_train_rt, "%s/%s_foggy_beta_0.02.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        foggy3 = os.path.join(city_train_rt, "%s/%s_foggy_beta_0.005.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        clean = item
        label = item.replace("leftImg8bit", "gtFine")[:-4] + "_labelIds.png"
        fp.write('{} {} {} {} {}\n'.format(foggy1, foggy2, foggy3, clean, label))
print()

# validation
city_val_rt = r'./FoggyCityscapes/leftImg8bit_foggyDBF/val'
city_clean_rt = r'./Cityscapes/leftImg8bit/val'
city_gt_rt = r'./Cityscapes/gtFine/val'
list_file = []
list_file += sorted(glob(os.path.join(city_clean_rt, "**/*.*"), recursive=True))
print(len(list_file), list_file[0])
with open(os.path.join(rt,'val.list'), 'w') as fp:
    for item in list_file:
        foggy1 = os.path.join(city_val_rt, "%s/%s_foggy_beta_0.01.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        foggy2 = os.path.join(city_val_rt, "%s/%s_foggy_beta_0.02.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        foggy3 = os.path.join(city_val_rt, "%s/%s_foggy_beta_0.005.png"%(os.path.dirname(item).split('/')[-1], os.path.basename(item)[:-4]))
        clean = item
        label = item.replace("leftImg8bit", "gtFine")[:-4] + "_labelIds.png"
        fp.write('{} {} {} {} {}\n'.format(foggy1, foggy2, foggy3, clean, label))
print()
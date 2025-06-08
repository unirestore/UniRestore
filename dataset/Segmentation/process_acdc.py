# pre-process for Foggy_Zurich
import os, sys 
import shutil
from glob import glob

# (degradation, None, label)
rt = r'./ACDC'
image_rt = os.path.join(rt, 'rgb_anon')
gt_rt = os.path.join(rt, 'gt')

for de_type in ['fog', 'night', 'rain', 'snow']:
    image_folder_rt = os.path.join(image_rt, de_type)
    gt_folder_rt = os.path.join(gt_rt, de_type)
    for dset in ['train', 'val']:
        image_folder = os.path.join(image_folder_rt, dset)
        image_list = []
        image_list = sorted(glob(os.path.join(image_folder, "**/*.*"), recursive=True))
        print(de_type, dset, len(image_list))
        with open(os.path.join(rt,'%s_%s.list'%(dset, de_type)), 'w') as fp:
            for item in image_list:
                degradation = item
                clean = None
                label = os.path.join(os.path.dirname(item).replace("rgb_anon", "gt"),
                                     os.path.basename(item).replace("rgb_anon", "gt_labelIds"))
                fp.write('{} {} {}\n'.format(degradation, clean, label))
        print()



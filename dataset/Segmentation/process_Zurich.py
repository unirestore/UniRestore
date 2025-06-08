# pre-process for Foggy_Zurich
import os, sys 
import shutil
from glob import glob

# (foggy, None, label)
rt = r'./Foggy_Zurich'

meta_path = r'./Foggy_Zurich/lists_file_names/RGB_testv2_filenames.txt'
image_list = []
with open(meta_path) as fin:
    for line in fin:
        line = line.strip().split()
        image_list.append(line[0])

# RGB/GP010229/1508040913.0_frame_000955.png
# gt_labelIds/GP010229/1508040913.0_frame_000955.png
with open(os.path.join(rt,'val.list'), 'w') as fp:
    for item in image_list:
        degradation = os.path.join(rt, item)
        clean = None
        label = degradation.replace("RGB", "gt_labelIds")
        fp.write('{} {} {}\n'.format(degradation, clean, label))
print()



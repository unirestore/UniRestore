import os, sys 
import shutil, json
from xml.etree.ElementTree import Element, tostring
import xml.etree.cElementTree as ET
from tqdm import tqdm
dataset = {
    'CoCo': r'./COCO',
    # 'RTTS': r'./RTTS'
}

def indent_tree(elem, level=0):
    i = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_tree(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def dict_to_xml(tag, d):
    root = ET.Element(tag)
    for key, val in d.items():
        if "object" in key:
            object_ = ET.SubElement(root, "object")
            for key_, val_ in val.items():
                if key_ == "bndbox":
                    bbox = ET.SubElement(object_, key_)
                    for bdn, bdv in val_.items():
                        ET.SubElement(bbox, bdn).text = str(bdv)
                else:
                    ET.SubElement(object_, key_).text = str(val_)
        elif key == 'size':
            object_ = ET.SubElement(root, "size")
            for key_, val_ in val.items():
                ET.SubElement(object_, key_).text = str(val_)
        else: # information
            ET.SubElement(root, key).text = str(val)
    return root

def read_json_ann(json_pth, img_folder):
    """
    Target:
    "annotation":{
        "folder": img_folder, 
        "filename": img_name, 
        "license": 4,
        "coco_url": "http://images.cocodataset.org/val2017/000000397133.jpg",
        "date_captured": "2013-11-14 17:02:52",
        "flickr_url": "http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg",
        "id": 397133
        "size": {"width": w, "height": h, "depth": depth},
        "num_obj": n, 
        "segmented": 1, 
        "object1": {"name": car,
                   "supercategory": vehicle, 
                   "category_id": 18,
                   "difficult": 0,  # == iscrowd
                   "bbox": {"xmin": x, "ymin": y, "xmax": x, "ymax":y},
                   "ann_id": 1768},
        "object2" ...
    }
    """
    # load json
    with open(json_pth, 'r', encoding='utf-8') as f:
        db = json.load(f)

    category_dict = {} # id: {"supercategory": "person","id": 1,"name": "person"}
    for cat_info in db['categories']:
        cat_id = cat_info['id']
        category_dict[cat_id] = cat_info 

    database = {}
    # updating image informaiton
    for img_info in db['images']:
        image_id = img_info['id']
        database[image_id] = {"folder": img_folder,
                              "filename": img_info['file_name'], 
                              "license": img_info['license'],
                              "coco_url": img_info['coco_url'],
                              "date_captured": img_info['date_captured'],
                              "flickr_url": img_info['flickr_url'],
                              "id": img_info['id'],
                              "size": {"width": img_info['width'], "height": img_info['height'], "depth": 3},
                              "num_obj": 0,
                              "segmented": 0}

    # updating annotation information
    for ann_info in db['annotations']:
        image_id = ann_info['image_id']
        database[image_id]["object_%d"%database[image_id]["num_obj"]] = {
            "name": category_dict[ann_info["category_id"]]["name"],
            "supercategory": category_dict[ann_info["category_id"]]["supercategory"],
            "category_id": ann_info["category_id"],
            "difficult": ann_info["iscrowd"],
            "bndbox": {"xmin": round(ann_info["bbox"][0]), 
                       "ymin": round(ann_info["bbox"][1]), 
                       "xmax": round(ann_info["bbox"][0]+ann_info["bbox"][2]), 
                       "ymax": round(ann_info["bbox"][1]+ann_info["bbox"][3])},
            "ann_id": ann_info["id"]
        }
        database[image_id]["num_obj"] += 1
        if database[image_id]['segmented'] == 0 and ann_info["segmentation"]:
            database[image_id]['segmented'] = 1

    return database, category_dict

def json_to_xml_ann(json_path, img_folder, save_folder):
    save_ann_folder = os.path.join(save_folder, img_folder)
    os.makedirs(save_ann_folder, exist_ok=True)
    database, cat_dict = read_json_ann(json_path, img_folder)
    for img_id, img_ann in database.items():
        img_name = img_ann["filename"]
        root = dict_to_xml('annotation', img_ann)
        indent_tree(root, level=0)
        tree = ET.ElementTree(root)
        tree.write(os.path.join(save_ann_folder, "%s.xml"%img_name.split(".")[0]), encoding="utf-8")

def json_to_json_ann(json_path, img_folder, save_folder):
    save_ann_folder = os.path.join(save_folder, img_folder)
    os.makedirs(save_ann_folder, exist_ok=True)
    database, cat_dict = read_json_ann(json_path, img_folder)
    for img_id, img_ann in database.items():
        img_name = img_ann["filename"]
        with open(os.path.join(save_ann_folder, "%s.json"%img_name.split(".")[0]), 'w') as json_file:
            json.dump(img_ann, json_file, indent=4)

def process_CoCo_xml(rt):
    subset = {'train': 'train2017', 'val': 'val2017', 'test': 'test2017'}
    # process annotation to voc format
    save_folder = os.path.join(rt, "annotation_voc/")
    os.makedirs(save_folder, exist_ok=True)
    train_ann_pth = os.path.join(rt, 'annotations/instances_train2017.json')
    val_ann_pth = os.path.join(rt, 'annotations/instances_val2017.json')
    json_to_xml_ann(train_ann_pth, 'train2017', save_folder)
    json_to_xml_ann(val_ann_pth, 'val2017', save_folder)

    # create sample list
    for subname, dset in subset.items():
        print("\n", dset)
        img_folder = os.path.join(rt, dset)
        ann_folder = os.path.join(save_folder, dset)
        list_file = []
        for ct, sample in enumerate(os.listdir(img_folder)):
            print(ct, end='\r')
            input_file = os.path.join(img_folder, sample)
            if subname in ["val"]:
                crp_file = input_file.replace(dset, "%s_crp"%dset)
            else:
                crp_file = None
            if subname in ["train", "val"]:
                ann_file = os.path.join(ann_folder, "%s.xml"%sample.split(".")[0])
                # check label exists
                tree = ET.parse(ann_file)
                root = tree.getroot()
                label = []
                save = False
                for ct, dd in enumerate(root):
                    if dd.tag == 'object':
                        label.append(dd[0].text)
                        if dd[0].text in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'motorbike']:
                            save = True  
                if save:
                    list_file.append([crp_file, input_file, ann_file])            
            else:
                ann_file = None
                list_file.append([crp_file, input_file, ann_file])

        with open(os.path.join(rt, '%s.list'%subname), 'w') as fp:
            for item in list_file:
                fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
        print()

def process_CoCo_json(rt):
    subset = {'train': 'train2017', 'val': 'val2017', 'test': 'test2017'}
    # process annotation to voc format
    save_folder = os.path.join(rt, "annotation_voc/")
    os.makedirs(save_folder, exist_ok=True)
    train_ann_pth = os.path.join(rt, 'annotations/instances_train2017.json')
    val_ann_pth = os.path.join(rt, 'annotations/instances_val2017.json')
    json_to_json_ann(train_ann_pth, 'train2017', save_folder)
    json_to_json_ann(val_ann_pth, 'val2017', save_folder)

    # create sample list
    for subname, dset in subset.items():
        print("\n", dset)
        img_folder = os.path.join(rt, dset)
        ann_folder = os.path.join(save_folder, dset)
        list_file = []
        for ct, sample in enumerate(os.listdir(img_folder)):
            print(ct, end='\r')
            input_file = os.path.join(img_folder, sample)
            if subname in ["val"]:
                crp_file = input_file.replace(dset, "%s_crp"%dset)
            else:
                crp_file = None
            if subname in ["train", "val"]:
                ann_file = os.path.join(ann_folder, "%s.json"%sample.split(".")[0])
                # check label exists
                with open(ann_file, 'r') as json_file:
                    data_dict = json.load(json_file)
                label = []
                save = False
                for k, v in data_dict.items():
                    if 'object' in k:
                        label.append(v['name'])
                        if v['name'] in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'motorbike']:
                            save = True  
                if save:
                    list_file.append([crp_file, input_file, ann_file])            
            else:
                ann_file = None
                list_file.append([crp_file, input_file, ann_file])

        with open(os.path.join(rt, '%s.list'%subname), 'w') as fp:
            for item in list_file:
                fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
        print()


# RTTS
def rtts_xml2dict(json_path):
    print(json_path)
    # Read Annotation
    tree = ET.parse(json_path)
    root = tree.getroot()
    database = {"num_obj":0}
    for ct, dd in enumerate(root):
        if dd.tag == 'object':
            object_info = {}
            for data in dd:
                if data.tag == 'bndbox':
                    object_info[data.tag] = {
                        "xmin": int(data[0].text),
                        "ymin": int(data[1].text),
                        "xmax": int(data[2].text),
                        "ymax": int(data[3].text)
                    }
                elif data.tag in ['truncated', "difficult", "inferred"]:
                    object_info[data.tag] = int(data.text)
                else:
                    object_info[data.tag] = data.text

            database["object_%d"%database['num_obj']] = object_info
            database["num_obj"] += 1
        elif dd.tag == 'size':
            database['size'] = {
                'width': int(dd[0].text),
                'height': int(dd[1].text),
                'depth': int(dd[2].text)
            }
        elif dd.tag in ['segmented']:
            database[dd.tag] = int(dd.text)
        else:
            database[dd.tag] = dd.text
    return database

def dict2json(json_path, save_folder):
    save_ann_folder = save_folder
    os.makedirs(save_ann_folder, exist_ok=True)
    database = rtts_xml2dict(json_path)
    img_name = database['filename']
    with open(os.path.join(save_ann_folder, "%s.json"%img_name.split(".")[0]), 'w') as json_file:
        json.dump(database, json_file, indent=4)
    return database

def process_RTTS(rt):
    img_folder = os.path.join(rt, 'JPEGImages')
    ann_folder = os.path.join(rt, 'Annotations')
    list_file = []        
    for ct, sample in enumerate(os.listdir(img_folder)):
        print(ct, end='\r')
        input_file = os.path.join(img_folder, sample)
        ann_file = os.path.join(ann_folder, sample.replace('.png', '.xml'))
        if os.path.isfile(ann_file):
            list_file.append((input_file, None, ann_file))
        else:
            print("Not annotation file: ", ann_file)
    
    with open(os.path.join(rt, 'test.list'), 'w') as fp:
        for item in list_file:
            fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
    print()

def process_RTTS_json(rt):
    img_folder = os.path.join(rt, 'JPEGImages')
    ann_folder = os.path.join(rt, 'Annotations')
    new_ann_folder = os.path.join(rt, 'Annotations_json')
    list_file = []        
    for ct, sample in enumerate(os.listdir(img_folder)):
        print(ct, end='\r')
        input_file = os.path.join(img_folder, sample)
        ann_file = os.path.join(ann_folder, sample.replace('.png', '.xml'))
        if os.path.isfile(ann_file):
            database = dict2json(ann_file, new_ann_folder)
            ann_file = os.path.join(new_ann_folder, sample.replace('.png', '.json'))
            if os.path.isfile(ann_file):
              list_file.append((input_file, None, ann_file))
        else:
            print("Not annotation file: ", ann_file)
    
    with open(os.path.join(rt, 'test.list'), 'w') as fp:
        for item in list_file:
            fp.write('{} {} {}\n'.format(item[0], item[1], item[2]))
    print()


if __name__ == "__main__":
    for dset, rt in dataset.items():
        if dset == 'RTTS':
            process_RTTS_json(rt)
        elif dset == 'CoCo':
            process_CoCo_json(rt)
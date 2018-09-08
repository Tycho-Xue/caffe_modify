import random
import os
dataset_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
with open(dataset_addr, 'r') as f:
    input = f.readlines()

data = [x.strip() for x in input]
num_pick = 700
img_pick = random.sample(data, num_pick)
cwd = os.getcwd()
VOCdevkit_addr = os.path.dirname(cwd)
YOLO_dir = os.path.dirname(cwd)
VOC_dir = os.path.join(YOLO_dir, 'VOCdevkit', 'VOC2012')
img_dir = os.path.join(VOC_dir, 'JPEGImages')
anno_dir = os.path.join(VOC_dir, 'Annotations')

imglist_addr = os.path.join(YOLO_dir, 'dataset', 'imglist.txt')

with open(imglist_addr) as f:
    set = [for id in img_pick]
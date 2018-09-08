# prepare the dataset
# read the voc dataset
import random
import os
dataset_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
calibration_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/darkNet_calibration_list.txt'
validation_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/darkNet_validation_list.txt'
test_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/darkNet_test_list.txt'
with open(dataset_addr,'r') as f:
    input = f.readlines()
# print(input)
data = [x.strip() for x in input]
# print("length of the whole dataset is {}".format(len(data)))
num_pick = 200
img_calibration = random.sample(data, num_pick)
# print(random_img)
img_test = random.sample(data, num_pick)
img_validation = random.sample(data, num_pick)
img_set = [img_calibration, img_test, img_validation]
img_addr = [calibration_addr, test_addr, validation_addr]
# make the necessary paths
cwd = os.getcwd()
YOLO_dir = os.path.dirname(cwd)
VOC_dir = os.path.join(YOLO_dir, 'VOCdevkit', 'VOC2012')
img_dir = os.path.join(VOC_dir, 'JPEGImages')
anno_dir = os.path.join(VOC_dir, 'Annotations')

for addr, set in zip(img_addr, img_set):
    with open(addr, 'w') as f:
        set = [os.path.abspath(os.path.join(img_dir, id))+'.jpg' + ' ' + os.path.abspath(os.path.join(anno_dir, id) + '.xml') for id in set]
        set.insert(0, 'softmax:softmax')
        set.insert(0, '{}'.format(len(set)-1))
        output = set
        # output = [id+'\n' for id in img_calibration]
        output = '\n'.join(output)
        f.write(output)



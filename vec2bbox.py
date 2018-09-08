# convert the 21125 dimensions vector to the actual bounding box

import sys
import numpy as np
from math import *
import os
import cv2
import matplotlib


num_total = 13*13*5
num_h = 13
num_w = 13
num_perGrid = 5
classes = 20
coords_num = 4
biasData = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
cell_size = 25
thresh = 0.5
nmsThreshold = 0.4
top_k = -1

# read the blob data and shape
def blob_file_reader(inputfile):
    with open(inputfile) as f: ## Be careful while reading the file if binary
        line = f.readline()
        split_list = line.strip().split(" ")
        #print(split_list)
        dequant_scale = float(split_list[0])
        #print(dequant_scale) 
        blob_shape =tuple(map(int,split_list[1:4][::-1]))
        blob_shape = (21125,1,1)
        #print(line,blob_shape)
        if (len(split_list)< 4) and (len(split_list) >= 6):
            print("Blob file not valid")
            exit(0)
        line2 = f.readline().strip().split(" ")
        line2 = list(map(float, line2))
        #print(len(line2))
    flt_out = np.divide(np.asarray(line2), dequant_scale)
    #print("flt_out",flt_out)
    return flt_out.reshape(blob_shape)

def recOverlap(a, b):
    min_x = max(a[2], b[2])
    min_y = max(a[3], b[3])
    max_x = min(a[2]+a[0], b[2]+b[0])
    max_y = min(a[3]+a[1], b[3]+b[1])
    intersection = max(max_x-min_x, 0) * max(max_y-min_y, 0)
    union = a[0]*a[1] + b[0]*b[1] - intersection
    return intersection/union

def takeFirst(elem):
    return elem[0]

def GetMaxScoreIndex(scores, threshold, top_k, score_index_vec):
    for i in range(len(scores)):
        if scores[i] > threshold:
            score_index_vec.append((scores[i], i))

    score_index_vec.sort(key = takeFirst, reverse = True)

    if (top_k > 0 and top_k < len(score_index_vec)):
        score_index_vec = score_index_vec[:top_k]

def NMSFast(bboxes, scores, score_threshold, nms_threshold, eta, top_k, indices, computeOverlap):
    # Get top_k scores (with corresponding indices)
    score_index_vec = []
    GetMaxScoreIndex(scores, score_threshold, top_k, score_index_vec)
    # Do nms
    adaptive_threshold = nms_threshold
    for i in range(len(score_index_vec)):
        idx = score_index_vec[i][1]
        keep = True
        for k in range(len(indices)):
            kept_index = indices[k]
            overlap = computeOverlap(bboxes[idx], bboxes[kept_index])
            if (overlap >= adaptive_threshold):
                keep = False       ################  question ?
        if (keep):
            indices.append(idx)
        if (keep and eta < 1 and adaptive_threshold > 0.5):
            adaptive_threshold *= eta

def do_nms_sort(detections, total, score_thresh, nms_thresh):
    boxes = []
    for i in range(total):
        box_index = i * (classes + coords_num + 1)
        width = detections[box_index + 2]
        height = detections[box_index + 3]
        x = detections[box_index + 0] - width / 2
        y = detections[box_index + 1] - height / 2
        boxes.append([width, height, x, y])
    for k in range(classes):
        indices = []
        scores = []  # scores is the class score for all the boxes
        for i in range(total):
            box_index = i * (classes + coords_num + 1) # coords index in the raw vector for each bbox
            class_index = box_index + 5 # start of the class index in the raw vector
            scores.append(detections[class_index + k]) # score for the specific class for that bbox
            detections[class_index + k] = 0 # clear the socre for that class?

        NMSFast(boxes, scores, score_thresh, nms_thresh, 1, 0, indices, recOverlap)

        for i in range(len(indices)):
            box_index = indices[i] * (classes + coords_num + 1)
            class_index = box_index + 5
            detections[class_index + k] = scores[indices[i]]
    return detections

def logistic_activate(x):
    return 1./(1.+exp(-x))
def takeSecond(elem):
    return elem[1]

id = '2008_001643'
img_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/VOCdevkit/VOC2012/JPEGImages/'+id+'.jpg'
img = cv2.imread(img_addr)
img_dims = img.shape[:2]
print('img_dims = {}'.format(img_dims))
# reading the vector file

float_result = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/temp/xtensa/'+id+'_reg_reshape_regression.blob'
fix_result = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/temp/QP_xtensa_3/'+id+'_reg_reshape_regression.blob'
paths_results = [float_result, fix_result]
for indicator, path_result in enumerate(paths_results):
    bbox_raw = []
    dst_data = []
    indices = []
    srcData = blob_file_reader(path_result)
    srcData = srcData[:, 0, 0]
    print("vec_raw shape: {}".format(srcData.shape))
    dstData = srcData
    for x in range(num_h):
        for y in range(num_w):
            for a in range(num_perGrid):
                index = (y*num_w+x)*num_perGrid + a  # the index for the bbox
                p_index = index*cell_size + 4  # index for the confidence score
                box_index = index*cell_size  # start with the coords
                dstData[box_index + 0] = (x + logistic_activate(srcData[box_index + 0])) / num_w
                dstData[box_index + 1] = (y + logistic_activate(srcData[box_index + 1])) / num_h
                dstData[box_index + 2] = exp(srcData[box_index + 2]) * biasData[2 * a] / num_w
                dstData[box_index + 3] = exp(srcData[box_index + 3]) * biasData[2 * a + 1] / num_h
                class_index = index*cell_size + 5  # start point of the class possibility

                for m in range(classes):
                    prob = dstData[class_index + m]
                    if prob > thresh:
                        dstData[class_index + m] = prob
                    else:
                        dstData[class_index + m] = 0

    if nmsThreshold > 0:
        detections = do_nms_sort(dstData, num_h*num_w*num_perGrid, thresh, nmsThreshold)
    detections = detections.reshape((num_h, num_w, num_perGrid*cell_size))
# print(detections)
# parse the raw vector into bounding boxes
    bbox = []

    for x in range(num_w):
        for y in range(num_h):
            for d in range(num_perGrid):
                vec = detections[x, y, d*cell_size:(d+1)*cell_size]
                coords = vec[0:4]
                coords[0], coords[1], coords[2], coords[3] = coords[0]-0.5*coords[2], coords[1] - 0.5*coords[3], coords[0]+0.5*coords[2], coords[1] + 0.5*coords[3]
                coords[0], coords[1], coords[2], coords[3] = coords[0]*img_dims[0], coords[1]*img_dims[1], coords[2]*img_dims[0], coords[3]*img_dims[1]
                prob_score = vec[4]
                prob_classes = vec[5:]
                #if max(prob_classes) <= 0.2:
                    #continue
                label = np.argmax(prob_classes)
                bbox.append([list(coords), prob_score, label, prob_classes[label]])
                bbox.sort(key=takeSecond, reverse=True)



    bbox = bbox[:2]
    for box in bbox:
        print(box)
        rec = box[0]
        if indicator == 1:
            continue
        cv2.rectangle(img,(int(rec[0]), int(rec[1])), (int(rec[2]), int(rec[3])), (0,0,indicator*255), 2)

# read the ground truth
gnd_addr = 'C:/Users/lxue/Desktop/cadence_training/XNNC/Example/YOLO_v2/dataset_convert/results/'+id+'.txt'
gnd_bboxes = []
with open(gnd_addr, 'r') as f:
    line = f.readline()
    while(line):
        split_list = line.strip().split(' ')
        coords = split_list[:4]
        coords = [int(coord) for coord in coords]
        label = int(split_list[4])
        gnd_bboxes.append([coords, label])
        cv2.rectangle(img, (coords[0], coords[1]), (coords[2], coords[3]), (255,255,255), 2)
        line = f.readline()

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# draw bounding boxes

# save the detection results to files

# current_path = os.getcwd()
# base_path = os.path.dirname(current_path)
# res_path = os.path.join(base_path, 'detection_results')
# with open(os.path.join(res_path, 'temp.txt'), 'w') as f:
#     output = [str(data) for data in list(detections)]
#     f.write(' '.join(output))


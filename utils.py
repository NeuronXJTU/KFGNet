import os
import time
import random
import numpy as np

def check_dir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)

def tic():
    return time.time()

def toc(start):
    stop = time.time()
    print('\nUsed {:.2f} s\n'.format(stop - start))
    return stop - start

def compute_iou(rec1, rec2):
    # xmin, ymin, xmax, ymax = (coor[0], coor[1], coor[2], coor[3])
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
 
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
 
    # find the each edge of intersect rectangle
    left_line = max(rec1[0], rec2[0])
    right_line = min(rec1[2], rec2[2])
    top_line = min(rec1[3], rec2[3])
    bottom_line = max(rec1[1], rec2[1])
 
    # judge if there is an intersect
    if left_line >= right_line or top_line <= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (top_line - bottom_line)
        return (intersect / (sum_area - intersect)) * 1.0


def similarity(v_data, crucial_frame_num, sfeature_weight=1, siou_weight=1, sframe_weight=1):
    crucial_frame = v_data[crucial_frame_num,:]

    distance = np.sqrt(np.sum((crucial_frame[6:] - v_data[:,6:]) ** 2, axis=1))
    max_distance = np.max(distance)
    sfeature = np.exp(- distance / (max_distance + 1e-8))
    
    iou_list = []
    for each_rec in v_data[:,1:5]:
        iou = compute_iou(crucial_frame[1:5], each_rec)
        iou_list.append(iou)
    siou = np.array(iou_list)

    frame_distance = abs(v_data[:,0] - crucial_frame[0])
    max_frame_distance = np.max(frame_distance)
    sframe = 1 - frame_distance / (max_frame_distance + 1e-8)

    sim = (sfeature * sfeature_weight + siou * siou_weight + sframe * sframe_weight) / 3
    
    return sim 
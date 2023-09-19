from torch.utils.data import Dataset
import glob
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import *
import torch


TOTAL_FRAME_NUM = 300


class VideoDataset(Dataset):
    def __init__(self, data_dir):
        self.videos = glob.glob(data_dir + '/*.json')
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, index):
        v_data = []

        video_data = self.videos[index]
        with open(video_data, encoding="utf-8") as f:
            json_data = json.load(f)
        crucial_frame = json_data.get('frame_list')[1]
        det_result = json_data.get('det_result')
        
        best_diff = 1000
        crucial_frame_index = 0
        for idx, frame in enumerate(det_result):
            frame_data = []
            frame_num = frame.get('frame_num')
            
            diff = abs(crucial_frame - frame_num)

            if diff < best_diff:
                best_diff = diff
                crucial_frame_index = idx
            
            frame_data.append(frame_num)
            bbox = frame.get('boxes')[0]
            frame_data += bbox
            score = frame.get('scores')[0]
            frame_data.append(score)
            feature = frame.get('features')[0]
            frame_data += feature
            
            v_data.append(np.array(frame_data))
        
        v_data = np.array(v_data).astype(np.float32)

        
        if len(v_data) >= TOTAL_FRAME_NUM:
            v_len = len(v_data)
            idxs = np.linspace(0, v_len-1, TOTAL_FRAME_NUM-1, dtype=int).tolist()
            idxs.append(crucial_frame_index)
            idxs.sort()
            new_idx = idxs.index(crucial_frame_index)
            
            v_data = v_data[idxs, :]
            sim = similarity(v_data, new_idx)
        else:
            sim = similarity(v_data, crucial_frame_index)
        label = sim.astype(np.float32)

        scaler = MinMaxScaler()
        v_data[:, :6] = scaler.fit_transform(v_data[:, :6])
        
        return torch.Tensor(v_data), torch.Tensor(label)

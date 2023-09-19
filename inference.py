import glob
import torch
import json
import numpy as np
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from utils import *
from model import scoreNet


def read_video(video_data):
    v_data = []
    frame_num_list = []

    with open(video_data, encoding="utf-8") as f:
        json_data = json.load(f)
    video_name = json_data.get('video_name')
    crucial_frame = json_data.get('frame_list')[1]
    det_result = json_data.get('det_result')
    
    for idx, frame in enumerate(det_result):
        frame_data = []
        frame_num = frame.get('frame_num')
        frame_num_list.append(frame_num)
        
        frame_data.append(frame_num)
        bbox = frame.get('boxes')[0]
        frame_data += bbox
        score = frame.get('scores')[0]
        frame_data.append(score)
        feature = frame.get('features')[0]
        frame_data += feature
        
        v_data.append(np.array(frame_data))
    
    v_data = np.array(v_data).astype(np.float32)

    scaler = MinMaxScaler()
    v_data[:, :6] = scaler.fit_transform(v_data[:, :6])
    
    return torch.Tensor(v_data).unsqueeze(0), frame_num_list, video_name, crucial_frame



device = torch.device('cuda:0')

model = scoreNet()
state_dict = torch.load('./OutTrain/RNN-fold0-best.ckpt')
# create new OrderedDict that does not contain `module.`
# new_state_dict = OrderedDict()
# for k, v in state_dict.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# load params
model.load_state_dict(state_dict)

model = model.to(device)

model.eval()

fwriter_result = open('./result.csv', 'w')


with torch.no_grad():
    for i, videos in enumerate(glob.glob('../data_for_miccai/test/*.json')):
        
        video, frame_num_list, video_name, crucial_frame = read_video(videos)
        print(video_name)
        print('crucial_frame:' + str(crucial_frame))

        video = video.to(device)

        output = model(video)

        pred_idx = F.softmax(output, dim=1).data.max(1)[1]
        
        pred_frame = frame_num_list[pred_idx]

        print('pred_frame:' + str(pred_frame))
        
        fwriter_result.write('{},{},{}\n'.format(video_name, crucial_frame, pred_frame))
        fwriter_result.flush()

fwriter_result.close()

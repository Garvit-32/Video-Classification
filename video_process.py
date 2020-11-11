import glob
import os
import time
from tqdm import tqdm
import datetime
import cv2
import numpy as np

try:
    import av
except ModuleNotFoundError:
    os.system('!pip install av')
    import av

# data_path = 'data'
# res_path = 'res'

# seq_length = 16


def extract_frame(video_path):
    frames = []
    video = av.open(video_path)
    for frame in video.decode(0):
        yield frame.to_image()


def create_frame(data, res):
    for i in tqdm(os.listdir(data)):
        p1 = os.path.join(data, i)
        r1 = os.path.join(res, i)
        os.makedirs(r1, exist_ok=True)
        print(f'Extract frame from {p1}')
        for j in tqdm(os.listdir(p1)):
            vid_path = os.path.join(p1, j)
            r2 = os.path.join(r1, j[:-11])
            os.makedirs(r2, exist_ok=True)
            for j, frame in enumerate(extract_frame(vid_path)):
                frame.save(os.path.join(r2, f"{j}.jpg"))


# create_frame(data_path, res_path)


def preprocess_data(seq_length=16, data, frame_path, res):
    create_frame(data, frame_path)

    dir = os.listdir(frame_path)
    for i in dir:
        p1 = os.path.join(frame_path, i)
        r1 = os.path.join(res, i)
        os.makedirs(r1, exist_ok=True)
        print(f'Extract required number of frame from {p1} directory')
        for j in tqdm(os.listdir(p1)):
            p2 = os.path.join(p1, j)
            r2 = os.path.join(r1, j)
            l = 0
            skip_length = int(len(os.listdir(p2))/seq_length)
            k = 0
            while(l != seq_length):
                p3 = os.path.join(p2, str(k)+'.jpg')
                try:
                    img = cv2.imread(p3)
                    img = cv2.resize(img, (128, 128))
                    cv2.imwrite(r2 + str(k) + '.jpg', img)
                except:
                    print(p3)
                # if (k == 0):
                #     img1 = img
                # else:
                #     img1 = np.append(img1, img, axis=1)
                k = k + skip_length
                l += 1


# data = 'data'
# frame_path = 'frame'
# res = 'dataset'
# seq_length = 16
# preprocess_data(seq_length, data, frame_path, res)

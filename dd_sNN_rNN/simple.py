import dataPrep
import os
import time
import sys
import threading
import numpy as np
import requests
import json
import RawImage
import cv2
from utils import decode_netout
'''
SPM_URLs = ['http://c220g5-110504.wisc.cloudlab.us:8502/v1/models/spm',
            'http://c220g5-110530.wisc.cloudlab.us:8502/v1/models/spm'  ]
'''

SPM_URLs = ['http://c220g5-110504.wisc.cloudlab.us:8502/v1/models/spm']
'''
REFM_URLs = ['http://localhost:8504/v1/models/refm_cpu']
'''
REFM_URLs =['http://c220g5-110530.wisc.cloudlab.us:8504/v1/models/refm_cpu']

def get_features(feature_fn, frames):
    return np.array([feature_fn(frame) for frame in frames])

def get_distances(dist_fn, features, delay):
    return np.array([dist_fn(features[i], features[i-delay]) for i in
        range(delay, len(features))])

def sendSNNRequest(frame):
   # SERVER_URL = 'http://c220g5-110504.wisc.cloudlab.us:8502/v1/models/spm'
    SERVER_URL = SPM_URLs[0] 
   # SERVER_URL = 'http://localhost:8502/v1/models/spm'
    payload = {'signature_name': 'serving_default',
               'inputs':
                   {
                       'image': [ frame.tolist() ]
                   },
              }
    start = time.time()
    r = requests.post(SERVER_URL + ':predict', json=payload)
    snn_time = time.time() - start
    probs = r.json()['outputs']
    return probs[0][1], snn_time

def sendRNNRequest(frame, target):
    f = open('yolo.meta','r')
    metadata = json.loads(f.read())
    SERVER_URL = REFM_URLs[0]

    payload = {'signature_name': 'serving_default',
               'inputs': 
                  {   
                      'images': [ frame.tolist() ]
                  }   
              }   
    start = time.time()
    r = requests.post(SERVER_URL + ':predict', json = payload)
    rnn_time = time.time() - start

    result = r.json()['outputs'][0] #(19,19,425) = (19,19,5*(5+80))
    result = np.array(result)
    result = result.reshape((19,19,5,-1))

    anchors = metadata['anchors']
    nb_class = metadata['classes']

    boxes = decode_netout(result, anchors, nb_class)
    if boxes:
        if metadata['labels'][boxes[0].label] == target:
            return 1.0, rnn_time
        else:
            return 0.0, rnn_time
    else:
        return 0.0, rnn_time

def runNNRequest(sNNFrameCount, rNNFrameCount, sNNTotalTime, rNNTotalTime, smallFrame, origFrame, c_low, c_high, target):
    sNNFrameCount = sNNFrameCount + 1 
    conf, snn_time = sendSNNRequest(smallFrame)
    
    if conf >= c_low and conf < c_high:
        start = time.time()
        rNNFrameCount = rNNFrameCount + 1 
        frame = cv2.resize(origFrame, (608, 608))
        conf, rnn_time = sendRNNRequest(frame, target)
    return sNNFrameCount, rNNFrameCount, snn_time, rnn_time, conf

def main():
    video_fname = "/data/dataset/jackson-town-square.mp4"
    csv_in = "/data/dataset/jackson-town-square.csv"
    avg_fname = "/data/dataset/jackson-town-square.npy"
    avg_fname_orig = "/data/dataset/jackson-town-square_400_600.npy"
    num_frames = 30 
    start_frame = 3 
    target = 'car'
    
    # Param: difference detector
    dist_metric = 'mse'
    thresh = 0.00024391713548410724 #0.002  #0.00024391713548410724
    frame_delay = 1 
    feature_fn = RawImage.compute_feature
    get_distance_fn = RawImage.get_distance_fn

    # Param: specialized NN
    c_low = 0.10
    c_high = 0.90

    # Param: time
    ddTotalTime = 0
    sNNTotalTime = 0
    rNNTotalTime = 0

    start = time.time()
    data, nb_classes = dataPrep.get_data_for_test(
            csv_in, video_fname, avg_fname,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=[target],
            resol=(50, 50), shiftEn=1)
    X, Y = data
    print(X[0].shape)

    origData, orig_nb_classes = dataPrep.get_data_for_test(
            csv_in, video_fname, avg_fname_orig,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=[target],
            resol=(400, 600), shiftEn=0)
    origX, origY = origData
    print(origX[0].shape)


#    start = time.time()
    ddTotalTime = ddTotalTime + (time.time() - start)

    conf = None
    ddCount = 0
    sNNFrameCount = 0
    rNNFrameCount = 0

    for i in range(1):
        if i == 0:
            sNNFrameCount, rNNFrameCount, sNNTotalTime, rNNTotalTime, conf = runNNRequest(
                sNNFrameCount, rNNFrameCount, sNNTotalTime, rNNTotalTime, X[i], origX[i], c_low, c_high, target)

        print('Frame %d: Confidence = %f'%(i, conf))

    print("0. Total frames: {} from frame {} to frame {}".format(len(X), start_frame, start_frame+num_frames-1))
    print("1. Difference detector: {} times, {:.2f} sec".format(ddCount, ddTotalTime))
    print("2. Specialized NN: {} times, {:.2f} sec".format(sNNFrameCount, sNNTotalTime))
    print("3. Reference NN: {} times, {:.2f} sec".format(rNNFrameCount, rNNTotalTime))


if __name__ == '__main__':
    main()


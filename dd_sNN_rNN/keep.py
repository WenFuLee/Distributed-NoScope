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
import asyncio
import aiohttp
import random
#from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

SPM_URLs = ['http://c220g5-110504.wisc.cloudlab.us:8502/v1/models/spm',
            'http://c220g5-110504.wisc.cloudlab.us:8503/v1/models/spm_2'  ]
'''
REFM_URLs = ['http://localhost:8504/v1/models/refm',
             'http://c220g5-110530.wisc.cloudlab.us:8503/v1/models/refm_cpu',
             'http://c220g5-110504.wisc.cloudlab.us:8503/v1/models/refm_cpu' ]
'''
REFM_URLs = [ 'http://c220g5-110530.wisc.cloudlab.us:8504/v1/models/refm_cpu',
              'http://c220g5-111323.wisc.cloudlab.us:8504/v1/models/refm_cpu',
              'http://c220g5-111327.wisc.cloudlab.us:8504/v1/models/refm_cpu',
              'http://c220g5-111312.wisc.cloudlab.us:8504/v1/models/refm_cpu',
              'http://c220g5-110530.wisc.cloudlab.us:8505/v1/models/refm_cpu_2',
              'http://c220g5-111323.wisc.cloudlab.us:8505/v1/models/refm_cpu_2',
              'http://c220g5-111327.wisc.cloudlab.us:8505/v1/models/refm_cpu_2',
              'http://c220g5-111312.wisc.cloudlab.us:8505/v1/models/refm_cpu_2'
           ]

num_of_RNN = 4 
num_of_SNN = 2

def get_features(feature_fn, frames):
    return np.array([feature_fn(frame) for frame in frames])

def get_distances(dist_fn, features, delay):
    return np.array([dist_fn(features[i], features[i-delay]) for i in
        range(delay, len(features))])


async def sendSNNRequest(session, frames):
    i = random.randint(0,len(SPM_URLs)-1)
    SERVER_URL = SPM_URLs[i]
    payload = {'signature_name': 'serving_default',
               'inputs':
                   {
                       'image': frames.tolist()
                   },
              }
#    print("Sending SNN Request to server:", i)
    async with session.post(SERVER_URL + ':predict', json=payload) as r:
        r_json = await r.json()
#        print("Recievied SNN Response!")
    probs = r_json['outputs']
    return  probs[0][1]

async def sendRNNRequest(metadata, session, frames, target):
    i = random.randint(0, num_of_RNN-1)
    SERVER_URL = REFM_URLs[i]
    payload = {'signature_name': 'serving_default',
               'inputs': 
                  {   
                      'images': frames.tolist()
                  }   
              }
#    print("Sending RefNN Request to server:", i)
    async with session.post(SERVER_URL+ ':predict', json=payload) as r:
        r_json = await r.json()
        print("Recievied RefNN Response!")

    result = r_json['outputs'][0] #(19,19,425) = (19,19,5*(5+80))
    result = np.array(result)
    result = result.reshape((19,19,5,-1))

    anchors = metadata['anchors']
    nb_class = metadata['classes']
    
    boxes = decode_netout(result, anchors, nb_class)

    if boxes:
        if metadata['labels'][boxes[0].label] == target:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0

async def sendNNRequest(metadata, spm_frames, refm_frames, c_low, c_high, target):
    async with aiohttp.ClientSession() as session:
        r1 = await sendSNNRequest(session, spm_frames)
        if r1 > c_high or r1 < c_low:
            return r1
        else:
            r2 = await sendRNNRequest(metadata, session, refm_frames, target)
            return r2

def runSendNNRequest(corofn, *args):
    loop = asyncio.new_event_loop()
    try:
        coro = corofn(*args)
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def sendAllRequests(executor, metadata, X, origX, c_low, c_high, target):
    loop = asyncio.get_event_loop()
    blocking_tasks = []
    for i in range(len(X)):
        task = loop.run_in_executor(executor, runSendNNRequest, sendNNRequest, metadata, X[i:i+1], origX[i:i+1], c_low, c_high, target)
        blocking_tasks.append(task)

    completed, pending = await asyncio.wait(blocking_tasks)
    results = [t.result() for t in completed]

    return results 

def main(num_of_frames=50, num_of_workers=4):
    video_fname = "/data/dataset/jackson-town-square.mp4"
    csv_in = "/data/dataset/jackson-town-square.csv"
    avg_fname = "/data/dataset/jackson-town-square.npy"
    avg_fname_orig = "/data/dataset/jackson-town-square_400_600.npy"
    num_frames = num_of_frames
    start_frame = 5001
    target = 'car'
    
    # Param: difference detector
    dist_metric = 'mse'
    thresh = 0.00024391713548410724
    frame_delay = 1 
    feature_fn = RawImage.compute_feature
    get_distance_fn = RawImage.get_distance_fn

    # Param: specialized NN
    c_low = 0.1
    c_high = 0.9

#    start = time.time()

    data, nb_classes = dataPrep.get_data_for_test(
            csv_in, video_fname, avg_fname,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=[target],
            resol=(50, 50), shiftEn=1)
    X, Y = data

    origData, orig_nb_classes = dataPrep.get_data_for_test(
            csv_in, video_fname, avg_fname_orig,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=[target],
            resol=(608, 608), shiftEn=0)
    origX, origY = origData

    meta_f = open('yolo.meta','r')
    metadata = json.loads(meta_f.read())

    # Run Difference Detector, and generate index of frames which need Specialized NN

    start = time.time()

    todos = [0]
    for i in range(1, len(X)):
        features = get_features(feature_fn, X[i-1:i+1])
        dists = get_distances(get_distance_fn(dist_metric), features, frame_delay)
        if dists > thresh:
            todos.append(i)
 
    X_new = X[todos]
    origX_new = origX[todos]
    ''' 
    tasks = []
    for i in range(len(X_new)):
        task = asyncio.ensure_future(sendNNRequest(metadata, X_new[i:i+1], origX_new[i:i+1], c_low, c_high, target))
        tasks.append(task)
 
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(asyncio.gather(*tasks)) 
    '''
    '''
    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=10,
    )
    '''
    executor = concurrent.futures.ProcessPoolExecutor(
       max_workers = num_of_workers,
    )

    loop = asyncio.get_event_loop()
    loop.run_until_complete(sendAllRequests(executor, metadata, X_new, origX_new, c_low, c_high, target))  

    print('Num of RNN: ', str(num_of_RNN), " Run Time: ", time.time()-start)


if __name__ == "__main__":
    
    num_of_frames = int(sys.argv[1])
    num_of_SNN = int(sys.argv[2])
    num_of_RNN = int(sys.argv[3])

    main(num_of_frames, 2)

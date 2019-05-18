import dataPrep
import os
import time
import sys
import threading
import numpy
import requests
import json


SERVER_URL = 'http://localhost:8502/v1/models/spm'

def main():
    if not SERVER_URL:
        print('please specify server host:port')
        return

    video_fname = "/data/dataset/jackson-town-square.mp4"
    csv_in = "/data/dataset/jackson-town-square.csv"
    avg_fname = "/data/dataset/jackson-town-square.npy"
    num_frames = 500
    start_frame = 1

    data, nb_classes = dataPrep.get_data_for_test(
            csv_in, video_fname, avg_fname,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=['car'],
            resol=(50, 50))

    X, Y = data
    print(Y)
    #r = requests.get(SERVER_URL + '/' + 'metadata')
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(X)):
        print("Sending request for %d frame" %(i))
        payload = {
                   'signature_name': 'serving_default',
                   'inputs': 
                      { 
                        'image': [ X[i].tolist() ]
                      },
                  }

        # print(payload) 
        r = requests.post(SERVER_URL + ':predict', json=payload)
        probs = r.json()['outputs']
        print('Confidence: %f'%(probs[0][1]))
        if probs[0][1] > 0.5:
            if Y[i][1] == 1.:
                TP +=1
            else:
                FP +=1
        else:
            if Y[i][1] == 0.:
                TN +=1
            else:
                FN +=1

    print('Accuracy: %f' % ((TP+TN)/float(TP+FP+TN+FN)))

if __name__ == '__main__':
    main()


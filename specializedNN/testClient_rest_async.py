import dataPrep
import os
import time
import sys
import threading
import numpy
import requests
import json
import asyncio
import aiofiles
import aiohttp



def main():
    SERVER_URL = 'http://localhost:8502/v1/models/spm'
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

    '''   
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
    '''
    async def sendRequests(X,Y):
        pass
    loop = asyncio.get_event_loop()
    loop.run_until_complete(sendRequests(X,Y))
     

if __name__ == '__main__':
    main()


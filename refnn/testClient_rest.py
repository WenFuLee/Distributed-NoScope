import json
import cv2
from utils import decode_netout
import requests
import numpy as np

f = open('yolo.meta','r')
metadata = json.loads(f.read())
SERVER_URL = 'http://localhost:8504/v1/models/refm'

frame = cv2.imread('/data/dataset/images/frame1.jpg')
frame = cv2.resize(frame, (608, 608))
frame = frame / 255. 

payload = {'signature_name': 'serving_default',
           'inputs': 
              { 
                  'images': [ frame.tolist() ]
              }
          }
r = requests.post(SERVER_URL + ':predict', json = payload)
result = r.json()['outputs'][0] #(19,19,425) = (19,19,5*(5+80))
result = np.array(result)
result = result.reshape((19,19,5,-1))

anchors = metadata['anchors']
nb_class = metadata['classes']
# print(result.shape)
# print(len(anchors))
# print(nb_class)


boxes = decode_netout(result, anchors, nb_class)
if boxes:
    # print(boxes[0].classes)
    # print(boxes[0].label)
    print(metadata['labels'][boxes[0].label])
else:
    print('Obj not found!')

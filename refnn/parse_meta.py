import json

f = open('yolo.meta','r')
obj = json.loads(f.read())
# print(len(obj['labels']))    # 80

import tensorflow as tf
import sys
from tensorflow.python.platform import gfile

from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.util import compat
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

import cv2
from utils import decode_netout

import numpy as np
import json
import os


f = open('yolo.meta','r')
metadata = json.loads(f.read())

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45, allow_growth=False)
config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

with tf.Session(config=config) as sess:

  with gfile.FastGFile("yolo.pb",'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())    
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes=[n for n in graph_def.node]
    # tensor_names = [[v.name for v in op.values()] for op in sess.graph.get_operations()]
    # for n in tensor_names:
    #   print(n)
    input_tensor = sess.graph.get_tensor_by_name('input:0')
    output_tensor = sess.graph.get_tensor_by_name('output:0')
    ''' 
    # frame = [ np.ndarray(shape=(608,608,3))]
    frame = cv2.imread('/data/dataset/images/frame3.jpg')
    frame = cv2.resize(frame, (608, 608))
    frame = frame / 255. 
    frames = [frame]
    result = sess.run(output_tensor,{'input:0':frames})
    result = result[0] #(19,19,425) = (19,19,5*(5+80))
    result = result.reshape((19,19,5,-1))
 
    anchors = metadata['anchors']
    nb_class = metadata['classes']
    print(result.shape)
    print(len(anchors))
    print(nb_class)
     
    boxes = decode_netout(result, anchors, nb_class)
    print(boxes[0].classes)
    print(boxes[0].label)
    print(metadata['labels'][boxes[0].label])

    '''
    export_path_base = "/data/refm_cpu"
    export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes("2"))
    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def({"images": input_tensor}, {"prediction":output_tensor})
    builder = saved_model_builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)

    builder.save() 
    

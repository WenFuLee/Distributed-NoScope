import models
import dataPrep
import tempfile
import os
import time
import tensorflow as tf
from keras import backend as K

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.45, allow_growth=False)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Missing this was the source of one of the most challenging an insidious bugs that I've ever encountered.  
# Without explicitly linking the session the weights for the dense layer added below don't get loaded
# and so the model returns random results which vary with each model you upload because of random seeds.
K.set_session(sess)

# Use this only for export of the model.  
# This must come before the instantiation of ResNet50
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)


video_fname = "/data/dataset/jackson-town-square.mp4"
csv_in = "/data/dataset/jackson-town-square.csv"
avg_fname = "/data/dataset/jackson-town-square_400_600.npy"
num_frames = 6250 # 6250*0.8 = 5000
start_frame = 1

data, nb_classes = dataPrep.get_data_with_split(
            csv_in, video_fname, avg_fname,
            num_frames=num_frames,
            start_frame=start_frame,
            OBJECTS=['car'],
            resol=(400, 600),
            train_ratio=0.8)


X_train, Y_train, X_test, Y_test = data

model = models.generate_conv_net_base(input_shape=X_train.shape[1:] , nb_classes=nb_classes)
temp_fname = tempfile.mkstemp(suffix='.hdf5', dir='/data/tmp')[1]

### Training
model.fit(X_train, Y_train,
                  batch_size=32,
                  nb_epoch=1,
                  shuffle=True,
                  class_weight='auto',
                  callbacks=models.get_callbacks(temp_fname))


### Testing
eval_result = model.evaluate(X_test,Y_test)

print(model.metrics_names)
print(eval_result)
pred_result = model.predict(X_test)
# print(pred_result)

## Making a model
# model.load_weights(temp_fname)
os.remove(temp_fname)


prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
                        {"image": model.input}, {"prediction":model.output}
                       )

valid_prediction_signature = tf.saved_model.signature_def_utils.is_valid_signature(prediction_signature)
if(valid_prediction_signature == False):
    raise ValueError("Error: Prediction signature not valid!")

export_path_base = "/data/spm"
export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes("1"))

builder = saved_model_builder.SavedModelBuilder(export_path)
legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

# Initialize global variables and the model
# init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
# sess.run(init_op)

# Add the meta_graph and the variables to the builder
builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
           signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
               prediction_signature,
      },
      legacy_init_op=legacy_init_op)
# save the graph      
builder.save()  
sess.close()

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_path)
    graph = tf.get_default_graph()

 

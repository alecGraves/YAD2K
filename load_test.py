'''

'''
import os
import tensorflow as tf
import numpy as np
from keras.models import load_model
from yad2k.models.keras_yolo import yolo_body

from yad2k.models.keras_yolo import yolo_eval, yolo_head

# Set up paths.
filepath = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(filepath, 'model_data', 'trained_body.h5')
anchors_path = os.path.join(filepath, 'model_data', 'yolo_anchors.txt')
classes_path = os.path.join(filepath, 'model_data', 'aerial_classes.txt')

sess = K.get_session()

# Load classes and anchors.
with open(classes_path) as f:
        class_names = f.readlines()
class_names = [c.strip() for c in class_names]

with open(anchors_path) as f:
    anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)

# Load model and set up computation graph.
model_body = load_model(model_path)

model_body.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

input_image_shape = K.placeholder(shape=(2, ))

boxes, scores, classes = yolo_eval(
    yolo_outputs,
    input_image_shape,
    score_threshold=args.score_threshold,
    iou_threshold=args.iou_threshold)


def predict(image_s):
    assert len(image_s.shape) == 4 # image(s) ha(s/ve) 4 dims (batch, height, width, channel), ready to be sent into the graph
    out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                yolo_model.input: image_s,
                input_image_shape: [image_s.size[1], image_s.size[0]],
                K.learning_phase(): 0
            })
    ros.info('found ')


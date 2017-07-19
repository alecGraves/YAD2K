"""
This is a script that can be used to retrain the YOLOv2 model for your own dataset.
"""
from __future__ import division
import argparse
import os
import time
import json

import matplotlib.pyplot as plt
import numpy as np
import PIL
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, model_from_json, Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body,
                                     yolo_eval, yolo_head_np, yolo_loss,
                                     SpaceToDepth)
from yad2k.utils.draw_boxes import draw_boxes

FILEDIR = os.path.dirname(os.path.abspath(__file__))

# Args
argparser = argparse.ArgumentParser(
    description="Retrain or 'fine-tune' a pretrained YOLOv2 model for your own data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images'",
    default=os.path.join('..', 'DATA', 'my_data.npz'))

argparser.add_argument(
    '-a',
    '--anchors_path',
    help='path to anchors file, defaults to yolo_anchors.txt',
    default=os.path.join('model_data', 'yolo_anchors.txt'))

argparser.add_argument(
    '-c',
    '--classes_path',
    help='path to classes file, defaults to pascal_classes.txt',
    default=os.path.join('..', 'DATA', 'my_classes.txt'))

# Default anchor boxes
YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))

CUSTOM_DICT = {'SpaceToDepth' : SpaceToDepth}

def _main(args):
    data_path = os.path.expanduser(args.data_path)
    classes_path = os.path.expanduser(args.classes_path)
    anchors_path = os.path.expanduser(args.anchors_path)

    class_names = get_classes(classes_path)
    anchors = get_anchors(anchors_path)

    data = np.load(data_path) # custom data saved as a numpy file.
    #  has 2 arrays: an object array 'boxes' (variable length of boxes in each image)
    #  and an array of images 'images'

    image_data, boxes, detectors_mask, matching_true_boxes = process_data(data['images'], data['boxes'], anchors)

    model_body, model = create_model(anchors, class_names)

    # Training
    train(
        model,
        class_names,
        anchors,
        image_data,
        boxes,
        detectors_mask,
        matching_true_boxes
    )

    # Evaulation
    draw(model_body,
        class_names,
        anchors,
        image_data,
        image_set='val', # assumes training/validation split is 0.9
        weights_name='trained_stage_2_best.h5',
        score_threshold=0.3,
        iou_threshold=0.6)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None, anchors=None, image_train_shape=(416, 416)):
    '''
    Processes the data for training.
    # Params:

    images: Numpy array or list with each image allowed to have its own height and width. Must have 3 channels.

    boxes: Original boxes stored as 1D array of [class, x_min, y_min, x_max, y_max]

    anchors: List of tuples describing box priors

    image_train_shape: Shape to convert dataset to for training. Defaults to (416, 416). Must be a multiple of 32.

    # Returns:

    images: Images as an array for training. Has shape (nb_images, image_train_shape, 3)
    
    boxes: Proessed boxes for training. Returned if input boxes specified.
    
    detectors_mask: for training; is 1 for each spatial position in the final conv layer and \
    anchor that should be active for the given boxes and 0 otherwise. Returned if boxes and anchors specified.
    
    matching_true_boxes: for training; gives the regression targets for the ground truth box \
    that caused a detector to be active or 0 otherwise. Returned if boxes and anchors specified.
    '''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_sizes = [np.expand_dims(np.array([image.width, image.height]), axis=0) for image in images]

    # Image preprocessing.
    images = [np.array(i.resize(image_train_shape, PIL.Image.BICUBIC), dtype=np.float16)/255 for i in images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for orig_size, boxxy in zip(orig_sizes, boxes_xy)]
        boxes_wh = [boxwh / orig_size for orig_size, boxwh in zip(orig_sizes, boxes_wh)]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))
        
        if anchors is not None:
            detectors_mask, matching_true_boxes = _get_detector_mask(boxes, anchors, image_train_shape)
            return np.array(images), np.array(boxes), detectors_mask, matching_true_boxes

        return np.array(images), np.array(boxes)

    else: # boxes not given
        return np.array(images)

def _get_detector_mask(boxes, anchors, image_train_shape=(416, 416)):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, image_train_shape)

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, load_pretrained=True, frozen=None, json_path=None):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    frozen: fraction of model to be frozen, defaults to all but last layer.

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (None, None, 5, 1)
    matching_boxes_shape = (None, None, 5, len(anchors))

    # Create model input layers.
    image_input = Input(shape=(None, None, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Save model as JSON.
    yolo = None
    if json_path is None:
        yolo_json_path = os.path.join(FILEDIR, 'model_data', 'yolo.json')
    else:
        yolo_json_path = json_path

    if not os.path.isfile(yolo_json_path): # if not already saved:
        # serialize model to JSON and save
        yolo_path = os.path.join(FILEDIR, 'model_data', 'yolo.h5')
        yolo = load_model(yolo_path, CUSTOM_DICT)
        yolo_json = yolo.to_json()
        with open(yolo_json_path, "w") as json_file:
            json_file.write(yolo_json)

    # Load model from JSON.
    yolo_json_file = open(yolo_json_path, 'r')
    yolo_json = yolo_json_file.read()
    yolo_json_file.close()
    yolo_model = model_from_json(yolo_json, CUSTOM_DICT)


    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)


    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join(FILEDIR, 'model_data', 'yolo_topless.h5')
        if not os.path.isfile(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            if yolo is None: # yolo is not loaded
                # load yolo
                yolo_path = os.path.join(FILEDIR, 'model_data', 'yolo.h5')
                yolo = load_model(yolo_path, CUSTOM_DICT)

            model_body = Model(yolo.inputs, yolo.layers[-2].output)
            model_body.save_weights(topless_yolo_path)

        topless_yolo.load_weights(topless_yolo_path)


    if frozen is None:
        # Freeze all layers.
        for layer in topless_yolo.layers:
            layer.trainable = False
    elif frozen > .0001:
        # Freeze first <frozen>% of the model.
        unfreeze = int(frozen*len(topless_yolo.layers))
        for i, layer in enumerate(topless_yolo.layers):
            if i < unfreeze:
                layer.trainable = False

    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear', name='final_layer')(topless_yolo.output)

    model_body = Model(yolo_model.input, final_layer)


    # TODO: Replace Lambda with custom Keras layer for loss.
    model_loss = Lambda(
        yolo_loss,
        output_shape=(1, ),
        name='yolo_loss',
        arguments={'anchors': anchors,
                    'num_classes': len(class_names)})([
                        model_body.output, boxes_input,
                        detectors_mask_input, matching_boxes_input])


    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)


    return model_body, model

def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes, validation_split=0.1):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''
    #--------------------------------------------------------------
    # Train for 5 epochs, allowing weights of last layer to adjust
    #--------------------------------------------------------------
    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.

    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=validation_split,
              batch_size=32,
              epochs=5)

    model.save_weights('trained_stage_1.h5')


    #--------------------------------------------------------------
    # train with fewer frozen layers, stop when improvements stop
    #--------------------------------------------------------------
    model_body, model = create_model(anchors, class_names, load_pretrained=False, frozen=.7) # freeze first 20 layers

    model.load_weights('trained_stage_1.h5')

    checkpoint = ModelCheckpoint("trained_stage_2_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto')

    model.compile(
        optimizer='adam', loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        })  # This is a hack to use the custom loss function in the last layer.


    model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
              np.zeros(len(image_data)),
              validation_split=0.1,
              batch_size=8,
              epochs=30,
              callbacks=[checkpoint, early_stopping])

    model.save_weights('trained_stage_2.h5')

def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_2_best.h5', out_path="output_images",
            save_all=True, score_threshold=0.3, iou_threshold=0.6):
    '''
    Draw bounding boxes on image data
    '''
    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")
    # model.load_weights(weights_name)
    print(image_data.shape)
    model_body.load_weights(weights_name)
    yolo_model = model_body # TODO: turn yolo_head into a layer and attach it here with Model()

    # Run predictions on specified images
    if  not os.path.exists(out_path):
        os.makedirs(out_path)

    num_classes = len(class_names)
    num_images = len(image_data)

    print("EVALUATION START")
    total_time = 0

    for i, image in enumerate(image_data):
        start = time.time() # Time the computation
        pred = yolo_model.predict(image)   
        yolo_out = yolo_head_np(pred, anchors, num_classes)

        out_boxes, out_scores, out_classes = yolo_eval(
            yolo_out,
            (image.shape[1], image.shape[2]),
            score_threshold=0.3,
            iou_threshold=0.6)

        end = time.time() # Do not include time to write to disk
        total_time += end - start

        print('Found {} boxes for image.'.format(len(out_boxes)))
        print(out_boxes)

        # Plot image with predicted boxes.
        image_with_boxes = draw_boxes(image[0], out_boxes, out_classes,
                                    class_names, out_scores)
        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))

        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()

    print('Predicted on',num_images , 'images in', total_time, 'seconds')
    print('That is', num_images/total_time, 'frames per second!')

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)

from keras.layers import Input
from yad2k.models import keras_yolo
image_input = Input(shape=(416, 416, 3))
yolo_model = keras_yolo.yolo_body(image_input, 13, 17)

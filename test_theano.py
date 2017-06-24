from keras.layers import Input
from keras.models import load_model
from yad2k.models import keras_yolo
from yad2k.models.keras_yolo import SpaceToDepth
# image_input = Input(shape=(416, 416, 3))
# yolo_model = keras_yolo.yolo_body(image_input, 13, 17)
# yolo_model.summary()
# yolo_model.compile(loss='categorical_crossentropy', optimizer='adam')
# yolo_model.save("test.h5")
load_model("test.h5").summary()
exit()
# !/usr/bin/env python

import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from PIL import Image
import glob

keras_file = "keras_cnn_model.h5"

# TFLiteでコンバータを用意
converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(keras_file
    # ,input_arrays=['input_input'], output_arrays=['output/Softmax'], input_shapes={'input_input': [None, 50, 50, 3]}
)
# コンバート
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

#!/usr/bin/env python
# coding: utf-8

# from __future__ import absolute_import, division, print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import sys
import pickle
import re
import numpy as np
import time
import json
from glob import glob
from PIL import Image


from utils import create_flickr_dict
from params import *
from dl_classes import *

num_examples = int(num_batches * BATCH_SIZE / (1-TEST_SIZE))

def create_flickr_val_json(imgs_caps):
    val_json = []
    for img, cap in imgs_caps.items():
        try:
            # captions = images_dict[img]
            # captions = [x for x in data['annotations'] if x['image_id'] == img]

            # for caption in captions:
            val_json.append({'image_id': img,
                             "id": 0,
                             "caption": cap})
        except Exception as e:
            print(e)
            continue

    with open(os.path.join(checkpoint_load_path, 'flickr_captions_val.json'), 'wt') as f:
        json.dump(val_json, f)


def load_image(image_path):
    print('load image: ', image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    print('loaded image: ', img.shape)
    if vgg:
        print('vgg')
        img = tf.image.resize(img, (224, 224))
        new_img = tf.keras.applications.vgg16.preprocess_input(img, data_format='channels_last')
    else:
        print('inception')
        img = tf.image.resize(img, (299, 299))
        new_img = tf.keras.applications.inception_v3.preprocess_input(img)
    print('processed: ', new_img.shape, img.shape)

    return new_img, image_path


def evaluate(image):
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    # print('img shape: ', img_tensor_val.shape)

    # print('here?')
    features = encoder(img_tensor_val)
    # print("feats shape: ", features.shape)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):

        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)
    print(result)
    return result


with open(os.path.join(checkpoint_load_path, 'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)
max_length = 64

if vgg:
    print('Using vgg.')
    image_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
else:
    print('Using Inception V3.')
    image_model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))

new_input = image_model.input
hidden_layer = image_model.layers[-2].output
num_feats = 64

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
print('vocab size: ', vocab_size)

features_shape = hidden_layer.shape[-1]

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_load_path, max_to_keep=5)
ckpt.restore(ckpt_manager.latest_checkpoint)
print('loaded from checkpoint: ', checkpoint_load_path)

image_path = os.path.abspath(single_image_val)
result = evaluate(image_path)
print('Prediction Caption:', ' '.join(result))

# if data_format == 'flickr':
#     with open(flickr_dev, 'rt') as f:
#         imgs = [os.path.join(PATH, x.strip()) for x in f.readlines()]
#
#     total = len(imgs)
#     imgs_caps = {}
#     for img in imgs:
#         imgs_caps[os.path.split(img)[-1]] = evaluate(img)
#         print('Todo: ', total-len(imgs_caps))
#     create_flickr_val_json(imgs_caps)
# elif data_format == 'coco':
#
# else:
#     print('Not a valid format.')
#     sys.exit(-1)


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

import re
import numpy as np
import time
import json
# from glob import glob
from PIL import Image
# import pickle

from utils import create_flickr_dict
from params import *
from dl_classes import *

num_examples = int(num_batches * BATCH_SIZE / (1-TEST_SIZE))

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
    # attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)
    # print('evaluate image:', image)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    print('img shape: ', img_tensor_val.shape)
    # img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    print('here?')
    features = encoder(img_tensor_val)
    print("feats shape: ", features.shape)
    # print('features shape: ', features.shape)
    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):

        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        # print(attention_weights.shape)
        # attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            # return result, attention_plot
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    # attention_plot = attention_plot[:len(result), :]
    # return result, attention_plot
    return result


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def coco_tokens():
# Read the json file
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    # Store captions and image names in vectors
    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    # Shuffle captions and image_names together
    # Set a random state
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)

    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]




    # Choose the top 5000 words from the vocabulary
    top_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    max_length = calc_max_length(train_seqs)

    return tokenizer, max_length


def flickr_tokens():
    images_dict = create_flickr_dict(flickr_captions)

    with open(flickr_training, 'rt') as f:
        train_images = [os.path.join(PATH, x.strip()) for x in f.readlines()]
    with open(flickr_test, 'rt') as f:
        test_images = [os.path.join(PATH, x.strip()) for x in f.readlines()]

    train_size = len(train_images)

    train_captions = []
    for lst_imgs in [train_images, test_images]:
        for img in lst_imgs:
            train_captions.append('<start> ' + images_dict[os.path.split(img)[-1]][0].strip() + ' <end>')

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    max_length = calc_max_length(train_seqs)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    return tokenizer, max_length


if data_format == 'coco':
    print('using coco tokens')
    tokenizer, max_length = coco_tokens()
else:
    print('using flickr tokens')
    tokenizer, max_length = flickr_tokens()

# def create_full_architecture(tokenizer):
if vgg:
    print('Using vgg.')
    # image_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    image_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_shape=(224, 224, 3))
else:
    print('Using Inception V3.')
    # image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
    image_model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', input_shape=(299, 299, 3))

new_input = image_model.input
hidden_layer = image_model.layers[-2].output
# num_feats = int(np.multiply(*hidden_layer.shape[1:3]))
num_feats = 64

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

# BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1
print('vocab size: ', vocab_size)
# num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = hidden_layer.shape[-1]
# attention_features_shape = num_feats

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

    # return encoder, decoder, image_features_extract_model

#
# def main():
#     if data_format == 'coco':
#         print('using coco tokens')
#         tokenizer = coco_tokens()
#     else:
#         print('using flickr tokens')
#         tokenizer = flickr_tokens()
#
#     encoder, decoder, image_features_extract_model = create_full_architecture(tokenizer)
#
#     image_path = os.path.abspath(single_image_val)
#     result = evaluate(image_path, decoder, image_features_extract_model, encoder, tokenizer)
#     print('Prediction Caption:', ' '.join(result))
#
#
# if __name__ == '__main__':
#     main()
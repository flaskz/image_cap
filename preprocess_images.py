#!/usr/bin/env python
# coding: utf-8

# from __future__ import absolute_import, division, print_function, unicode_literals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import json
from params import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_examples = int(num_batches * BATCH_SIZE / (1-TEST_SIZE))


def load_image(image_path):
    # print('load image: ', image_path)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # print('loaded image: ', img.shape)
    if vgg:
        print('vgg')
        img = tf.image.resize(img, (224, 224))
        new_img = tf.keras.applications.vgg16.preprocess_input(img, data_format='channels_last')
    else:
        print('inception')
        img = tf.image.resize(img, (299, 299))
        new_img = tf.keras.applications.inception_v3.preprocess_input(img)
    # print('processed: ', new_img.shape, img.shape)

    return new_img, image_path


def preprocess_coco():
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []

    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)


    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=1)


    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]
    #
    # import os
    # names = [os.path.split(x)[-1] for x in img_name_vector]
    # np_files = [x.split('.npy')[0] for x in os.listdir('train2014') if x.endswith('npy')]
    # len(set(names))
    # n_tem = []
    # i = 0
    # for x in names:
    #     print(len(names)-i)
    #     i += 1
    #     if x not in np_files:
    #         n_tem.append(x)
    # img_name_vector = [os.path.join(PATH, x) for x in os.listdir(PATH) if x.endswith('.jpg')]
    # print(PATH)
    # print(img_name_vector)
    return img_name_vector

def preprocess_flickr():
    all_names = []
    with open(flickr_training, 'rt') as f:
        [all_names.append(os.path.join(PATH, x.strip())) for x in f.readlines()]
    with open(flickr_dev, 'rt') as f:
        [all_names.append(os.path.join(PATH, x.strip())) for x in f.readlines()]
    with open(flickr_test, 'rt') as f:
        [all_names.append(os.path.join(PATH, x.strip())) for x in f.readlines()]
    return all_names


def process_images(img_name_vector):
    if vgg:
        print('Using vgg.')
        image_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        # new_input = image_model.input
        # hidden_layer = image_model.layers[-1].output
    else:
        print('Using Inception V3.')
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    num_feats = int(np.multiply(*hidden_layer.shape[1:3]))

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # Get unique images
    encode_train = sorted(set(img_name_vector))
    # print(encode_train)

    # Feel free to change batch_size according to your system configuration
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(lambda x: load_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(8)

    n_processed_imgs = 0
    total = len(encode_train)
    for img, path in image_dataset:
      batch_features = image_features_extract_model(img)
      batch_features = tf.reshape(batch_features,
                                  (batch_features.shape[0], -1, batch_features.shape[3]))

      for bf, p in zip(batch_features, path):
        path_of_feature = p.numpy().decode("utf-8")
        np.save(path_of_feature, bf.numpy())
        print('Remains: ', total - n_processed_imgs)
        n_processed_imgs += 1


def main():
    if data_format == 'coco':
        img_name_vector = preprocess_coco()
    elif data_format == 'flickr':
        img_name_vector = preprocess_flickr()
    else:
        print('Invalid data format.')
        import sys
        sys.exit(-1)

    process_images(img_name_vector)
    print('Finished processing images.')



if __name__ == '__main__':
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--annot_file', type=str, default='/tmp')
    # parser.add_argument('--vgg', type=bool, default=False)
    # parser.add_argument('--path_images', type=str, default='/tmp')
    # parser.add_argument('--num_examples', type=int, default=64)
    # args = parser.parse_args()

    main()


#!/usr/bin/env python
# coding: utf-8

# from __future__ import absolute_import, division, print_function, unicode_literals
import sys
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

from params import *
from dl_classes import *

num_examples = int(num_batches * BATCH_SIZE / (1-TEST_SIZE))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def create_data_vecs():
    all_captions = []
    all_img_name_vector = []
    if data_format == 'coco':
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

            all_img_name_vector.append(full_coco_image_path)
            all_captions.append(caption)
    elif data_format == 'flickr':
        images_dict = {}
        with open(flickr_captions, 'rt') as f:
            for line in f.readlines():
                line = line.strip()
                img_name, img_cap = line.split('\t')
                img_name = img_name[:-2]
                if img_name not in images_dict.keys():
                    images_dict[img_name] = []
                images_dict[img_name].append(img_cap)

        for img_name, img_caps in images_dict.items():
            caption = '<start> ' + img_caps[0] + ' <end>'
            full_flickr_image_path = os.path.join(PATH, img_name)

            all_img_name_vector.append(full_flickr_image_path)
            all_captions.append(caption)
    else:
        print('Formato invalido.')
        sys.exit(-1)

    return all_img_name_vector, all_captions


def shuffle_data(all_img_name_vector, all_captions, rs=1):
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=rs)

    train_captions = train_captions[:num_examples]
    img_name_vector = img_name_vector[:num_examples]

    return img_name_vector, train_captions


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def create_training_data(img_name_vector, train_captions, top_k=5000, rs=0):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                        cap_vector,
                                                                        test_size=TEST_SIZE,
                                                                        random_state=rs)

    return img_name_train, img_name_val, cap_train, cap_val, max_length, tokenizer


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


@tf.function
def train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def create_architecture(img_name_train, cap_train, tokenizer):
    if vgg:
        print('Using vgg.')
        image_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    else:
        print('Using Inception V3.')
        image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    num_feats = int(np.multiply(*hidden_layer.shape[1:3]))

    BUFFER_SIZE = 1000
    embedding_dim = 256
    units = 512
    vocab_size = len(tokenizer.word_index) + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    features_shape = hidden_layer.shape[-1]
    attention_features_shape = num_feats

    print('Vocab size: ', vocab_size)

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
              map_func, [item1, item2], [tf.float32, tf.int32]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    return dataset, encoder, decoder, optimizer, num_steps


def start_training(dataset, encoder, decoder, optimizer, tokenizer, num_steps, checkpoint_path=".\\checkpoints\\incept3"):
    loss_plot = []
    print('Starting training.')
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)
    start_epoch = 0
    # if ckpt_manager.latest_checkpoint:
    #   start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            # print(img_tensor.shape)
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if (epoch+1) % 5 == 0:
          ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


def main():
    print('Reading data.')
    all_imgs, all_caps = create_data_vecs()
    print('Shuffling data.')
    imgs_path, imgs_caps = shuffle_data(all_imgs, all_caps)
    print('Creating dataset.')
    img_name_train, img_name_val, cap_train, cap_val, max_length, tokenizer = create_training_data(imgs_path, imgs_caps)
    print('X_train, y_train, X_test, y_test: ', len(img_name_train), len(cap_train), len(img_name_val), len(cap_val))
    print('Creating architecture.')
    dataset, encoder, decoder, optimizer, num_steps = create_architecture(img_name_train, cap_train, tokenizer)
    print('Starting training.')
    start_training(dataset=dataset,
                   encoder=encoder,
                   decoder=decoder,
                   optimizer=optimizer,
                   tokenizer=tokenizer,
                   num_steps=num_steps,
                   checkpoint_path=checkpoint_save_path)
    print('Finished training.')


if __name__ == '__main__':
    main()
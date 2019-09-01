#!/usr/bin/env python
# coding: utf-8

# from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pickle
import re
import numpy as np
import time
import json
# from glob import glob
from PIL import Image
# import pickle

from params import *
from dl_classes import *
from utils import create_flickr_dict, load_weights

word_weight = load_weights('flickr_vgg_json/tag2score_list_2.json')
num_examples = int(num_batches * BATCH_SIZE / (1-TEST_SIZE))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

# a = str(int('inception/COCO_train2014_000000124567.jpg.npy'.split('.')[0].split('_')[-1]))
# r = []
# for k, v in word_weight.items():
#     if a in list(v.keys()):
#         print('achou')
#         r.append(k)

def create_data_coco():
    all_captions = []
    all_img_name_vector = []

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    return all_img_name_vector, all_captions


def generate_flickr_dataset():
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

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    max_length = calc_max_length(train_seqs)

    return train_images, test_images, cap_vector[:train_size], cap_vector[train_size:], max_length, tokenizer
    # return train_images[:16], test_images[:4], cap_vector[:16], cap_vector[16:20], max_length, tokenizer


def shuffle_data(all_img_name_vector, all_captions, rs=1):
    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=rs)
    if not full_coco_dataset:
        train_captions = train_captions[:num_examples]
        img_name_vector = img_name_vector[:num_examples]

    return img_name_vector, train_captions


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def create_training_data(img_name_vector, train_captions, top_k=5000, rs=0):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ')

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
    # print('img name:', img_name)
    load_path = os.path.join(save_features_path, os.path.split(img_name.decode('utf-8') + '.npy')[-1])
    # print('load path:', load_path)
    img_tensor = np.load(load_path)
    # print('img name: ', str(os.path.split(img_name)[-1]).split('.'))
    # print('path img id: ', str(os.path.split(img_name)[-1]).split('.')[0].split('_')[-1])
    if data_format == 'coco':
        imgs_ids = str(int(os.path.split(img_name)[-1].decode('utf8').split('.')[0].split('_')[-1]))
    else:
        imgs_ids = os.path.split(img_name)[-1].decode('utf8').split('.')[0]

    return img_tensor, cap, imgs_ids


def loss_function(real, pred, weights_):
    # print('real: ', real)
    # print('pred: ', pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    weights_loss_ = np.multiply(loss_, weights_)

    # print('loss: ', loss_)
    # print('half loss: ', loss_*tf.cast(0.5, dtype=loss_.dtype))
    # print('reduce mean: ', tf.reduce_mean(loss_))
    # return tf.reduce_mean(weights_loss_)
    return tf.reduce_mean(loss_)

wrong_words = set()
# @tf.function
def train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, img_names):
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    # print('tensor shape:', img_tensor.shape)

    # for x in target[:1]:
    #     my_r = []
    #     for y in x:
    #         my_r.append(tokenizer.index_word[np.int(y)])
    #     print('True cap: ', ' '.join(my_r))

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        # print('feats shape: ', features.shape)

        # my_p = []
        for i in range(1, target.shape[1]):
            # print('features shape,', features.shape)
            predictions, hidden, _ = decoder(dec_input, features, hidden)

            # print('pred 0:', tf.argmax(predictions[7]).numpy())
            # my_p.append(tokenizer.index_word[tf.argmax(predictions[7]).numpy()])

            weights_each = []
            for new_i in range(BATCH_SIZE):
                true_id = np.array(target[:, i])
                true_word = tokenizer.index_word[true_id[new_i]]

                # print('true word: ', tokenizer.index_word[true_id])
                # print('true word: ', true_word)
                # predicted_id = tf.argmax(predictions[new_i]).numpy()
                # pred_word = tokenizer.index_word[predicted_id[new_i]]
                # print('pred word: ', tokenizer.index_word[predicted_id])
                try:
                    if data_format == 'coco':
                        weights_each.append(word_weight[true_word][str(np.int(img_names[new_i]))])
                    else:
                        weights_each.append(word_weight[true_word][img_names[new_i].numpy().decode('utf8')])
                except Exception as e:
                    # if true_word not in wrong_words:
                    #     wrong_words.add(true_word)
                    #     print(wrong_words)
                    weights_each.append(1)
            np_weights_words = np.array(weights_each)

            loss += loss_function(target[:, i], predictions, np_weights_words)
            dec_input = tf.expand_dims(target[:, i], 1)
        # print('Pred cap: ', ' '.join(my_p))
    total_loss = (loss / int(target.shape[1]))

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def create_architecture(img_name_train, cap_train, tokenizer):
    if vgg:
        print('Using vgg.')
        if generate_dict_dataset:
            image_model = tf.keras.applications.vgg16.VGG16(include_top=True, weights='imagenet',
                                                            input_shape=(224, 224, 3))
        else:
            image_model = tf.keras.applications.vgg16.VGG16(include_top=False, weights='imagenet',
                                                            input_shape=(224, 224, 3))
    else:
        print('Using Inception V3.')
        if generate_dict_dataset:
            image_model = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet',
                                                            input_shape=(299, 299, 3))
        else:
            image_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet',
                                                            input_shape=(299, 299, 3))

    new_input = image_model.input

    if generate_dict_dataset:
        hidden_layer = image_model.layers[-2].output
    else:
        hidden_layer = image_model.layers[-1].output

    # hidden_layer = image_model.layers[-2].output
    num_feats = 64

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
              map_func, [item1, item2], [tf.float32, tf.int32, tf.string]),
              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # dataset = dataset.prefetch(buffer_size=1)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    optimizer = tf.keras.optimizers.Adam()

    return dataset, encoder, decoder, optimizer, num_steps


def start_training(dataset, encoder, decoder, optimizer, tokenizer, num_steps, checkpoint_path=".\\checkpoints\\incept3"):
    loss_plot = []
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

        for (batch, (img_tensor, target, img_names)) in enumerate(dataset):
            # print('tensor shape: ', img_tensor.shape)
            # print('target shape: ', target.shape)
            # print('img name: ', img_names)
            batch_loss, t_loss = train_step(img_tensor, target, encoder, decoder, tokenizer, optimizer, img_names)
            total_loss += t_loss

            if batch % 100 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if (epoch+1) % 5 == 0:
          print('Saving checkpoint.')
          ckpt_manager.save()

        print('Epoch {} Loss {:.6f}'.format(epoch + 1,
                                             total_loss/num_steps))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    print('Saving checkpoint.')
    ckpt_manager.save()

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()


def main():
    if data_format == 'coco':
        print('Working with coco data.')
        print('Reading data.')
        all_imgs, all_caps = create_data_coco()
        print('Shuffling data.')
        imgs_path, imgs_caps = shuffle_data(all_imgs, all_caps)
        print('Creating dataset.')
        img_name_train, img_name_val, cap_train, cap_val, max_length, tokenizer = create_training_data(imgs_path,
                                                                                                       imgs_caps)
    elif data_format == 'flickr':
        print('Working with flickr data.')
        print('Generating dataset.')
        img_name_train, img_name_val, cap_train, cap_val, max_length, tokenizer = generate_flickr_dataset()
    else:
        print('Works only with flickr or coco data.')
        sys.exit(-1)


    # r = []
    # for x in cap_train[0]:
    #     r.append(tokenizer.index_word[x])
    # print(' '.join(r))
    # img_id = [x['id'] for x in annotations['images'] if x['file_name']==os.path.split(img_name_train[0])[-1]]
    # my_r = [x for x in annotations['annotations'] if x['image_id']==img_id[0]]
    #

    with open(os.path.join(checkpoint_save_path, 'tokenizer.pickle'), 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

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
    # print('Saving model.')
    #
    # encoder.save(os.path.join(checkpoint_save_path, 'encoder.h5'))
    # decoder.save(os.path.join(checkpoint_save_path, 'decoder.h5'))
    #
    # print('Models saved.')


if __name__ == '__main__':
    main()
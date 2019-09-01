import os
import json
import string
from collections import defaultdict
import numpy as np
from scipy.spatial import distance
# import cPickle as pkl
# import pdb
import math
from params import save_features_path, data_format, vgg

# img_featuer_path = "/home/zhaoxin/chenminghai/mRNN-CR/image_features_mRNN/VGG_feat_mRNN_refine_dct_mscoco_2014.npy"
# annotation_path = '/home/zhaoxin/chenminghai/data/coco/annotations/captions_split_train2014.json'
# img_feature_path = 'E:\\User\\freelancer\\image_cap\\wr\\feats\\VGG_feat_mRNN_refine_dct_mscoco_2014.npy'
# img_feature_path = 'E:\\User\\freelancer\\image_cap\\features_coco\\all_npy_dict.npy'
# annotation_path = 'E:\\User\\freelancer\\image_captioning\\annotations\\captions_train2014.json'
# annotation_path = 'E:\\User\\freelancer\\image_cap\\features_coco\\train_captions_train2014.json'

print('Starting weight process on {} data using {} architecture.'.format(data_format, 'vgg' if vgg else 'inception'))

img_feature_path = os.path.join(save_features_path, 'all_npy_dict.npy')
annotation_path = os.path.join(save_features_path, 'train_captions_to_weigth.json')

sigma = 37.382034917914233

# with open(img_featuer_path) as f:
#     features = np.load(f).tolist()

with open(img_feature_path, 'rb') as f:
    features = np.load(f, encoding='bytes').tolist()

with open(annotation_path, 'rt') as f:
    data = json.load(f)

# type(features)
#
# for k, v in features.items():
#     break
#
# a = np.array([[1,1,1], [2,2,2], [3,3,3], [1,1,1]])
#
# distances = distance.cdist(a, a[:2], 'euclidean')

tag2img_list = defaultdict(list)
for x in data['annotations']:
    # tokens = str(x['caption']).lower().translate(str.maketrans(dict.fromkeys(string.punctuation))).strip().split()
    tags = list(set(str(x['caption']).lower().translate(str.maketrans(string.punctuation, ' '*len(string.punctuation))).strip().split()))
    # for tag in tags:
    #     tag2img_list[tag].append(x['image_id'])
    [tag2img_list[tag].append(x['image_id']) for tag in tags]

for x in tag2img_list:
    tag2img_list[x] = list(set(tag2img_list[x]))

tag2score_list_1 = {}
tag2score_list_2 = {}
num = 0
for tag in tag2img_list:
    num += 1
    img_list = tag2img_list[tag]
    print(tag, ": ", len(img_list), '-', num, '/', len(tag2img_list))

    #if tag == 'a':
    #    continue
    # a = features.reshape(1,-1)
    feature_list = np.array([features[x] for x in img_list])
    # feature_list = features.reshape(1,-1)
    tag2score_list_1[tag] = dict()
    tag2score_list_2[tag] = dict()
    k = 30000 if len(feature_list) < 30000 else 10000
    if len(feature_list) == 1:
        print('ehh:', feature_list)
        feature_list.reshape(1, -1)
    print(feature_list.shape)
    distances = distance.cdist(feature_list, feature_list[0:k], 'euclidean')
    for i in range(len(img_list)):
        print('Todo: ', len(img_list) - i)
        dists = [math.exp(-x/sigma) for x in distances[i]]
        dists = [math.exp(-x / sigma) for x in distances[0]]
        tag2score_list_1[tag][img_list[i]] = float(sum(dists))/len(dists)/math.exp(-1)
        d2 = [math.exp(-x/sigma) for x in distances[i] if x != 0]
        if d2 == []:
            d2 = [math.exp(-1)]
        tag2score_list_2[tag][img_list[i]] = np.mean(d2)/math.exp(-1)
    #for i in range(len(img_list)):
    #   distances = distance.cdist([feature_list[i]], feature_list, "euclidean").tolist()
    #   tag2score_list_1[tag][img_list[i]] = np.mean([math.exp(-x/sigma) for x in distances[0]])
    #    d2 = [math.exp(-x/sigma) for x in distances[0] if x!=0 ]
    #    if d2 == []:
    #        d2 = [math.exp(-1)]
    #    tag2score_list_2[tag][img_list[i]] = np.mean(d2)

    if num % 1000 == 0:
        print('Saving json.')
        with open('tag2score_list_1.json', 'wt') as f:
            json.dump(tag2score_list_1, f)
        with open('tag2score_list_2.json', 'wt') as f:
            json.dump(tag2score_list_2, f)

# pdb.set_trace()
with open('tag2score_list_1.json', 'wt') as f:
    json.dump(tag2score_list_1, f)

with open('tag2score_list_2.json', 'wt') as f:
    json.dump(tag2score_list_2, f)




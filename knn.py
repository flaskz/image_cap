import os
import pickle

from sklearn.neighbors import NearestNeighbors
import numpy as np

from params import *
from utils import create_coco_dict


def create_knn_model():
    img_feature_path = os.path.join(save_features_path, 'all_npy_dict.npy')

    with open(img_feature_path, 'rb') as f:
        features_dict = np.load(f, encoding='bytes').tolist()

    # features_array = np.array(list(features_dict.values()))
    features_array = []
    imgs_id = []

    for k, v in features_dict.items():
        imgs_id.append(k)
        features_array.append(v)

    knn = NearestNeighbors(n_neighbors=1)
    knn.fit(features_array)
    print('Saving model to disk')
    with open(os.path.join(save_features_path, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn, f)
    # with open(os.path.join(save_features_path, 'features_array.pkl'), 'wb') as f:
    #     pickle.dump(features_array, f)
    with open(os.path.join(save_features_path, 'imgs_id.pkl'), 'wb') as f:
        pickle.dump(imgs_id, f)


def find_neigh_caps(knn, feature_array, imgs_id, imgs_dict, nn=5):
    results = knn.kneighbors(X=feature_array.reshape(1, -1), n_neighbors=nn)
    all_caps = []
    for r in results[1][0]:
        [all_caps.append(x) for x in imgs_dict[imgs_id[r]]]
    return all_caps


def main():
    print('Creating knn model')
    create_knn_model()


if __name__ == '__main__':
    main()

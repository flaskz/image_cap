import os
import numpy as np
import json
from params import *
from utils import create_flickr_dict

def get_data_npy():
    all_npy = [x for x in os.listdir(save_features_path) if x.endswith('.npy')]
    len(all_npy)

    aux_dict = {}
    if data_format == 'coco':
        for npy_file in all_npy:
            try:
                with open(os.path.join(save_features_path, npy_file), 'rb') as f:
                    aux_dict[int(npy_file.split('.')[0].split('_')[-1])] = np.load(f)
            except Exception as e:
                print('Couldn\'t load file:', npy_file)
                print('Error:', e)
                continue
    else:
        for npy_file in all_npy:
            try:
                with open(os.path.join(save_features_path, npy_file), 'rb') as f:
                    aux_dict[npy_file.split('.')[0]] = np.load(f)
            except Exception as e:
                print('Couldn\'t load file:', npy_file)
                print('Error:', e)
                continue

    with open(os.path.join(save_features_path, 'all_npy_dict.npy'), 'wb') as f:
        np.save(f, aux_dict)

    return aux_dict


def create_flickr_json(imgs_id):
    images_dict = create_flickr_dict(flickr_captions)
    new_json = {}
    new_json['annotations'] = []
    for k, captions in images_dict.items():
        img_id = k.split('.')[0]
        if img_id in imgs_id:
            for caption in captions:
                new_json['annotations'].append({'image_id': img_id, 'caption': caption})

    with open(os.path.join(save_features_path, 'train_captions_to_weigth.json'), 'wt') as f:
        json.dump(new_json, f)


def create_coco_train_json(imgs_id):
    with open(annotation_file, 'rt') as f:
        data = json.load(f)

    new_json = {}
    new_json['annotations'] = []
    for x in data['annotations']:
        if x['image_id'] in imgs_id:
            new_json['annotations'].append(x)

    with open(os.path.join(save_features_path, 'train_captions_to_weigth.json'), 'wt') as f:
        json.dump(new_json, f)


def main():
    print('Fetching numpy data.')
    aux_dict = get_data_npy()
    print('Creating ids list.')
    imgs_id = list(aux_dict.keys())
    if data_format == 'coco':
        print('Creating coco training json.')
        create_coco_train_json(imgs_id)
    else:
        print('Creating flickr training json.')
        create_flickr_json(imgs_id)


if __name__ == '__main__':
    main()
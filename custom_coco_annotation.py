import json
import os
from utils import create_flickr_dict
from params import flickr_captions, annotation_file, flickr_dev

with open(annotation_file, 'rt') as f:
    data = json.load(f)

def create_coco_validation_set():
    new_json = {}
    new_json['annotations'] = []

    new_json['annotations'].append({'image_id': '1', 'id': '1', 'caption': 'my first caption'})
    new_json['annotations'].append({'image_id': '2', 'id': '2', 'caption': 'my second caption'})

    with open('./test_real_captions.json', 'wt') as f:
        json.dump(new_json, f)


    new_json = {}
    new_json['annotations'] = []

    new_json['annotations'].append({'image_id': '1', 'id': '1', 'caption': 'my first other caption'})
    new_json['annotations'].append({'image_id': '2', 'id': '2', 'caption': 'my second other caption'})

    with open('./test_val_captions.json', 'wt') as f:
        json.dump(new_json, f)


def create_flickr_validation_set():
    images_dict = create_flickr_dict(flickr_captions)
    total = len(images_dict)
    # lets give fixed values to mandatory fields
    license_ = str(3)
    url_ = 'asdasdsda.com'
    width_ = str(640)
    height_ = str(480)
    date_captured = str(14)

    out_json_tr = []
    captions_tr = []
    ims = []
    anns = []
    #captions_en = []
    # current_converted = int(total_images / 5)
    offset = 0
    found = 0
    id_ = 0
    index_ = 0

    true_json = {}
    true_json['images'] = []
    true_json['annotations'] = []
    true_json['type'] = 'captions'
    num_processed = 0

    for img, captions in images_dict.items():
        true_json["images"].append({'license': license_,
                         "url": url_,
                         "file_name": 'my_file.jpg',
                         "id": img,
                         "width": width_,
                         "date_captured": date_captured,
                         "height": height_})

        for caption in captions:
            true_json['annotations'].append({'image_id': img,
                                  "id": id_,
                                  "caption": caption})
            id_ += 1

        num_processed += 1
        print('Todo: ', total-num_processed)

    with open('./flickr_captions_cocoapi.json', 'wt') as f:
        json.dump(true_json, f)


def generate_flickr_val():
    # images_dict = create_flickr_dict(flickr_captions)

    with open(flickr_dev, 'rt') as f:
        imgs = [x.strip() for x in f.readlines()]

    with open('./flickr_captions_cocoapi.json', 'rt') as f:
        data = json.load(f)

    val_json = []
    for img in imgs:
        try:
            # captions = images_dict[img]
            captions = [x for x in data['annotations'] if x['image_id'] == img]

            for caption in captions:
                val_json.append({'image_id': caption['image_id'],
                                 "id": caption['id'],
                                 "caption": caption['caption']})
        except Exception as e:
            print(e)
            continue

    with open('./flickr_captions_val.json', 'wt') as f:
        json.dump(val_json, f)







    # d = [{'image_id': '1',"id": '1',"caption": 'my other first caption'}]
    #
    # with open('./test_val_captions.json', 'wt') as f:
    #     json.dump(d, f)
import json

def create_flickr_dict(flickr_captions):
    images_dict = {}
    with open(flickr_captions, 'rt') as f:
        for line in f.readlines():
            line = line.strip()
            img_name, img_cap = line.split('\t')
            img_name = img_name.split('.jpg')[0] + '.jpg'
            if img_name not in images_dict.keys():
                images_dict[img_name] = []
            images_dict[img_name].append(img_cap)

    return images_dict


def load_weights(weights_path):
    with open(weights_path, 'rt') as f:
        data = json.load(f)
    return data


def create_coco_dict(annotation_file):
    with open(annotation_file, 'rt') as f:
        data = json.load(f)

    images_dict = {}
    for annot in data['annotations']:
        # break
        if annot['image_id'] not in images_dict.keys():
            images_dict[annot['image_id']] = []
        images_dict[annot['image_id']].append(annot['caption'])

    return images_dict


# data = load_weights('./tag2score_list_2.json')

################

#
# import os
# from params import flickr_training, flickr_test
#
# with open(flickr_training, 'rt') as f:
#     train = [os.path.join('E:\\User\\freelancer\\image_cap\\features\\flickr\\vgg', line.strip()+'.npy') for line in f.readlines()]
#
# with open(flickr_test, 'rt') as f:
#     test = [os.path.join('E:\\User\\freelancer\\image_cap\\features\\flickr\\vgg', line.strip()+'.npy') for line in f.readlines()]
#
#
# my_train = train[:160]
# my_test = test[:40]
#
# my_path = 'E:\\User\\freelancer\\image_cap\\features\\flickr\\test_flickr'
#
# import shutil
# for lst in [my_test, my_train]:
#     for img in lst:
#         shutil.copy(img, my_path)



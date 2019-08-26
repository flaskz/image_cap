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

################


import os
from params import flickr_training, flickr_test

with open(flickr_training, 'rt') as f:
    train = [os.path.join('E:\\User\\freelancer\\image_cap\\features\\flickr\\vgg', line.strip()+'.npy') for line in f.readlines()]

with open(flickr_test, 'rt') as f:
    test = [os.path.join('E:\\User\\freelancer\\image_cap\\features\\flickr\\vgg', line.strip()+'.npy') for line in f.readlines()]


my_train = train[:160]
my_test = test[:40]

my_path = 'E:\\User\\freelancer\\image_cap\\features\\flickr\\test_flickr'

import shutil
for lst in [my_test, my_train]:
    for img in lst:
        shutil.copy(img, my_path)



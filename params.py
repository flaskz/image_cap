import os

# using softmax layer, the same from neuraltalk
# use to generate weights from dict dataset
generate_dict_dataset = True

# coco data only
full_coco_dataset = False
num_batches = 100

# parameters for training
TEST_SIZE = 0.2

# batch size must be multiple of train
BATCH_SIZE = 8
EPOCHS = 10

# cnn architecture (vgg/inception)
# if vgg = False then use inception
vgg = False

# which dataset to use (coco/flickr)
data_format = 'flickr'

# BEAM SEARCH PARAMS
num_k_beam = 5
alpha = 0.7

# path for saved features
if generate_dict_dataset:
    save_features_path = 'E:\\User\\freelancer\\features\\include_top\\{}\\{}\\'.format(data_format, 'vgg' if vgg else 'inception')
else:
    save_features_path = 'E:\\User\\freelancer\\features\\{}\\{}\\'.format(data_format, 'vgg' if vgg else 'inception')

os.makedirs(save_features_path, exist_ok=True)

weights_dir = './weights_json'
os.makedirs(weights_dir, exist_ok=True)

# type of weight (pos/euclidean)
weight_type = 'pos'
# path to weight file
weight_path = os.path.join(weights_dir, 'tag2score_pos_weight.json') if weight_type == 'pos' else os.path.join(weights_dir, 'tag2score_list_2.json')

# path to coco train annotation file
annotation_file = 'E:\\User\\freelancer\\image_captioning\\annotations\\captions_train2014.json'

# path to flickr captions
flickr_captions = 'E:\\User\\freelancer\\image_cap\\Flickr8k\\Flickr8k.token.txt'

# path to flickr train/test/dev images txt
flickr_training = 'E:\\User\\freelancer\\image_cap\\Flickr8k\\Flickr_8k.trainImages.txt'
flickr_test = 'E:\\User\\freelancer\\image_cap\\Flickr8k\\Flickr_8k.testImages.txt'
flickr_dev = 'E:\\User\\freelancer\\image_cap\\Flickr8k\\Flickr_8k.devImages.txt'

# which data to use (coco/flickr)
if data_format == 'coco':
    # path to coco images
    PATH = 'E:\\User\\freelancer\\datasets\\train2014\\'
else:
    # path to flickr images
    PATH = 'E:\\User\\freelancer\\datasets\\Flickr8k\\'
    # PATH = 'E:\\User\\freelancer\\datasets\\Flickr8k\\test_flickr'

# directory to save checkpoint
# checkpoint_save_path = ".\\checkpoints\\test_flickr_train"
if data_format == 'coco':
    checkpoint_save_path = ".\\checkpoints\\test_coco_train_500"
else:
    checkpoint_save_path = ".\\checkpoints\\test_flickr_train_weights"

os.makedirs(checkpoint_save_path, exist_ok=True)


################################## VALIDATION ##################################

# directory with the saved model for validation
# checkpoint_load_path = checkpoint_save_path
if data_format == 'coco':
    checkpoint_load_path = ".\\checkpoints\\train"
    # checkpoint_load_path = ".\\checkpoints\\test_coco_train"
    # checkpoint_load_path = checkpoint_save_path
else:
    checkpoint_load_path = ".\\checkpoints\\train"
    # checkpoint_load_path = ".\\checkpoints\\test_flickr_train"
    # checkpoint_load_path = checkpoint_save_path

# use trained model on single image
# single_image_val = os.path.join(PATH, 'COCO_train2014_000000093496.jpg')
validate_batch = False

# single_image_val = 'E:\\User\\freelancer\\datasets\\Flickr8k\\1579798212_d30844b4c5.jpg']
# single_image_val = 'E:\\User\\Imagem\\formacao_1600x1200-um-milhao-de-amigos.jpg'
single_image_val = 'E:\\User\\Imagem\\foto1.jpg'


validation_dir = 'E:\\User\\freelancer\\datasets\\train2014'


human_caption = 'a group of friends meeting at a cafe.'


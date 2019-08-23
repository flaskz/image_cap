import os

# using softmax layer, the same from neuraltalk

# coco data only
num_batches = 20

# parameters for training
TEST_SIZE = 0.2
BATCH_SIZE = 16
EPOCHS = 15

# cnn architecture (vgg/inception)
# if vgg = False then use inception
vgg = True

# which dataset to use (coco/flickr)
data_format = 'flickr'

# path for saved features
save_features_path = './features/{}/{}/'.format(data_format, 'vgg' if vgg else 'inception')
os.makedirs(save_features_path, exist_ok=True)

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

# directory to save checkpoint
checkpoint_save_path = ".\\checkpoints\\test_train"
os.makedirs(checkpoint_save_path, exist_ok=True)


################################## VALIDATION ##################################

# directory with the saved model for validation
checkpoint_load_path = ".\\checkpoints\\train"

# use trained model on single image
single_image_val = os.path.join(PATH, 'COCO_train2014_000000093496.jpg')



import os
from faceDB.face_db import FaceDB
from faceDB.util import *   # only required for saving cluster images
import random

def load_imgs(img_directory):        
    imgs = []
    for root, subdirs, files in os.walk(img_directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                 imgs.append(os.path.join(root, file))
    
    return imgs

model_dir= '/usr/src/app/deps/openface/models'

faceDB = FaceDB(open_face_model_dir=model_dir,num_clusters=10)

# Part 1: Add negative images
# For the svm-merge step to be useful at all, we need a bunch of negative
# examples. I guess this can be just faces selected randomly from our database
# etc. Here, I just pass in game of throne faces as negative examples.
negative_imgs = load_imgs('./data/lfw')
random.seed(1234)
negative_imgs = random.sample(negative_imgs, 750)

# This method does not do any face detection. It expects each to be an image
# path - where the image is of a cropped face. (does not have to be aligned)
faceDB.add_negative_features(negative_imgs)

# Part 2: Adding images that we want to actually cluster.
# Add imgs as either:
#   a. files of detected faces (faceDB.add_detected_faces)
#   b. features: (faceDB.add_features)
#   c. frames:  (faceDB.add_frames)
# All of these return: [ids], clusters = {} 

imgs  = load_imgs('./data/lfw')

# I will simulate icrementally adding multiple videos by treating each 250
# image chunk in imgs as a new video.

clusters = None
for i in range(1):
    imgs_to_add = imgs[i:i+250]
    faceDB.add_detected_faces('test_vid', imgs_to_add,clusters)

if clusters is not None:
    print('final len of clusters = ', len(clusters))

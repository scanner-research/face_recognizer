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

model_dir= '/Users/parimarjann/openface/models'

# Not using this anymore, but I could set it up so that it pickles features
# based on db_name? 
db_name = 'does_not_matter'

# Currently just using Agglomerative clustering - but should ideally be doing
# multiple clustering algorithms and finding the best one based on its results.
cluster_algs = ['does_not_matter']

faceDB = FaceDB(open_face_model_dir=model_dir, db_name=db_name,
            num_clusters=10, cluster_algs=cluster_algs,verbose=False)   

# Part 1: Add negative images
# For the svm-merge step to be useful at all, we need a bunch of negative
# examples. I guess this can be just faces selected randomly from our database
# etc. Here, I just pass in game of throne faces as negative examples.
negative_imgs = load_imgs('./data/got/got1_faces')
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

imgs  = load_imgs('./data/friends/friends1_faces')

# I will simulate icrementally adding multiple videos by treating each 250
# image chunk in imgs as a new video.

clusters = None
for i in range(0,len(imgs), 250):
    imgs_to_add = imgs[i:i+250]
    ids, clusters = faceDB.add_detected_faces('test_vid', imgs_to_add,
            clusters)
    for id in ids:
        assert id in clusters, 'has to be one of the keys'

print('final len of clusters = ', len(clusters))
print('saving cluster images')
total_faces = 0
for _, cluster in clusters.iteritems():
    total_faces += len(cluster.faces)
    save_cluster_image(cluster.faces, 'test29')

print('total faces are: ', total_faces)
print('imgs were ', len(imgs))

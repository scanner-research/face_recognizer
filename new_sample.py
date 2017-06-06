import os
from faceDB.face_db import FaceDB
from faceDB.util import *   # only required for saving cluster images
import random
import csv

add_method = 'features' # or "faces"

model_dir= '/Users/parimarjann/openface/models' # not required for features
faceDB = FaceDB(open_face_model_dir=model_dir,num_clusters=10, cuda=False)

# Part 1: Add negative images
# For the svm-merge step to be useful at all, we need a bunch of negative
# examples. I guess this can be just faces selected randomly from our database
# / or from an external source.

# Note: Just for illustrating it here I just add the same negative features as
# the positive features. 

if add_method == 'faces':
    negative_imgs = load_imgs('./data/lfw')
    random.seed(1234)
    negative_imgs = random.sample(negative_imgs, 750)
    # It expects each to be an image
    # path - where the image is of a cropped face. (does not have to be aligned)
    faceDB.add_negative_faces(negative_imgs)

elif add_method == 'features':
    with open('data/mini_extracted_features.csv', 'rb') as f:
        reader = csv.reader(f)
        extracted_features = list(reader)[1:1000]
    
    # extract just the relevant columns
    extracted_features = [f[7:] for f in extracted_features]
    for i, f in enumerate(extracted_features):
        extracted_features[i] = [float(j) for j in f]

    assert len(extracted_features[0]) == 128, 'test'
    faceDB.add_negative_features(extracted_features)
else:
    assert False, 'not supported add method'


# Part 2: Adding images that we want to actually cluster.
# Add imgs as either:
#   a. files of detected faces (faceDB.add_detected_faces)
#   b. features: (faceDB.add_features)

if add_method == 'faces':
    imgs = load_imgs('./data/lfw')
    # I will simulate icrementally adding multiple videos by treating each 250
    # image chunk in imgs as a new video.
    clusters = None
    for i in range(2):
        imgs_to_add = imgs[i*250:i*250+250]
        # Temporary solution. Actually should be the correct frame numbers for each
        # face.
        frame_numbers = range(i*250, i*250+250, 1)         
        print(frame_numbers)
        (ids, clusters, faces), indices = faceDB.add_detected_faces('test_vid', imgs_to_add, frame_numbers, clusters)

    if clusters is not None:
        print('final len of clusters = ', len(clusters))

elif add_method == 'features':
    clusters = None
    for i in range(3):
        features_to_add = np.array(extracted_features[i*250:i*250+250])
        (ids, clusters, faces) = faceDB.add_features('test_vid', features_to_add, clusters) 

    if clusters is not None:
        print('final len of clusters = ', len(clusters))

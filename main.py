import numpy as np
import matplotlib.pyplot as plt
import cv2
import time

import caffe

import os
import pickle
import hashlib

from tsne import *
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN

from timeit import default_timer as now

import copy

# Note: If want to turn caffe output on/off then toggle GLOG_minloglevel
# environment variable.

# FIXME: Turn this into argparse stuff

# Arbitrarily chosen - because it seemed to be identifying people correctly
THRESHOLD = 0.00040

FC7_SIZE = 4096
FC8_SIZE = 2622

CLUSTERS = 4
PICKLE = True
TSNE_PICKLE = True
INPUT_SIZE = 224
BATCH_SIZE = 50
CENTER = True

# Use opencv to display images in same cluster (on local comp)
DISP_IMGS = False

DATA_DIR = 'data'
# Change this appropriately
IMG_DIRECTORY = os.path.join(DATA_DIR, 'twilight1_imgs/')

# caffe files
model = 'nets/VGG_FACE_deploy.prototxt';
weights = 'nets/VGG_FACE.caffemodel';

def load_names():
    # Let's set up the names to check if we are right or wrong
    f = open('./data/names.txt')
    names = f.read()
    names = names.split('\n')
    f.close()
    return names

def load_img_files():

    file_names = os.listdir(IMG_DIRECTORY)
    imgs = []

    for i, name in enumerate(file_names):	

        # this was just a tmp folder I was using for processed images
        if 'proc' in name:
            continue
        imgs.append(IMG_DIRECTORY + name)  	

    return imgs

def get_features(imgs, names):
    '''
    Checks pickle for precomputed data, otherwise runs the caffe stuff on the
    all images in imgs.
    '''
    
    imgs.sort()
    pickle_name = gen_pickle_name(imgs, 'fc7')

    features, preds = do_pickle(PICKLE, pickle_name, 2, run_caffe, imgs, names)
    sanity_check_features(features)

    return features, preds
    
def centralize(img):
    '''
    img is an np array - width, height, 3.
    centers and crops it - then returns img.

    Based on faceCrop.m from the vgg matconvnet code

    FIXME: should compare outputs with matlab script for correctness...
    '''
    # Because we already have a cropped face - will deal with it as if the
    # whole image is the bounding box

    extend = 0.1
    x1, y1 = 0, 0
    x2, y2, _ = img.shape
    width = round(x2-x1)
    height = round(y2-y1)

    length = (width + height) / 2
    centrepoint = [round(x1) + width/2, round(y1) + height/2]
    x1 = centrepoint[0] - round(1+extend)*(length/2)
    y1 = centrepoint[1] - round(1+extend)*(length/2)
    x2 = centrepoint[0] + round(1+extend)*(length/2)
    y2 = centrepoint[1] + round(1+extend)*(length/2)

    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(min(x2, img.shape[0]))
    y2 = int(min(y2, img.shape[1]))

    img = img[x1:x2, y1:y2,:]

    return img

def run_caffe(img_files, names):
    '''
    Main loop in which we use caffe to recognize / score each of the images
    ''' 

    caffe.set_mode_gpu()

    net = caffe.Net(model, weights, caffe.TEST)
    
    features = {}
    features['fc7'] = []
    features['fc8'] = []
    features['fc6'] = []

    recognized = 0
    preds = []

    # using batches because otherwise running into some memory limits...
    imgs = []
    for i, img_file in enumerate(img_files):

        try:
            img = caffe.io.load_image(img_file)
        except IOError:
            continue
        
        # not sure if centralizing it really helps because we aren't cropping
        # out a bounding box from a bigger image as in the orig paper
        if CENTER:
            img = centralize(img)
            
        img = caffe.io.resize_image(img, (224,224), interp_order=3)
        assert img.shape == (224, 224, 3), 'img shape is not 224x224'
        imgs.append(img)
     
        if i != 0 and i % BATCH_SIZE == 0:

            # Let's run caffe on this batch
            net.blobs['data'].reshape(*(len(imgs), 3, INPUT_SIZE, INPUT_SIZE))
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2, 0, 1))
            transformer.set_channel_swap('data', (2, 1, 0))        

            data = np.asarray([transformer.preprocess('data', img) for img in imgs])

            final_output = net.forward_all(data=data)
            probs = final_output['prob']
            for prob in probs:
                assert np.linalg.norm(prob) != 0, 'prob = 0' 
                guess = int(prob.argmax())
                conf = prob[guess]
                # print('name was ', names[guess], 'confidence is ', conf)
                name = names[guess]
                if conf > THRESHOLD:
                    recognized += 1
                    name += '++'
                else:
                    name += '--'

                preds.append(name)
            
            for layer in features:
                output = net.blobs[layer].data
                for j, row in enumerate(output):
                    assert np.linalg.norm(row) != 0, 'output = 0'
                    features[layer].append(np.copy(row))
 
            # reset imgs for the next batch
            imgs = []

    print('recognized = ', recognized)

    sanity_check_features(features)
    for layer in features:
        features[layer] = np.array(features[layer])

    return features, preds

#FIXME: Combine these two functions
def gen_pickle_name(imgs, feature_layer):
    """
    Use hash of file names + which layer data we're storing. 
    """
    hashed_input = hashlib.sha1(str(imgs)).hexdigest()
    
    name = hashed_input + '_' + feature_layer

    directory = "./pickle/"

    return directory + name + ".pickle"

def tsne_gen_pickle_name(features):
    """
    Use hash of file names + which layer data we're storing. 
    """
    hashed_input = hashlib.sha1(str(features)).hexdigest()
    
    name = hashed_input + '_' + 'tsne' 

    directory = "./pickle/"

    return directory + name + ".pickle"

def kmeans_clustering(all_feature_vectors, preds, names, imgs):
    '''
    runs kmeans with mostly default values on all_feature_vectors - and then
    prints out the names <--> labels combinations in the end.

    Ideally, can then manually check if the faces clustered in the same label
    belong to the same person or not.
    '''

    # kmeans = KMeans(n_clusters=CLUSTERS, random_state=0).fit(all_feature_vectors) 

    # Gives too many clusters:
    # kmeans = AffinityPropagation(damping=0.50).fit(all_feature_vectors) 
        
    # 
    kmeans = DBSCAN().fit(all_feature_vectors) 
    
    # Visualizing the labels_ - this is comman to all the clustering
    # algorithms..
    label_names = {}
    for i, label in enumerate(kmeans.labels_):

        predicted_name = preds[i]
        file_name = imgs[i]

        label = str(label)
        if label not in label_names:
            label_names[label] = []

        label_names[label].append((predicted_name, file_name))

    for l in label_names:
        print('label is ', l)
        print(label_names[l])

        # Let's use opencv to display imgs one by one here in this cluster
        if DISP_IMGS:
            wait = raw_input("press anything to start this label")
            for label in label_names[l]:
                
                file_name = label[1]
                # open the image with opencv
                img = cv2.imread(file_name)
                cv2.imshow('ImageWindow', img)
                c = cv2.waitKey()
                if c == 'q':
                    break

                # wait for a keypress to go to next image

def run_tsne(all_feature_vectors, preds, names, imgs):
    '''
    '''
    assert len(all_feature_vectors) == len(preds), 'features and preds \
            should be same length'

    # Since we don't have correct labels - maybe we should just plot it without
    # labels - will just be the same color.

    pickle_name = tsne_gen_pickle_name(all_feature_vectors)
    
    Y = do_pickle(TSNE_PICKLE, pickle_name, 1, tsne, all_feature_vectors)

    # See if this is much different than direct kmeans
    # kmeans_clustering(Y, preds, names, imgs)

    #FIXME: Better way to visualize this? 

    # this won't work on the halfmoon cluster so run it on a local machine
    #size = 20
    # Plot.scatter(Y[:,0], Y[:,1], size);
    # Plot.show();
    
def do_pickle(pickle_bool, pickle_name, num_args, func, *args):
    '''
    General function to handle pickling.
    @func: call this guy to get the result if pickle file not available.

    '''
    if not pickle_bool:
        rets = func(*args)   
    elif os.path.isfile(pickle_name):
        #pickle exists!
        with open(pickle_name, 'rb') as handle:
            rets = []
            for i in range(num_args):
                rets.append(pickle.load(handle))

            print("successfully loaded pickle file!")    
            rets = tuple(rets)
            handle.close()

    else:
        rets = func(*args)
        
        # dump it for future
        with open(pickle_name, 'w+') as handle:
            for i in range(len(rets)):
                pickle.dump(rets[i], handle, protocol=pickle.HIGHEST_PROTOCOL) 
        handle.close()

    return rets

def sanity_check_features(features):

    for layer in features:
        for i, row in enumerate(features[layer]):
            f1 = np.linalg.norm(row)
            assert f1 != 0, ':((('

def main():

    names = load_names()
    imgs = load_img_files()
    
    features, preds = get_features(imgs, names)

    layer = features['fc8']

    kmeans_clustering(layer, preds, names, imgs)

    # do tsne clustering now.
    run_tsne(layer, preds, names, imgs)

    # FIXME: Add way to output clusters to csv file for better/easier checking

if __name__ == '__main__':

    main()

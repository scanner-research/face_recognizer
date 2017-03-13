import numpy as np
import matplotlib.pyplot as plt
import cv2

#import caffe

import os
import pickle
import hashlib

from tsne import *
from sklearn.cluster import KMeans

# Note: If want to turn caffe output on/off then toggle GLOG_minloglevel
# environment variable.

# Arbitrarily chosen - because it seemed to be identifying people correctly
# when the output was over this threshold, and wrong otherwise. Not sure why
# the values are so low - might have something to do with why it does not work
# in gpu mode?

THRESHOLD = 0.00040
FEATURE_LAYER = 'fc8'
CLUSTERS = 32
PICKLE = True

# Use opencv to display images in same cluster (on local comp)
DISP_IMGS = False

# Change this appropriately
IMG_DIRECTORY = './final_girl_imgs/'

# caffe files
model = 'VGG_FACE_deploy.prototxt';
weights = 'VGG_FACE.caffemodel';

def load_names():
    # Let's set up the names to check if we are right or wrong
    f = open('./names.txt')
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

def test_imgs(imgs, names):
    '''
    Checks pickle for precomputed data, otherwise runs the caffe stuff on the
    all images in imgs.
    '''
    
    imgs.sort()

    pickle_name = gen_pickle_name(imgs)
    print("pickle name is ", pickle_name)

    if os.path.isfile(pickle_name) and PICKLE:

        with open(pickle_name, 'rb') as handle:
            all_feature_vectors = pickle.load(handle)
            preds = pickle.load(handle)
            print("successfully loaded pickle file!")    

        handle.close()

        return all_feature_vectors, preds

    else:
        
        all_feature_vectors, preds = run_caffe(imgs, names)

        if PICKLE:
            with open(pickle_name, 'w+') as handle:
                pickle.dump(all_feature_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
            handle.close()

        return all_feature_vectors, preds
    

def run_caffe(imgs, names):
    '''
    Main loop in which we use caffe to recognize / score each of the images
    ''' 

    caffe.set_mode_cpu()

    net = caffe.Net(model, weights, caffe.TEST); 

    transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))

    all_feature_vectors = []
    recognized = 0
    # List with name of recognized celebrity or 'unknown'
    preds = []

    for img_file in imgs:

        try:
            img = caffe.io.load_image(img_file)
        except IOError:
            print("caught an IO error for file ", img_file)
            continue

        # Can resize it here / or just run a separate preprocessing script that
        # resizes it.
        img = caffe.io.resize_image(img, (224,224), interp_order=3)

        assert img.shape == (224, 224, 3), 'img shape is not 224x224'

        # Check this step - might want to do more pre-processing etc? 
        # Also maybe something going wrong in the GPU model here.
        img_data = transformer.preprocess('data', img)
        net.blobs['data'].data[...] = img_data

        output = net.forward()

        guess = int(output['prob'].argmax())
        conf = output['prob'][0, guess]
        
        print('name was ', names[guess], 'confidence is ', conf)

        # Get feature activations
        feature_vector = net.blobs[FEATURE_LAYER].data
        
        # Need to make np.copy because list assignment is by ref, and
        # net.blobs[FEATURE_LAYER].data will change in next it
        all_feature_vectors.append(np.copy(feature_vector[0]))

        if conf >= THRESHOLD:
            # recognized someone
            # add file_name to list of recogs etc
            recognized += 1
            name = names[guess] + '++'
        else:
            name = names[guess] + '--'

        preds.append(name)


    print('recognized users = ', recognized, 'from total = ', len(imgs))
    # Use some sort of clustering on the final layer

    all_feature_vectors = np.array(all_feature_vectors)
    
    return all_feature_vectors, preds

#FIXME: Combine these two functions
def gen_pickle_name(imgs):
    """
    Use hash of file names + which layer data we're storing. 
    """
    hashed_input = hashlib.sha1(str(imgs)).hexdigest()
    
    name = hashed_input + '_' + FEATURE_LAYER

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

    kmeans = KMeans(n_clusters=CLUSTERS, random_state=0).fit(all_feature_vectors) 

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
    print("features = ", len(all_feature_vectors))
    print("preds = ", len(preds))

    # FIXME: extra imgs with final_girl?
    # print("imgs = ", len(imgs))

    assert len(all_feature_vectors) == len(preds), 'features and preds \
            should be same length'

    # Since we don't have correct labels - maybe we should just plot it without
    # labels - will just be the same color.

    # labels = range(len(imgs))
    # labels = np.array(labels)
    
    # In Y, every feature is reduced to an x,y point.

    pickle_name = tsne_gen_pickle_name(all_feature_vectors)

    if os.path.isfile(pickle_name) and TSNE_PICKLE:

        with open(pickle_name, 'rb') as handle:
            Y = pickle.load(handle)
            print("successfully loaded pickle file!")    

        handle.close()

        return all_feature_vectors, preds

    else:
        
        Y = tsne(all_feature_vectors)

        if TSNE_PICKLE:
            with open(pickle_name, 'w+') as handle:
                pickle.dump(all_feature_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(preds, handle, protocol=pickle.HIGHEST_PROTOCOL)
             
            handle.close()

    # See if this is much different than direct kmeans
    #kmeans_clustering(Y, preds, names, imgs)

    #FIXME: Better way to visualize this? 

    size = 20
    Plot.scatter(Y[:,0], Y[:,1], size);
    Plot.show();
    
def main():

    names = load_names()
    imgs = load_img_files()
    
    all_feature_vectors, preds = test_imgs(imgs, names)

    # sanity check
    for i,f in enumerate(all_feature_vectors):
        norm = np.linalg.norm(f)
        # assert norm != 0, 'norm of features is 0'
        if norm == 0:
           print i
           print f
    
    #kmeans_clustering(all_feature_vectors, preds, names, imgs)

    # do tsne clustering now.
    run_tsne(all_feature_vectors, preds, names, imgs)

if __name__ == '__main__':

    main()

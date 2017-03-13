import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import os
import pickle

from sklearn.cluster import KMeans

# Arbitrarily chosen - because it seemed to be identifying people correctly
# when the output was over this threshold, and wrong otherwise. Not sure why
# the values are so low - might have something to do with why it does not work
# in gpu mode?
THRESHOLD = 0.00040
FEATURE_LAYER = 'fc8'
CLUSTERS = 32

# Let's set up the names to check if we are right or wrong
f = open('./names.txt')
names = f.read()
names = names.split('\n')
f.close()

IMG_DIRECTORY = './final_girl_imgs/'

file_names = os.listdir(IMG_DIRECTORY)
imgs = []

for i, name in enumerate(file_names):	
    imgs.append(IMG_DIRECTORY + name)  	


# for testing cpu seems just fine
caffe.set_mode_cpu()
# caffe.set_mode_gpu()


model = 'VGG_FACE_deploy.prototxt';
weights = 'VGG_FACE.caffemodel';
net = caffe.Net(model, weights, caffe.TEST); 

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))

all_feature_vectors = []
recognized = 0

# List with name of recognized celebrity or 'unknown'
preds = []

# So the names are in order of same guys together (more or less..)
imgs.sort(reverse=True)

for img_file in imgs:
    img = caffe.io.load_image(img_file)

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
        preds.append(names[guess])
    else:
        preds.append('Unknown')


print('recognized users = ', recognized, 'from total = ', len(imgs))
# Use some sort of clustering on the final layer

all_feature_vectors = np.array(all_feature_vectors)

# Sanity check - not required anymoe
# for i, f in enumerate(all_feature_vectors):
    # print('norm of ', str(i),  'is ', np.linalg.norm(f))

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
    print("label is ", l)
    print(label_names[l])


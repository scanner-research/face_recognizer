import numpy as np
import matplotlib.pyplot as plt
import cv2
import caffe
import os

from sklearn.cluster import KMeans

def is_in_names(names, file_name):
    """	
    """
    for name in names:
        if name in file_name:
            return True

    return False


# Let's set up the names to check if we are right or wrong
f = open('./names.txt')
names = f.read()
names = names.split("\n")
f.close()

#IMG_DIRECTORY = "./imgs/"
IMG_DIRECTORY = "./imgs/"
file_names = os.listdir(IMG_DIRECTORY)
imgs = []

for i, name in enumerate(file_names):
	
#	if i > 32:
#		continue
    imgs.append(IMG_DIRECTORY + name)  	


# for testing cpu seems just fine
caffe.set_mode_cpu()
#caffe.set_mode_gpu()


model = 'VGG_FACE_deploy.prototxt';
weights = 'VGG_FACE.caffemodel';
net = caffe.Net(model, weights, caffe.TEST); 

transformer = caffe.io.Transformer({'data':net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_channel_swap('data', (2,1,0))

#print(net.inputs) -- ['data']

correct = 0
final_layers = []

# So the names are in order of same guys together (more or less..)
imgs.sort(reverse=True)

for img_file in imgs:
    img = caffe.io.load_image(img_file)
    print "shape of img is ", img.shape	
    #img = caffe.io.resize_image(img, (224,224), interp_order=3)
    #print "shape of resized img is ", img.shape

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    #print "shape of resized img is ", img.shape

    output = net.forward()
    print "img file is ", img_file

    guess = int(output['prob'].argmax())

    #print("guess is ", guess)
    
    #print("output['prob'].shape = ", output['prob'].shape)
    #print("o['prob'][0].shape", output['prob'][0].shape)

    for k in output:
            print k

    conf = output['prob'][0, guess]
    print(output['prob'][0])
    
    print("name was ", names[guess], "confidence is ", conf)

    # Get middle activations
    fc7 = net.blobs['fc7'].data
    final_layers.append(fc7)

    #print("shape of fc7 is ", fc7.shape)
    
    if is_in_names(names, img_file):

            if names[guess] in img_file and conf >= 0.00040:
                    correct += 1
    
    else:	
            if conf < 0.00040:
                    print("got a negative sample right!!!!")
                    correct += 1

print("% of correct is ", float(correct) / len(imgs))

# Use some sort of clustering on the final layer

"""
final_layers = np.array(final_layers)
print(final_layers[0].shape)
dataset_size = len(final_layers)
final_layers = final_layers.reshape(dataset_size,-1)
print(final_layers[0].shape)

kmeans = KMeans(n_clusters=8, random_state=0).fit(final_layers) 
print(len(kmeans.labels_))
print(kmeans.labels_)
print(kmeans.clusters_)
"""


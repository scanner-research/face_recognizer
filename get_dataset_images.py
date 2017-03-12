import os
import urllib
import cv2
import numpy as np
import time

DATA_DIR = "./dataset/"

files = os.listdir(DATA_DIR)
 
def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
        try:
            resp = urllib.urlopen(url)

            # if resp.getcode() != 200:
                # return None
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
        except:
            return None
	# return the image
	return image

def subtract_mean(image):
    """
    """
    pass


for i, file in enumerate(files):

    file = DATA_DIR + file
    name = file.replace(".txt", "")
    name = name.replace("./dataset/", "")

    print("name is ", name)
    
    f = open(file, "r")
    images_info = f.read().split("\n")
    print(len(images_info))

    if i > 10:
        break

    for j, info in enumerate(images_info):

        # print info 
        info = info.split()
        url = info[1]
        bbox = info[2:6]

        for i in range(len(bbox)):
            bbox[i] = int(float(bbox[i]))

        print "downloading %s" % (url)
        image = url_to_image(url)
        if image is None:
            continue

        cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        print("writing out to file")

        file_name = "./dataset_images/" + name + "_" + str(j) + ".jpg"
        
        cropped = cv2.resize(cropped, (224,224))
        cv2.imwrite(file_name, cropped)
        print("written!")

        # use bounding box on the image:


        if j > 10:
            break 

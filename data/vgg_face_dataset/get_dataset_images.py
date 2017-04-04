import os
import urllib
import cv2
import numpy as np
import time

DATA_DIR = "./files/"

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

urls_file = open('tried_urls.txt', 'r')
tried_urls = urls_file.read()
tried_urls = tried_urls.split('\n')
urls_file.close()

urls_file = open('tried_urls.txt', 'a')

for i, file in enumerate(files):
    
    # have already downloaded this too many times
    if i in [0, 1, 2, 3, 4]:
        continue

    file = DATA_DIR + file
    name = file.replace(".txt", "")
    name = name.replace("./files/", "")

    print("name is ", name)
    
    f = open(file, "r")
    images_info = f.read().split("\n")
    print(len(images_info))
    try:
        for j, info in enumerate(images_info):
 
            if j > 100:
                continue

            # print info 
            info = info.split()
            url = info[1]
            if url in tried_urls:
                continue
            urls_file.write(url + '\n')

            bbox = info[2:6]

            file_name = "./dataset_images/" + name + "_" + str(j) + ".jpg"
            if os.path.exists(file_name):
                print('os.path exists...')
                continue


            for i in range(len(bbox)):
                bbox[i] = int(float(bbox[i]))

            print "downloading %s" % (url)
            image = url_to_image(url)
            if image is None:
                continue

            cropped = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            
            print("writing out to file")


            print('file name is ', file_name) 
            try:
                cropped = cv2.resize(cropped, (224,224))
            except:
                continue

            cv2.imwrite(file_name, cropped)
            print("written!")
    except:
        continue


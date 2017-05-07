import os

from faceDB.face_db import FaceDB
from helper import ArgParser

from sklearn.model_selection import train_test_split
from collections import defaultdict
import random
from faceDB.util import *

def load_img_files(args):
    
    # So we can append name to the end.
    if args.dataset[-1] != '/':
        args.dataset += '/'

    imgs = []
    img_directory = os.path.join('data', args.dataset) 

    i = 0
    # TODO: fix quick test in case of lfw
    for root, subdirs, files in os.walk(img_directory):

        for file in files:
            if i > 200:
                break
            if i > 100 and args.quick_test:
                print('time to break!')
                break
            i += 1
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                 imgs.append(os.path.join(root, file))
    
    print('found images = {}'.format(len(imgs)))
    return imgs

def load_imgs(img_directory):
        
    imgs = []
    for root, subdirs, files in os.walk(img_directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                 imgs.append(os.path.join(root, file))
    
    return imgs

def get_name_from_path(path):
    '''
    This function might have different implementations depending on the format
    of the images/labels we are dealing with. With the vgg dataset, I have just
    named the files by the convention path/to/name_i.jpg - so just extract the
    'name' part here.
    '''
    file_name = path.split('/')[-1]
    name = '' 
    for c in file_name:
        if c.isdigit():
            break
        name += c

    return name

def get_labels(paths):
    '''
    works for vgg, lfw format of files
    '''
    labels = []
    for p in paths:
        labels.append(get_name_from_path(p))
    return labels

def select_imgs(imgs, labels, num_labels, num_faces):
    '''
    select n labels, num_faces from those.
    '''
    ret_imgs = []
    ret_labels = []
    count = defaultdict(int)

    unique_labels = []
    for i, label in enumerate(labels):

        if len(unique_labels) > num_labels:
            continue
        if label not in unique_labels:
            unique_labels.append(label)

        if label in unique_labels:
            if count[label] > num_faces:
                continue
            ret_imgs.append(imgs[i])
            ret_labels.append(labels[i])

            count[label] += 1
              
    return ret_imgs, ret_labels

def main():
    
    args = ArgParser().args 
    model_dir= '/Users/parimarjann/openface/models'
    # For other parameters, defaults are good for now.
    labeled_dataset = False
    videos = True
    
    if labeled_dataset:
        imgs = load_img_files(args)
        labels = get_labels(imgs) 
        faceDB = FaceDB(open_face_model_dir=model_dir, db_name=args.db_name,
                    num_clusters=args.clusters, cluster_algs=args.cluster_algs,
                    svm_merge=args.svm_merge)   
        if args.random_imgs:
            train_imgs, _, train_labels, _ = train_test_split(imgs, labels,
                    train_size=0.1, random_state=1234)
        elif args.select_n_imgs:
            train_imgs, train_labels = select_imgs(imgs, labels, 10, 100)

        faceDB.add_base_faces_from_videos(['test'], [train_imgs],
                labels=[train_labels], cluster=False)
        
        print('num unique faces are ', faceDB.num_unique_faces())
        faceDB.cluster_analysis(faceDB.main_clusters) 
        # faceDB.create_cluster_images()

    elif videos:
        imgs = load_img_files(args)
        # print('len of imgs is ', len(imgs))
        # print('db name is ', args.db_name)

        faceDB = FaceDB(open_face_model_dir=model_dir, db_name=args.db_name,
                    num_clusters=args.clusters, cluster_algs=args.cluster_algs,
                    svm_merge=args.svm_merge,
                    save_bad_clusters=args.save_bad_clusters, verbose=False)   
        train_imgs = imgs
        
        # Original:
        # faceDB.add_base_faces_from_videos([video_name], [train_imgs],
                # labels=None, frame=args.frame)
        # faceDB.label_images()
        # faceDB.cluster_analysis(None) 
        
        # args.dataset = 'got/got2_faces/'
        # new_imgs = load_img_files(args)
        # print('len of new imgs is ', len(new_imgs))
        # faceDB.add_faces_from_video(['got2'],[new_imgs], db_old='got2') 
        
        negative_imgs = load_imgs('./data/got/got1_faces')
        random.seed(1234)
        negative_imgs = random.sample(negative_imgs, 750)
        faceDB.add_negative_features(negative_imgs)

        # testing new interface:
        # faceDB.add_base_faces_from_videos([video_name], [train_imgs],
                # labels=None, frame=args.frame, cluster=False)

        imgs  = load_imgs('./data/friends/friends1_faces')
        # Add 250 imgs at a time.
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
        
if __name__ == '__main__': 

    main()

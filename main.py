import os

from faceDB.face_db import FaceDB
from helper import ArgParser

from sklearn.model_selection import train_test_split

def load_img_files(args):
    
    # So we can append name to the end.
    if args.dataset[-1] != '/':
        args.dataset += '/'
    img_directory = os.path.join('data', args.dataset)
    imgs = []
    
    i = 0

    # TODO: fix quick test in case of lfw
    for root, subdirs, files in os.walk(img_directory):

        for file in files:
            if i > 100 and args.quick_test:
                print('time to break!')
                break
            i += 1
            if os.path.splitext(file)[1].lower() in ('.jpg', '.jpeg'):
                 imgs.append(os.path.join(root, file))
             
    return imgs

    # file_names = os.listdir(img_directory)
    # file_names.sort()

    # for i, name in enumerate(file_names):	
        # if args.quick_test and i > 100:
            # break

        # imgs.append(img_directory + name)  	
    
    # return imgs

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

def main():
    
    args = ArgParser().args 

    imgs = load_img_files(args)
    labels = get_labels(imgs)

    model_dir= '/Users/parimarjann/openface/models'
    # For other parameters, defaults are good for now.
    print(args.cluster_algs)
    exit(0)
    faceDB = FaceDB(open_face_model_dir=model_dir, db_name=args.db_name,
                num_clusters=args.clusters, cluster_algs=args.cluster_algs)  

    # faceDB = FaceDB(open_face_model_dir=model_dir)  

    train_imgs, _, train_labels, _ = train_test_split(imgs, labels,
            train_size=0.5, random_state=1234)

    # train_imgs = imgs    
    # train_labels = labels
    faceDB.add_base_faces_from_videos(['test'], [train_imgs], labels=[train_labels])
    
    print('num unique faces are ', faceDB.num_unique_faces())
    faceDB.cluster_analysis(faceDB.main_clusters) 
    # faceDB.create_cluster_images()

if __name__ == '__main__': 

    main()

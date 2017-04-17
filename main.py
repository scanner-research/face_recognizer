import os

from faceDB.face_db import FaceDB
from helper import ArgParser

def load_img_files(args):
    
    # So we can append name to the end.
    if args.dataset[-1] != '/':
        args.dataset += '/'
    
    img_directory = os.path.join('data', args.dataset)
    file_names = os.listdir(img_directory)

    imgs = []
    file_names.sort()

    for i, name in enumerate(file_names):	
        if args.quick_test and i > 100:
            break

        imgs.append(img_directory + name)  	
    
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

def get_labels_vgg(paths):
    '''
    '''
    labels = []
    for p in paths:
        labels.append(get_name_from_path(p))
    return labels

def main():
    
    args = ArgParser().args 
    model_dir= '/Users/parimarjann/openface/models'
    # For other parameters, defaults are good for now.
    faceDB = FaceDB(open_face_model_dir=model_dir)  
    imgs = load_img_files(args)
    labels = get_labels_vgg(imgs)
    faceDB.add_base_faces_from_videos(['test'], [imgs], labels=[labels])
    
    print('num unique faces are ', faceDB.num_unique_faces())
    
    faceDB.create_cluster_images()

if __name__ == '__main__': 

    main()

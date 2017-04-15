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
        if i > 50:
            break

        imgs.append(img_directory + name)  	
    
    return imgs

def main():
    
    args = ArgParser().args 
    model_dir= '/Users/Parimarjann/openface/models/'

    # For other parameters, defaults are good for now.
    faceDB = FaceDB(open_face_model_dir=model_dir) 
    
    dataset = 'vgg_face_dataset/dataset_images'
    imgs = load_img_files(args)

    faceDB.add_base_faces_from_videos(['test_vgg'], [imgs])
    
    print(faceDB.num_unique_faces())

if __name__ == '__main__': 

    main()

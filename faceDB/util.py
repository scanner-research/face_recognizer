import os
import pickle
import numpy as np
from PIL import Image
import errno

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
            rets = pickle.load(handle)

            print("successfully loaded pickle file!", pickle_name)    
            handle.close()

    else:
        rets = func(*args)
        
        # dump it for future
        with open(pickle_name, 'w+') as handle:
            pickle.dump(rets, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        handle.close()

    return rets

def combine_imgs(imgs, direction):
    '''
    '''
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)

    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    if 'hor' in direction:
        min_shape = (30,30)

    if 'hor' in direction:
        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape, Image.ANTIALIAS) ) for i in imgs ) )
        imgs_comb = Image.fromarray(imgs_comb)
    else: 
        imgs_comb = np.vstack( (np.asarray( i.resize(min_shape, Image.ANTIALIAS) ) for i in imgs ) )
        imgs_comb = Image.fromarray(imgs_comb)

    return imgs_comb


def mix_samples(train_genuine, train_impostors):
    """
    Returns a single np.array with samples, and corresponding labels.
    """ 
    samples = np.vstack((train_genuine, train_impostors))

    labels = []
    # Add labels: 1 - user, and 0 - impostor.
    for i in train_genuine:
        labels.append(1)
    for i in train_impostors:
        labels.append(0)

    labels = np.array(labels)

    return samples, labels

# Taken from http://stackoverflow.com/a/600612/119527
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

import os
import pickle

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

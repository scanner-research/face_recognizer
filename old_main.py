import numpy as np
import pylab as Plot

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
import pickle
import hashlib
import random

from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN, Birch
from sklearn.cluster import SpectralClustering, AgglomerativeClustering,MeanShift

from sklearn.model_selection import train_test_split

from PIL import Image
import math

import face_recognition
import dlib

from helper import *
from sklearn import preprocessing
from collections import defaultdict

from sklearn.manifold import TSNE
from rank_order_cluster import Rank_Order
from util import *

def load_img_files(args):
    
    # So we can append name to the end.
    if args.dataset[-1] != '/':
        args.dataset += '/'
    
    img_directory = os.path.join('data', args.dataset)
    file_names = os.listdir(img_directory)

    imgs = []
    file_names.sort()

    for i, name in enumerate(file_names):	
        imgs.append(img_directory + name)  	
    
    return imgs

def print_clusters(clusters, name):

    label_lens = []
    for l in clusters:
        label_lens.append(len(clusters[l]))
        
    print('************************************')
    print('name of cluster alg is ', name)
    print('len of clusters is ', len(clusters))
    print('smallest label is ', min(label_lens))
    print('biggest label is ', max(label_lens))

def get_features(imgs, args):
    '''
    Checks pickle for precomputed data, otherwise runs the face detector stuff on the
    all images in imgs.
    ''' 
    pickle_name = gen_pickle_name(imgs, args)

    # Add the vgg option here as well.
    if args.face_recognizer:
        func = run_face_recog
    else:
        func = run_open_face

    features, names = do_pickle(args.pickle, pickle_name, 1, func, imgs, args)   
     
    # Let's normalize the features
    # Might also want to try scaling them.
    if args.normalize:
        features = preprocessing.normalize(features, norm='l2')
    
    if args.scale:
        features = preprocessing.scale(features)

    sanity_check_features(features)
    
    return features, names

def run_open_face(img_files, args):
    '''
    Uses the open face package on all the imgs - and returns the feature
    vectors.
    '''
    of = Open_Face_Helper(args) 
    features = []
    faces = []
    
    failed_to_load = 0

    for img_file in img_files:    
        
        # if face has been cropped from before, then do_bb=False.
        try:
            feature_vector, path = of.get_rep(img_file, do_bb=args.do_bb,
            new_dir='data/bb_dataset')
        except Exception:
            failed_to_load += 1
            continue

        features.append(feature_vector)

        if path is not None:
            # this option was useful if do_bb=True - then we would return the newly
            # cropped faces. Otherwise, we just return the old paths.
            faces.append(path)
        else:
            # so will only append if the load was successful
            faces.append(img_file)
    
    print('failed to load images = ', failed_to_load)
    
    return np.array(features), faces

def run_face_recog(img_files, args):
    '''
    Using face_recognizer package (which depends on dlib) to get feature
    vectors. In concept, this should be very similar to open face.
    ''' 
    features = [] 
    bad_count = 0
    for i, img_file in enumerate(img_files):
        
        try:
            image = face_recognition.load_image_file(img_file)
        except IOError:
            print('bad input')
            continue
        
        # a tuple in (top, right, bottom, left) order as wanted
        # TODO: Check the width vs height.
        bounding_box = (image.shape[1], image.shape[0], 0, 0) 

        encodings = face_recognition.face_encodings(image, [bounding_box], 50)
         
        if len(encodings) == 1:
            features.append(encodings[0]) 
        else:
            print('len of encodings is :( ', len(encodings))
            bad_count += 1

    print('ratio of bad encodings is ', float(bad_count)/len(img_files))
    print('len of features is ', len(features))
    
    return np.array(features), img_files

def gen_pickle_name(imgs, args):
    """
    Use hash of file names + which classifier we're using
    """
    hashed_input = hashlib.sha1(str(imgs)).hexdigest() 
    movie = args.dataset.split('/')[-2]

    if args.openface:
        cl_name = 'OPEN_FACE'
    else:
        cl_name = 'FR'

    name = movie + '_' + str(args.do_bb) + '_' + hashed_input[0:5] + '_' + cl_name

    directory = "./pickle/"
    return directory + name + ".pickle"

def tsne_gen_pickle_name(features):
    """
    Use hash of file names + which layer data we're storing. 
    """
    hashed_input = hashlib.sha1(str(features)).hexdigest() 
    name = hashed_input + '_' + 'tsne' 
    directory = "./pickle/"
    return directory + name + ".pickle"


def random_clustering(all_feature_vectors, func, *args, **kwargs):
    '''
    returns a clusters object - this depends on the function, for eg. will
    return a kmeans obj for kmeans, or dbscan object for dbscan - but all
    these have .clusters - which is what I use later.
    '''

    clusters = func(**kwargs).fit(all_feature_vectors)
    return clusters

def get_clusters(kmeans, imgs, features):
    '''
    Combines file_names for each of label elements the clusters_ - this is comman to all the clustering
    algorithms.
    TODO: Also add each of the feature vectors to the list so we can take
    mean/std etc of those.
    '''
    label_names = {}

    # kmeans.labels_ is the label assigned to each point.
    for i, label in enumerate(kmeans.labels_):
        
        label = str(label)
        if label not in label_names:
            label_names[label] = []

        file_name = imgs[i]
        label_names[label].append((file_name, features[i]))
    
    return label_names

def is_in_all_clusters(cluster, el, combined_clusters):
    '''
    Every element of cluster should be 'compatible' with el across all
    clusters.

    Crazy inefficient.
    '''
    for c in cluster:
        if not is_compatible(c[0], el[0], combined_clusters):
            return False

    return True

def is_compatible(file_1, file_2, combined_clusters):
    '''
    returns False, if file_1, file_2 are not together in each of the clusters
    in cluster paths.
    if they are together, then it is compatible, and returns True.
    '''
    for cluster_algs in combined_clusters:
        # each cluster alg has a bunch of labels, and lists (path, feature)
        # associated with them.
        for k, label in cluster_algs.iteritems():
            # elements of label are (file_name, feature_vec) tuples
            
            if check_file(file_1, label) != check_file(file_2, label):
                return False
     
    # print('returning True for compatible!!!!')
    # print(file_1)
    # print(file_2)

    return True

def check_file(file_name, label):
    '''
    file is path, lablel is list of (path, features)
    if file is in list[0]'s, then returns true.
    '''
    for l in label:
        path = l[0]
        if path == file_name:
            return True

    return False

def vote_clusters(combined_clusters, args):
    '''
    combined_clusters: list of 'label' dicts.
    label_dict: 'label_name': list of (img_name, feature_vector) 

    Algorithm idea:

    @ret: better_label_dict: 
    '''
    better_label_dict = defaultdict(list) 
    
    #TODO: Choose base_clusters to be the longer of the two.
    base_clusters = combined_clusters[0]
    
    for name, cluster in base_clusters.iteritems():
 
        for el in cluster:

            # Check if it belongs to any of the clusters so far.
            assigned_cluster = False
            
            # FIXME: Very inefficient.
            for k, new_cluster in better_label_dict.iteritems():
                # new_cluster is a list of guys. The label 'key' is the 0th
                # guys file name
                if is_in_all_clusters(new_cluster, el, combined_clusters):
                   better_label_dict[k].append(el) 
                   assigned_cluster = True
                   # print('el no. ', j, 'was assigned cluster!!!!!')
                   # print('el[0] is', el[0])
                   break

            # create new cluster
            if not assigned_cluster:
                # print('el no. ', j, 'was not assigned cluster')
                # print('el[0] is', el[0])
                better_label_dict[el[0]].append(el)
    
    new_better_dict = defaultdict(list)
    for k,v in better_label_dict.iteritems():
        if len(v) > 10:
            new_better_dict[k] = v

    return new_better_dict
        
def process_clusters(label_names, args, name=''):
    '''
    After we get cluster set up - either with kmeans, dbscan, or other
    algorithms - then we can display images (using opencv) or make big ass
    montages of them here.
    '''
    
    scores = []
    for l in label_names:
 
        img_names = [a[0] for a in label_names[l]]
        score = score_cluster(img_names)

        scores.append((score, len(label_names[l])))

        # Let's calculate the std/variance of the feature vectors in this
        # cluster.
        features = [a[1] for a in label_names[l]]
        var = np.var(features, axis=0)
        std = np.std(features, axis=0)
        
        if args.verbose:
            print('---------------------------------------')
            print('name of alg is ', name)
            print('label is ', l)
            unique_names = get_names(img_names) 
            for k,v in unique_names.iteritems():
                print('{} : {}'.format(k,v))


        if score < 0.60 and args.save_bad_clusters or args.save_cluster_imgs:

            # Let's save these in a nice view

            n = math.sqrt(len(label_names[l]))
            n = int(math.floor(n))
            
            rows = []
            for i in range(n):
                # i is the current row that we are saving.
                row = []
                for j in range(i*n, i*n+n, 1):
                    
                    file_name = label_names[l][j][0]
                    try:
                        img = Image.open(file_name)
                    except:
                        continue

                    row.append(img)

                if len(row) != 0: 
                    rows.append(combine_imgs(row, 'horiz'))

            final_image = combine_imgs(rows, 'vertical')

            img_names = [a[0] for a in label_names[l]]
            file_name = get_cluster_image_name(name, img_names, l, args)
            
            final_image.save(file_name, quality=100)

        if args.disp_pics:
            wait = raw_input("press anything to start this label")
            for label in label_names[l]:
                
                file_name = label
                # open the image with opencv
                img = cv2.imread(file_name)
                cv2.imshow('ImageWindow', img)
                c = cv2.waitKey()
                if c == 'q':
                    break
                # wait for a keypress to go to next image

    # sort according to the length of the clusters
    scores.sort(key=lambda x: x[1])
    
    total = 0
    for s in scores:
        print('num = {}, score = {}'.format(s[1], s[0]))
        total += s[0]
    
    print('average score = ', float(total)/len(scores))

def get_cluster_image_name(name, lst, label, args):
    '''
    Generates a unique name for the cluster_image - hash of the img names
    of the cluster should ensure that things don't clash.
    '''

    hashed_input = hashlib.sha1(str(lst)).hexdigest()

    movie = args.dataset.split('/')[-2]

    name = 'results/' + name + '_' + movie + '_' + hashed_input[0:5] + '_' + label + '.jpg'

    return name

def combine_imgs(imgs, direction):
    '''
    '''
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    if 'hor' in direction:
        min_shape = (30,30)

    # imgs = [i.resize(min_shape, refcheck=False) for i in imgs]

    if 'hor' in direction:

        imgs_comb = np.hstack( (np.asarray( i.resize(min_shape, Image.ANTIALIAS) ) for i in imgs ) )
        imgs_comb = Image.fromarray(imgs_comb)
    else:
        
        imgs_comb = np.vstack( (np.asarray( i.resize(min_shape, Image.ANTIALIAS) ) for i in imgs ) )
        imgs_comb = Image.fromarray(imgs_comb)

    return imgs_comb

def imgscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()

    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0, img_name in zip(x, y, images):

        try:
            img = plt.imread(img_name)
        except:
            continue

        im = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

def tsne_wrapper(features):
    tsne = TSNE(verbose=True, init='pca', learning_rate=500)
    return tsne.fit_transform(features) 


def run_tsne(all_feature_vectors, imgs, args):
    '''
    '''
    pickle_name = tsne_gen_pickle_name(all_feature_vectors) 
    Y = do_pickle(args.tsne_pickle, pickle_name, 1, tsne_wrapper, all_feature_vectors)
    
    return Y

def tsne_pic_plot(all_feature_vectors, Y, imgs):
    '''
    FIXME: Need to resize each image to something small.
    ''' 
    x = []
    y = []
    img_names = []
    
    # TODO: Resize the images.
    for i, coord in enumerate(Y):

        assert len(coord) == 2, 'coord not 2?'
        x.append(coord[0])
        y.append(coord[1])
        file_name = imgs[i]
        img_names.append(file_name)

    fig, ax = plt.subplots()
    imgscatter(x, y, img_names, zoom=0.5, ax=ax)
    ax.scatter(x,y)
    
    hashed_names = hashlib.sha1(str(img_names)).hexdigest()
    file_name = 'tsne_plt_pics' + hashed_names[0:5] + '.png'
    
    print('tsne name is ', file_name)
    plt.savefig(file_name, dpi=1200)
    
def tsne_color_plot(all_feature_vectors, Y, imgs):
    '''
    FIXME: Cycles through limited colors still...

    all_feature vectors are actually not required...
    '''
    assert len(all_feature_vectors) == len(Y) == len(imgs), 'all \
            must be equal'

    hashed_names = hashlib.sha1(str(imgs)).hexdigest()
    file_name = 'tsne_plt_color' + hashed_names[0:5] + '.png'

    print('tsne name is ', file_name)
    
    labels = get_tsne_names(imgs)
    assert len(labels) == len(imgs), 'simple assert'

    Plot.scatter(Y[:,0], Y[:,1], 20, labels);
    Plot.savefig(file_name, dpi=1200)

def sanity_check_features(features):
    '''
    Just ensures that the norm of features isn't 0 - as that is clearly a bad
    sign. Might want to do more extensive checks here.
    '''
    for i, row in enumerate(features):
        f1 = np.linalg.norm(row)
        assert f1 != 0, ':((('

def get_tsne_names(paths):
    '''
    Returns a num for each unique name in paths - so this can be used for
    tsne-colored graphs.
    '''
    names = []
    num = 0
    d = {}

    for p in paths:
        name = get_name_from_path(p)
        if name not in d:
            d[name] = num
            names.append(num)
            num += 1
        else:
            names.append(d[name])

    return names

def get_names(paths):
    '''
    Returns a list of unique names in paths.
    '''
    names = []
    d = defaultdict(int)

    for p in paths:
        name = get_name_from_path(p)

        d[name] += 1

    return d

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

def score_cluster(cluster):
    '''
    cluster is a list of files belonging in the cluster.
    Score = (max_same_name) / len(cluster)
    '''
    d = defaultdict(int)
    for path in cluster:
        name = get_name_from_path(path)
        d[name] += 1

    val = d[max(d, key=d.get)]

    return float(val) / len(cluster)

def main():
    
    start = time.time()

    args = ArgParser().args 
    imgs = load_img_files(args)

    all_features, all_imgs = get_features(imgs, args) 
    
    assert len(all_features) == len(all_imgs), 'features=images'

    '''
    AP - Not scalable, but best performance?
    Others all seem to scale quite well to large datasets.
    Just using a subset of data so can run it easily on AP and compare
    performance.
    '''
    train, _, train_imgs, _ = train_test_split(all_features, all_imgs,
            train_size=args.data_size, random_state=1234)
    
    feature_vectors = train
    imgs = train_imgs

    unique_names = get_names(imgs)
    print('num of unique names are ', len(unique_names))
    print('len of feature vectors is ', len(feature_vectors))
    cluster_algs = []

    if args.ap: 
        # divide it up into n-groups and then do stuff...
        # mini_features = feature_vectors[2000:4000]
        cluster_algs.append((random_clustering(feature_vectors,
            AffinityPropagation, damping=0.7), 'AP'))
    
    if args.kmeans:
        name = 'kmeans_' + str(args.clusters)
        cluster_algs.append((random_clustering(feature_vectors, KMeans,
                        n_clusters=args.clusters), name))
    
    if args.dbscan:
        cluster_algs.append((random_clustering(feature_vectors, DBSCAN,
            eps=0.5), 'DBScan'))
    
    if args.ac:
        cluster_algs.append((random_clustering(feature_vectors,
            AgglomerativeClustering,n_clusters=args.clusters), 'AC'))

    if args.mean_shift:
        cluster_algs.append((random_clustering(feature_vectors,
            MeanShift,cluster_all=False), 'MeanShift'))

    if args.birch:
        cluster_algs.append((random_clustering(feature_vectors, Birch,
            n_clusters=args.clusters), 'Birch'))

    if args.tsne:
        Y = run_tsne(feature_vectors, imgs, args)

        if args.tsne_pic_plot:
            tsne_pic_plot(feature_vectors, Y, imgs)
        if args.tsne_color_plot:
            tsne_color_plot(feature_vectors, Y, imgs)

        cluster_algs.append((random_clustering(Y, AgglomerativeClustering,
                n_clusters=args.clusters), 'tsne+AC'))
        
    if args.rank_order:

        # rank_order = Rank_Order(feature_vectors,
                # num_neighbors=args.ro_neighbors, Y = imgs,
                # alg_type=args.ro_alg)

        # D = rank_order.compute_all_distances()
        # cluster_algs.append((random_clustering(D, AgglomerativeClustering,
                # n_clusters=args.clusters, affinity='precomputed',
                # linkage='complete'), 'rank_order+AC'))

        for threshold in [0.2, 0.5]:

            rank_order = Rank_Order(feature_vectors,
                    num_neighbors=args.ro_neighbors, Y = imgs,
                    alg_type=args.ro_alg, cluster_threshold=threshold)

            D = rank_order.compute_all_distances()

            rank_order.cluster_threshold_ac(d_type=args.ro_cluster_dist)
            print('threshold = {}, num clusters = {} '.format(threshold,
                len(rank_order.clusters_)))

            cluster_algs.append((rank_order, 'rank_order_' + str(threshold)))

    # This part is only appropriate for sklearn cluster results
    combined_clusters = []
    for c, name in cluster_algs:
        clusters = get_clusters(c, imgs, feature_vectors)
        combined_clusters.append(clusters)
        
        print_clusters(clusters, name)    
        process_clusters(clusters, args, name=name)


    # better_clusters = vote_clusters(combined_clusters, args)

    # print_clusters(better_clusters, 'better')
    # process_clusters(clusters, args, name='better')

    # print('took {} seconds'.format(time.time() - start))

if __name__ == '__main__': 

    main()

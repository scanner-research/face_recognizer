import pickle
import os

from collections import defaultdict
from PIL import Image

from face import Face
from open_face_helper import OpenFaceHelper

from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

import math
import numpy as np

def random_clustering(all_feature_vectors, func, *args, **kwargs):
    '''
    returns a clusters object - this depends on the function, for eg. will
    return a kmeans obj for kmeans, or dbscan object for dbscan - but all
    these have .clusters - which is what I use later.
    '''
    clusters = func(**kwargs).fit(all_feature_vectors)
    return clusters

def get_cluster_image_name(name):
    '''
    Generates a unique name for the cluster_image - hash of the img names
    of the cluster should ensure that things don't clash.
    '''
    return name + '.jpg'
    # hashed_input = hashlib.sha1(str(lst)).hexdigest()

    # movie = args.dataset.split('/')[-2]

    # name = 'results/' + name + '_' + movie + '_' + hashed_input[0:5] + '_' + label + '.jpg'

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

class FaceDB:

    def __init__(self, feature_extractor = 'openface', open_face_model_dir=None,
            cluster_algs = None, db_name='test'):
        '''
        Specify parameters here.
        '''
        self.db_name = db_name
        self.feature_extractor = feature_extractor

        if cluster_algs is None:
            self.cluster_algs = ['AC']
        
        if feature_extractor == 'openface':
            assert open_face_model_dir is not None, 'specify open face model dir'
            self.open_face = OpenFaceHelper(model_dir=open_face_model_dir)
        else:
            assert False, 'Only using open face feature extractor'

        # Initialize previous version of the DB from disk. 
        self.faces = []
        self._load_old_faces()
        print('num faces are ', len(self.faces))
        
        # For now, just run the AP / or whatever algorithm repeatedly to create
        # the clusters.
        self.clusters = defaultdict(list)

        # Initialize the trained svms (?) or whatever other classifier we
        # choose to use in the end.

    def add_faces_from_video(self, video_id, paths_to_face_images, frame=False):
        '''
        For each face, ether assign it to an existing cluster (and keep track
        of that), or make a new cluster. At the end of the step, incorporate
        new clusters into global face database.

        @frame: Is each image a frame or an already detected face?
        '''
        pass
    
    def _already_added(self, face):
        '''
        If faces have already been added before, don't add them again -
        can check each face individually.
        '''
        for f in self.faces:
            if f.img_path == face.img_path:
                return True

        return False

    def _load_old_faces(self):
        '''
        Read pickle names from a file, read all those into self.faces.
        '''
        pickle_name = self._gen_pickle_name('faces')

        if os.path.isfile(pickle_name):
            #pickle exists!
            with open(pickle_name, 'rb') as handle:
                self.faces = pickle.load(handle)

    def _gen_pickle_name(self, name):
        '''
        Use hash of file names + which classifier we're using
        '''
        return './pickle/' + name + '_' + self.db_name + '.pickle'

    def _extract_features(self, face):
        '''
        Extracts features for one face object.
        '''
        
        #FIXME: Support option for openface itself to extract frames in the
        # image.
        try:
            face.features, _ = self.open_face.get_rep(face.img_path, do_bb=False,
                                                    new_dir=None)
        except Exception:
            # if it fails to open the file for whatever reason.
            return False
        
        return True

    def add_base_faces_from_videos(self, video_id_list, paths_to_face_images_list,
            frame=False): 
        '''
        @video_id_list: has same length as paths_to_face_images_list.
        @paths_to_face_images_list: each element is a list of images.

        Everything at once, if we haven't implemented the SVM version yet (for
        above function). Don't try to add faces to any cluster etc - but just
        runs the clustering algs etc and assigns clusters to faces.
        '''

        assert not frame, 'Right now we assume that faces are already extracted'
        assert len(video_id_list) == len(paths_to_face_images_list), 'same length'

        faces = []
        
        for i, vid_id in enumerate(video_id_list):
            for j, path in enumerate(paths_to_face_images_list[i]):
                face = Face(path, vid_id)
                if self._already_added(face):
                    print('already added file ', face.img_path)
                    continue

                if self._extract_features(face):
                    faces.append(face)
         
        # Add more guys to the faces list.
        self.faces += faces
        # Save it on disk for future.
        pickle_name = self._gen_pickle_name('faces')

        with open(pickle_name, 'w+') as handle:
            pickle.dump(self.faces, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        # Do further clustering and analysis on these faces
        self._cluster()
    
    def _cluster(self):
        '''
        Clusters the faces in self.faces.
        
        TODO: Need to see how we will integrate this with an incremental
        approach?

        TODO: Reconcile different clustering algorithm results (?) -
        ensemble_clustering?
        '''
        print('starting to cluster!')
        cluster_results = {}
        feature_vectors = [face.features for face in self.faces]

        for alg in self.cluster_algs:
            if alg == 'AP': 
                cluster_results['AP'] = random_clustering(feature_vectors,
                    AffinityPropagation, damping=0.5)
            elif alg == 'AC':
                cluster_results['AC'] = random_clustering(feature_vectors,
                    AgglomerativeClustering, n_clusters=5)
            
        #FIXME: tmp thing to assign clusters 
        for alg in cluster_results:
            for i, cluster in enumerate(cluster_results[alg].labels_):

                self.faces[i].cluster = cluster
                self.clusters[cluster].append(self.faces[i]) 
        
        print('num of clusters are ', len(self.clusters))
    
    def create_cluster_images(self):
        '''
        Takes the images from each cluster and makes a montage out of them.
        '''
        for k, cluster in self.clusters.iteritems():
            
            n = math.sqrt(len(cluster))
            n = int(math.floor(n))

            rows = []
            for i in range(n):
                # i is the current row that we are saving.
                row = []
                for j in range(i*n, i*n+n, 1):
                    
                    file_name = cluster[j].img_path

                    try:
                        img = Image.open(file_name)
                    except:
                        continue

                    row.append(img)

                if len(row) != 0: 
                    rows.append(combine_imgs(row, 'horiz'))

            final_image = combine_imgs(rows, 'vertical')

            img_names = [a.img_path for a in cluster]
            file_name = get_cluster_image_name('test' + str(k))
            
            final_image.save(file_name, quality=100)

    def lookup_face(self, face_image):
        '''
        Return expected identity or None if unknown - use either knn or train
        svm on each cluster for this to work.
        '''
        pass

    def num_unique_faces(self):
        '''
        We will need to consolidate clusters for this to work.
        '''
        return len(self.clusters)

    def num_videos_for_identity(identity_id): 
        pass



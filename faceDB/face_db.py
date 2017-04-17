import pickle
import os

from collections import defaultdict
from PIL import Image

from face import Face
from open_face_helper import OpenFaceHelper
from rank_order_cluster import Rank_Order

from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

import math
import numpy as np
import random

from sklearn.svm import NuSVC, SVC, LinearSVC

def random_clustering(all_feature_vectors, func, *args, **kwargs):
    '''
    returns a clusters object - this depends on the function, for eg. will
    return a kmeans obj for kmeans, or dbscan object for dbscan - but all
    these have .clusters - which is what I use later.
    '''
    clusters = func(**kwargs).fit(all_feature_vectors)
    return clusters

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

    #FIXME: Do a unison shuffle on these? Might help with training?
    return samples, labels

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
            cluster_algs = None, db_name='test', verbose=True,
            num_clusters=200, svm_merge=False):
        '''
        Specify parameters here.
        '''
        self.db_name = db_name
        self.feature_extractor = feature_extractor
        self.verbose = verbose
        
        self.num_clusters = num_clusters
        self.svm_merge = svm_merge
    
        if cluster_algs is None:
            self.cluster_algs = ['AP']
        
        if feature_extractor == 'openface':
            assert open_face_model_dir is not None, 'specify open face model dir'
            self.open_face = OpenFaceHelper(model_dir=open_face_model_dir)
        else:
            assert False, 'Only using open face feature extractor'

        # Initialize previous version of the DB from disk. 
        self.faces = []
        self._load_old_faces()
        print('num faces in the db are ', len(self.faces))
        
        # For now, just run the AP / or whatever algorithm repeatedly to create
        # the clusters.
        self.clusters = {}
        for cluster_alg in self.cluster_algs:
            self.clusters[cluster_alg] = defaultdict(list)
        
        # temporary: Need to decide on how to choose main clusters later.
        self.main_clusters = self.clusters[cluster_alg]

        # Initialize the trained svms (?) or whatever other classifier we
        # choose to use in the end.
        self.svms = {}

    def add_faces_from_video(self, video_id, paths_to_face_images, frame=False):
        '''
        For each face, ether assign it to an existing cluster (and keep track
        of that), or make a new cluster. At the end of the step, incorporate
        new clusters into global face database.

        @frame: Is each image a frame or an already detected face?
        '''
        pass
    
    def add_base_faces_from_videos(self, video_id_list, paths_to_face_images_list,
            labels = None, frame=False): 
        '''
        @video_id_list: has same length as paths_to_face_images_list.
        @paths_to_face_images_list: each element is a list of images.
        @labels: If it is labeled data - same format as
        paths_to_face_images_list.

        Everything at once, if we haven't implemented the SVM version yet (for
        above function). Don't try to add faces to any cluster etc - but just
        runs the clustering algs etc and assigns clusters to faces.
        '''

        assert not frame, 'Right now we assume that faces are already extracted'
        assert len(video_id_list) == len(paths_to_face_images_list) \
                            == len(labels), 'same length'

        faces = []
        
        for i, vid_id in enumerate(video_id_list):
            for j, path in enumerate(paths_to_face_images_list[i]):
                if labels is None:
                    face = Face(path, vid_id)
                else:
                    face = Face(path, vid_id, label=labels[i][j])

                if self._already_added(face) and self.verbose:
                    continue

                if self._extract_features(face):
                    faces.append(face)
                    if self.verbose:
                        print('added face ', face.img_path)
         
        # Add more guys to the faces list.
        self.faces += faces
        # Save it on disk for future.
        pickle_name = self._gen_pickle_name('faces')

        with open(pickle_name, 'w+') as handle:
            pickle.dump(self.faces, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        # Do further clustering and analysis on these faces
        self._cluster()
    
    def cluster_analysis(self, clusters):
        '''
        @clusters: given cluster dict - since we are still experimenting with
        different ways to set up the cluster.

        Measures the quality of the given cluster, which is a dict of the format: 
            cluster_name : [face1, face2...]
         
        Pairwise Precision: 

        Pairwise Recall:
        '''
        if self.verbose:
            print('starting cluster analysis')
        
        pairwise_precision = 0
        total_pairs = 0

        for _, cluster in clusters.iteritems():
            # cluster is a list of faces
            for i, face1 in enumerate(cluster):
                for j, face2 in enumerate(cluster): 
                    # To avoid double counting.
                    if i >= j: 
                        continue
                    if face1.label is None and face2.label is None:
                        continue

                    # If it is a mix of labelled / unlabelled, then having
                    # labeled and unlabeled together is considered a false
                    # match in the paper.
                    total_pairs += 1
                    if face1.label == face2.label:
                        pairwise_precision += 1
        
        pairwise_precision_score = float(pairwise_precision) / total_pairs
        print('pairwise precision = ', pairwise_precision_score)
        
        # Pairwise Recall:

        # collect all guys with same label together
        labels = defaultdict(list)
        for _, cluster in clusters.iteritems():
            for face in cluster:
                if face.label is not None:
                    labels[face.label].append(face)

        pairwise_recall = 0        
        total_recall_pairs = 0

        for _, labels in labels.iteritems():
            for i, face1 in enumerate(labels):
                for j, face2 in enumerate(labels): 
                    # To avoid double counting.
                    if i >= j: 
                        continue

                    # If it is a mix of labelled / unlabelled, then having
                    # labeled and unlabeled together is considered a false
                    # match in the paper.
                    total_recall_pairs += 1
                    if face1.cluster == face2.cluster:
                        pairwise_recall += 1
        
        pairwise_recall_score = float(pairwise_recall) / total_recall_pairs
        print('pairwise recall score is ', pairwise_recall_score)
        
        f_score = 2 * (pairwise_recall_score * pairwise_precision_score) \
                     / (pairwise_recall_score + pairwise_precision_score)

        print('F score is ', f_score)

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
        # image etc.
        try:
            face.features, name = self.open_face.get_rep(face.img_path,
                    do_bb=False, new_dir=None)

            if name is not None:
                face.img_path = name

        except Exception:
            # if it fails to open the file for whatever reason.
            return False
        
        return True
 
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
                    AgglomerativeClustering, n_clusters=self.num_clusters)
            
            elif alg == 'RO':
                rank_order = Rank_Order(feature_vectors,
                        num_neighbors=50, alg_type='approx')

                D = rank_order.compute_all_distances()
                cluster_results['RO'] = random_clustering(D, AgglomerativeClustering,
                        n_clusters=self.num_clusters, affinity='precomputed',
                        linkage='complete')

        #FIXME: tmp thing to assign clusters -- need to find a better way to
        # assign each image to its clusters
        for alg in cluster_results:
            for i, cluster in enumerate(cluster_results[alg].labels_):
                self.faces[i].cluster = cluster
                self.clusters[alg][cluster].append(self.faces[i]) 
        
        if self.svm_merge:
            self._merge_clusters()

        # Do further processing on each of the clusters now:

        # Cluster ensembling Fails to work. Find a better way to do ensemble
        # clustering somehow?
        # all_clusters = []
        # for alg in cluster_results:
            # assignments = []
            # for cluster in cluster_results[alg].labels_:
                # assignments.append(cluster)
            # all_clusters.append(assignments)
        
        # all_clusters = np.array(all_clusters)
        # consensus_clustering_labels = CE.cluster_ensembles(all_clusters,verbose = True, N_clusters_max = 200)        
    
    def _merge_clusters(self):
        '''
        Goes over self.main_clusters and tries to merge as many of them as
        possible. 
        '''
        print('len of clusters before merge is ', len(self.main_clusters))
        for cur_label, faces in self.main_clusters.iteritems():

            features = [face.features for face in faces]
            merge_label = self._check_for_merge(features)
            if merge_label is not None:
                self._merge_labels(merge_label, cur_label)
                continue

            self._train_svm(cur_label, features)        
        
        # Clean up the clusters which were marked as None
        self.main_clusters = {k:v for k,v in self.main_clusters.iteritems() if len(v) != 0}
        print('len of clusters after merge is ', len(self.main_clusters))
    
    def _train_svm(self, cur_label, features):
        '''
        '''
        negative_features = self._get_negative_features(len(features))
        X, Y = mix_samples(features, negative_features)
        # clf = NuSVC()
        # clf = SVC()
        clf = LinearSVC()
        clf.fit(X, Y)
        self.svms[cur_label] = clf

    def _merge_labels(self, merge_label, old_label):
        '''
        Will take all elements from old_label and put them into the cluster
        belonging to merge_label. Should also update the face objects etc.
        '''
        for face in self.main_clusters[old_label]:
            self.main_clusters[merge_label].append(face)
            # need to also update the face to point to the new cluster
            face.cluster = merge_label
            

        features = [face.features for face in self.main_clusters[merge_label]]
        # Should we do training of this svm again?
        self._train_svm(merge_label, features)

        self.main_clusters[old_label] = []

    def _check_for_merge(self, features):
        '''
        Iterates over all the svm's we have trained so far -- and if it finds
        one worth merging into, returns its key.
        Returns the label to merge to, or None.
        '''
        for label, clf in self.svms.iteritems():
            results = clf.predict(features)
            
            # if more than THRESHOLD (75%?) of these are predicted to be 1,
            # then we go ahead and merge.
            result = sum(results) / float(len(results))
            if result >= 1.00:
                return label

        return None

    def _get_negative_features(self, num):
        '''
        Where should we get these from? I guess the idea would be that these
        are chosen at random from a bigger dataset.
        
        TODO: get a bunch of random faces to use for this.
        '''
        
        feature_vectors = [face.features for face in self.faces]
        return random.sample(feature_vectors, num)

    def _get_cluster_image_name(self, name):
        '''
        Generates a unique name for the cluster_image - hash of the img names
        of the cluster should ensure that things don't clash.
        '''
        return name + '.jpg'
        # hashed_input = hashlib.sha1(str(lst)).hexdigest()

        # movie = args.dataset.split('/')[-2]

        # name = 'results/' + name + '_' + movie + '_' + hashed_input[0:5] + '_' + label + '.jpg'

        return name

    def create_cluster_images(self):
        '''
        Takes the images from each cluster and makes a montage out of them.

        TODO: main_clusters needs to be fixed upon.
        '''
        for k, cluster in self.main_clusters.iteritems(): 
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
            file_name = self._get_cluster_image_name('test' + str(k))
            
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
        return len(self.main_clusters)

    def num_videos_for_identity(identity_id): 
        pass



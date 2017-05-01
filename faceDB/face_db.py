import pickle
import os

from collections import defaultdict
from PIL import Image

from face import Face
from open_face_helper import OpenFaceHelper
from rank_order_cluster import Rank_Order
from util import *

from sklearn.cluster import KMeans, AffinityPropagation, DBSCAN
from sklearn.cluster import SpectralClustering, AgglomerativeClustering

import math
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.svm import NuSVC, SVC, LinearSVC
from sklearn.metrics import silhouette_samples, silhouette_score
import time

def _sklearn_clustering(all_feature_vectors, func, *args, **kwargs):
    '''
    returns a clusters object - this depends on the function, for eg. will
    return a kmeans obj for kmeans, or dbscan object for dbscan - but all
    these have .clusters - which is what I use later.
    '''
    clusters = func(**kwargs).fit(all_feature_vectors)
    return clusters

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
        self.cluster_algs = cluster_algs

        # Other parameters that I have just set at default for now.

        # 0,1,2 for different levels, with 2 being most verbose
        self.cluster_analysis_verbosity = 2
        self.save_bad_clusters = True
        self.torch_model = 'nn4.small2.v1.t7'

        self.merge_threshold = 0.70     # For merging clusters with svm.
        self.min_cluster_size = 10      # drop clusters with fewer
        self.good_cluster_score = 0.80  # used for testing
        
        # ugh: this was my convention for friends
        # u: unknown (usually too small faces), 'y': unknown female, 'x':
        # unknown male, z: not a face.
        # self._exclude_labels = ['u', 'y', 'x', 'z', 'small ', 'small']

        # convention for got:
        # a: unknown female, s: unknown male
        self._exclude_labels = ['a', 's', 'small', 'small ']
     
        if feature_extractor == 'openface':
            assert open_face_model_dir is not None, 'specify open face model dir'
            self.open_face = OpenFaceHelper(model_dir=open_face_model_dir,
                            torch_model=self.torch_model)
        else:
            assert False, 'Only using open face feature extractor'

        # Initialize previous version of the DB from disk. 
        self.faces = []
        self._load_old_faces()
        self.num_labeled_faces = self._calculate_labeled_faces()

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

    def _calculate_labeled_faces(self):
        '''
        '''
        # let's calculate the number of labeled faces.
        num_labeled = 0
        num_unlabeled = 0

        for face in self.faces:
            if face.label is None:
                num_unlabeled += 1
            else:
                num_labeled += 1

        print('num unlabeled faces are ', num_unlabeled)
        print('num labeled faces are ', num_labeled)
        return num_labeled

    def add_faces_from_video(self, video_id, paths_to_face_images, frame=False):
        '''
        For each face, ether assign it to an existing cluster (and keep track
        of that), or make a new cluster. At the end of the step, incorporate
        new clusters into global face database.

        @frame: Is each image a frame or an already detected face?
        '''
        pass
    
    def add_base_faces_from_videos(self, video_id_list, paths_to_face_images_list,
            labels = None, frame=False, cluster=True): 
        '''
        @video_id_list: has same length as paths_to_face_images_list.
        @paths_to_face_images_list: each element is a list of images.
        @labels: If it is labeled data - same format as
        paths_to_face_images_list.

        Everything at once, if we haven't implemented the SVM version yet (for
        above function). Don't try to add faces to any cluster etc - but just
        runs the clustering algs etc and assigns clusters to faces.
        '''
        assert len(video_id_list) == len(paths_to_face_images_list)

        faces = [] 
        for i, vid_id in enumerate(video_id_list):

            if frame:
                assert labels is None, 'cant label frames'
                # find new dir, based on the current dir of file
                new_dir = os.path.dirname(paths_to_face_images_list[i][0])
                new_dir = new_dir + '_faces'
                mkdir_p(new_dir)
            
            found_face = 0
            no_face = 0
            removed = 0

            all_paths = []
            for j, path in enumerate(paths_to_face_images_list[i]):
                 
                if frame:
                    # save files of different detected faces in the frame
                    # TODO: Deal with errors better
                    orig_path = path
                    try:
                        paths = self.open_face.frame_to_faces(path, new_dir)
                        found_face += 1
                    except Exception as e:
                        print('frame to face failed for path ', path)
                        print('Exception: ', e)
                        no_face += 1
                        continue
                else: 
                    paths = [path]

                for path in paths:
                    all_paths.append(path)

            self.open_face.save_images() 
            print('len of all paths is ', len(all_paths))
            for path in all_paths:
                # if we did face detection before, then path will be the
                # updated path of the new image.

                # TODO: get rid of the unneccessary ugly conditions here
                if labels is None:
                    face = Face(path, vid_id)
                else:
                    face = Face(path, vid_id, label=labels[i][j])
                if frame: 
                    face.orig_path = orig_path

                if self._already_added(face):
                    continue

                if self._extract_features(face):
                    faces.append(face)
                    if self.verbose:
                        print('added face ', face.img_path)

        # print('removed {} faces'.format(removed))
        # Add more guys to the faces list.
        self.faces += faces
        # Save it on disk for future.
        pickle_name = self._gen_pickle_name('faces')
        with open(pickle_name, 'w+') as handle:
            pickle.dump(self.faces, handle, protocol=pickle.HIGHEST_PROTOCOL) 
        
        # Do further clustering and analysis on these faces
        if cluster:
            self._cluster()
    
    def _score_cluster(self, faces):
        '''
        Simpler score measure, similar to pairwise_precision.
        cluster is a list of files belonging in the cluster.
        Score = (max_same_name) / len(cluster)
        '''
        d = defaultdict(int)
        total = 0
        for face in faces:
            if face.label in self._exclude_labels:
                continue
            d[face.label] += 1
            total += 1

        # empty cluster scores 1
        if len(d) == 0:
            return 1

        val = d[max(d, key=d.get)]

        return float(val) / total

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
        
        if self.num_labeled_faces != 0:
            self._f_score(clusters)

        total_faces = 0
        for k, faces in clusters.iteritems():
            total_faces += len(faces)

        # Assuming we know the labels, we can get the accuracy of each cluster
        # / and other statistics about that cluster, which might help us set
        # thresholds to better choose clusters.
        scores = []
        for k, faces in clusters.iteritems():
            
            if len(faces) < self.min_cluster_size:
                continue

            score = self._score_cluster(faces)

            # Let's calculate the std/variance of the feature vectors in this
            # cluster. 
            if score < self.good_cluster_score:
                print('---------------------------------------')
                print('going to start analyzing this bad cluster')
            else:
                print('**************************************')
                print('good cluster')

            features = [face.features for face in faces]
            print('num faces: ', len(faces))
            cohesion = self._cluster_cohesion(features, total_faces)
            print('cohesion : ', cohesion)
            print('score: ', score)
            scores.append((score, len(faces), cohesion)) 
 
            if score < self.good_cluster_score and self.cluster_analysis_verbosity >= 2:
                print('---------------------------------------')
                print('label is ', k)
                unique_names = defaultdict(int)
                for face in faces:
                    unique_names[face.label] += 1
                for k,v in unique_names.iteritems():
                    print('{} : {}'.format(k,v))

            if score < 0.60 and self.save_bad_clusters:
                # Let's save these in a nice view
                img_name = 'bad_cluster_' + str(k) + '_' + self.db_name
                self._save_cluster_image(faces, img_name)

        # sort according to the length of the clusters
        if self.cluster_analysis_verbosity == 2:
            scores.sort(key=lambda x: x[2])        
            total = 0
            for s in scores:
                print('num = {}, score = {}, cohesion = {}'.format(s[1], s[0],
                    s[2]))
                total += s[0]
        
            print('average score = ', float(total)/len(scores))
    
    def _cluster_cohesion(self, features, total_features):
        '''
        cluster is a list of features.
        '''
        row_mean = np.sum(features, axis=0)
        row_mean /= float(len(features))
        sum = 0
        for feature in features:
            sum += np.sum(np.square(row_mean - feature))
        
        # TODO: Does this normalization make sense?
        return float(sum) / len(features)
        # return float(sum) * (float(len(features)) / total_features)

    def _f_score(self, clusters):
        '''
        F-score analysis. We are not assuming we know the labels here, but just
        ignoring the faces with no labels as done in the Otto et al. paper.
        '''
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
                    if face1.label in self._exclude_labels:
                        continue
                    if face2.label in self._exclude_labels:
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
                    if face1.label is None and face2.label is None:
                        continue
                    if face1.label in self._exclude_labels:
                        continue
                    if face2.label in self._exclude_labels:
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
        final_name = name + '_' + self.db_name + '_' + self.torch_model
        return './pickle/' + final_name + '.pickle'

    def _extract_features(self, face):
        '''
        Extracts features for one face object.
        ''' 
        try:
            face.features = self.open_face.get_rep(face.img_path,
                    do_bb=False, new_dir=None)

        except Exception as e:
            # if it fails to open the file for whatever reason.
            print('extracting features failed, Exception was: ', e)
            return False
        
        return True
    def _find_best_k(self): 
        '''
        Use gap?
        So far this has been pretty inconclusive - but I think the
        normalization technique of dividing by num users in the cluster isn't
        the best one.
        '''
        X = [face.features for face in self.faces if not face.label in \
                    self._exclude_labels]

        for n_clusters in [2,8,16,22,28,32,64,128]:
            label_clusters = defaultdict(list)
            X = np.array(X)
            cluster_results = _sklearn_clustering(X,
                            AgglomerativeClustering, n_clusters=n_clusters)
            # silhouette_avg = silhouette_score(X, cluster_labels)
            # print("For n_clusters =", n_clusters,
                  # "The average silhouette_score is :", silhouette_avg)

            for i, cluster in enumerate(cluster_results.labels_):
                # all features
                label_clusters[cluster].append(X[i]) 

            cohesion = 0
            for _, features in label_clusters.iteritems():
                cohesion += self._cluster_cohesion(features, len(X))
            print('num clusters = ', n_clusters, 
                  "cohesion was ", cohesion)

    def _cluster(self):
        '''
        Clusters the faces in self.faces.
        
        TODO: Need to see how we will integrate this with an incremental
        approach?

        TODO: Reconcile different clustering algorithm results (?) -
        ensemble_clustering?
        '''
        print('starting to cluster!')
        print('cluster algs are: ', self.cluster_algs)
        k = self._find_best_k()

        cluster_results = {}
        # feature_vectors = [face.features for face in self.faces]
        feature_vectors = [face.features for face in self.faces if not face.label in \
                    self._exclude_labels]
        labeled_faces = [face for face in self.faces if not face.label in \
                    self._exclude_labels]

        assert len(feature_vectors) == len(labeled_faces), 'check'
        assert (labeled_faces[0].features == feature_vectors[0]).all(),\
                'features test'

        print('ignored faces = ', len(self.faces) - len(feature_vectors))
        feature_vectors = np.array(feature_vectors)
        
        for alg in self.cluster_algs:
            if alg == 'AP': 
                cluster_results['AP'] = _sklearn_clustering(feature_vectors,
                    AffinityPropagation, damping=0.5)
            elif alg == 'AC':
                cluster_results['AC'] = _sklearn_clustering(feature_vectors,
                    AgglomerativeClustering, n_clusters=self.num_clusters)
            
            elif alg == 'RO':
                rank_order = Rank_Order(feature_vectors,
                        num_neighbors=50, alg_type='approx')

                D = rank_order.compute_all_distances()
                cluster_results['RO'] = _sklearn_clustering(D, AgglomerativeClustering,
                        n_clusters=self.num_clusters, affinity='precomputed',
                        linkage='complete')

        #FIXME: tmp thing to assign clusters -- need to find a better way to
        # assign each image to its clusters
        for alg in cluster_results:
            for i, cluster in enumerate(cluster_results[alg].labels_):
                # self.faces[i].cluster = cluster
                # self.clusters[alg][cluster].append(self.faces[i]) 
                labeled_faces[i].cluster = cluster
                self.clusters[alg][cluster].append(labeled_faces[i]) 
        
        if self.svm_merge:
            self._merge_clusters()

        # Do further processing on each of the clusters now:

        # Cluster ensembling Fails to work. Find a better way to do ensemble
        # clustering somehow?
    
    def label_images(self): 
        '''
        makes it easier to label all the images, iterating through each cluster
        which should be reasonably similar. 
        And then
            - update face object with label
            - update img names too? 
        '''     
        start_time = time.time()
        
        # let's cluster first.
        label_clusters = defaultdict(list)
        feature_vectors = [face.features for face in self.faces]
        feature_vectors = np.array(feature_vectors) 
        cluster_results = _sklearn_clustering(feature_vectors,
            AgglomerativeClustering, n_clusters=self.num_clusters)
            
        for i, cluster in enumerate(cluster_results.labels_):
            # self.faces[i].cluster = cluster
            label_clusters[cluster].append(self.faces[i]) 


        cv2.namedWindow('image')

        print('len of mainclusters = ', len(label_clusters))
        small_images = 0
        i = 0
        for cl_name, faces in label_clusters.iteritems():
            print('going to start cl_name ', cl_name)
            print('i = ', i)
            i += 1

            prev_label = None
            j = 0
            # using while loop to support moving backwards
            while j < len(faces): 
                face = faces[j]
                j += 1
                
                # special casing for back option.
                if not face.label is None:
                    print('continuing to next label')
                    continue

                face_file = face.img_path
                # if this guy exists:
                # frame_file = face.img_path

                # open the image with opencv
                print('face file is ', face_file)
                img1 = cv2.imread(face_file)
                height, width = img1.shape[:2]

                # too small to judge
                if height < 50 or width < 40:
                    face.label = 'small'
                    small_images += 1
                    continue

                # img2 = cv2.imread(frame_file)
                cv2.imshow('image', img1)
                c = cv2.waitKey()
                print('c was ', chr(c))
                if c == 27:
                    break
                if c == 13:
                    # enter
                    face.label = prev_label
                elif chr(c) == 'b':
                    # go back one step.
                    print('going back!!!')
                    if j >= 2:
                        j -= 2
                        # so it doesn't get skipped over
                        faces[j].label = None

                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                    continue
                else: 
                    face.label = chr(c)
                    prev_label = face.label

                cv2.destroyAllWindows()
                cv2.waitKey(1)

                # using matplotlib
                # fig = plt.figure()
                # # ax = fig.add_subplot(111)
                # # ax.plot(np.random.rand(10))
                # img = mpimg.imread(face_file)
                # imgplot = plt.imshow(img)
                # plt.show()

        print('small images were ', small_images) 
        # self.faces has been updated by now!
        pickle_name = self._gen_pickle_name('faces')
        with open(pickle_name, 'w+') as handle:
            pickle.dump(self.faces, handle, protocol=pickle.HIGHEST_PROTOCOL) 

        print('saved new faces!')
        print('labeling took ', time.time() - start_time)
 
    def _merge_clusters(self):
        '''
        Goes over self.main_clusters and tries to merge as many of them as
        possible. 
        '''
        print('len of clusters before merge is ', len(self.main_clusters))

        # TODO: Iterate over the clusters in better order (by cohesion / num
        # faces in each cluster or some other heuristic)

        for cur_label, faces in self.main_clusters.iteritems():
            # features = [face.features for face in faces]
            features = [face.features for face in faces if not face.label in \
                        self._exclude_labels]
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
        TODO: Get a set of negative examples (just faces loaded into the db
        from before from which we can randomly chose)
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
        @features: multiple feature vectors.

        Iterates over all the svm's we have trained so far -- and if it finds
        one worth merging into, returns its key.
        Returns the label to merge to, or None.
        '''
        for label, clf in self.svms.iteritems():
            results = clf.predict(features)
            
            # if more than THRESHOLD (75%?) of these are predicted to be 1,
            # then we go ahead and merge.
            result = sum(results) / float(len(results))
            if result >= self.merge_threshold:
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

    def _get_cluster_image_name(self, name, images):
        '''
        Generates a unique name for the cluster_image - hash of the img names
        of the cluster should ensure that things don't clash.

        TODO: Use hash of images etc.
        '''
        return 'cluster_images/' + name + '.jpg'

    def create_cluster_images(self):
        '''
        Takes the images from each cluster and makes a montage out of them.

        TODO: main_clusters needs to be fixed upon.
        '''
        for k, cluster in self.main_clusters.iteritems(): 
            img_name = self.db_name + '_' + str(k)
            self._save_cluster_image(cluster, img_name) 
    
    def _save_cluster_image(self, cluster, img_name):
        '''
        Takes in a single cluster (of the type stored in self.main_clusters),
        and saves an nxn image of the faces in it.
        '''
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
        file_name = self._get_cluster_image_name(img_name, img_names) 
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



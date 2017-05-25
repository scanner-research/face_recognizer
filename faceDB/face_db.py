import pickle
import os

from collections import defaultdict
from PIL import Image

from face import Face, FaceCluster
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
import hashlib

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
            num_clusters=20, svm_merge=False, save_bad_clusters=False,
            merge_threshold=1.0):
        '''
        Specify parameters here.
        '''
        self.db_name = db_name
        self.feature_extractor = feature_extractor
        self.verbose = verbose
        self.merge_threshold = merge_threshold

        self.n_clusters = num_clusters
        self.svm_merge = svm_merge
        self.cluster_algs = cluster_algs

        # Other parameters that I have just set at default for now.

        # 0,1,2 for different levels, with 2 being most verbose
        # self.cluster_analysis_verbosity = 2
        # self.save_bad_clusters = save_bad_clusters
        self.torch_model = 'nn4.small2.v1.t7'

        # self.merge_threshold = 0.70     # For merging clusters with svm.
        self.min_cluster_size = 10      # drop clusters with fewer
        self.good_cluster_score = 0.80  # used for testing

        if feature_extractor == 'openface':
            assert open_face_model_dir is not None, 'specify open face model dir'
            self.open_face = OpenFaceHelper(model_dir=open_face_model_dir,
                            torch_model=self.torch_model)
        else:
            assert False, 'Only using open face feature extractor'

    def add_negative_features(self, detected_face_paths):
        '''
        @detected_faces: img paths of cropped faces
        '''
        pickle_name = self._get_paths_pickle_name(detected_face_paths)
        print('pickle name is ', pickle_name)
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as handle:
                self.negative_features = pickle.load(handle)
                print("successfully loaded pickle file!", pickle_name)
        else:
            self.negative_features = []
            for img_path in detected_face_paths:
                print(img_path)
                try:
                    features = self.open_face.get_rep(img_path)
                except Exception as e:
                    print('Exception: ', e)
                    continue
                self.negative_features.append(features)

            with open(pickle_name, 'w+') as handle:
                pickle.dump(self.negative_features, handle, protocol=pickle.HIGHEST_PROTOCOL)


        print('added {} negative \
        features'.format(len(self.negative_features)))

    def _get_paths_pickle_name(self, paths):

        hashed_name = hashlib.sha1(str(paths)).hexdigest()
        return './pickle/' + hashed_name[0:5] + '.pickle'

    def add_features(self, video_id, features, face_clusters=None):
        '''
        I prefer this method less because we won't have access to image paths -
        so can't make cluster images etc.
        @features: array of features.
        '''
        faces = []
        for feature in features:
            face = Face(features=features, video_id=video_id)
            faces.append(face)

        return self._add_faces(faces, face_clusters)

    def add_frames(self, video_id, frame_paths, face_clusters=None):
        '''
        @frames: img paths of frames.
        '''
        assert False, 'not implemented yet'

    def add_detected_faces(self, video_id, detected_face_paths, face_clusters=None):
        '''
        @detected_faces: img paths of detected faces.
        '''
        faces = []
        indices = []
        pickle_name = self._get_paths_pickle_name(detected_face_paths)
        if os.path.exists(pickle_name):
            with open(pickle_name, 'rb') as handle:
                (faces, indices) = pickle.load(handle)
                print("successfully loaded pickle file!", pickle_name)
        else:
            for i, path in enumerate(detected_face_paths):
                face = Face(img_path=path, video_id=video_id)
                if self._extract_features(face):
                    faces.append(face)
                    indices.append(i)
                    if self.verbose:
                        print('added face ', face.img_path)
            
            if len(faces) > self.n_clusters:
                with open(pickle_name, 'w+') as handle:
                    pickle.dump((faces, indices), handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                # Just return empty things.
                print("There were only {} faces in add detected faces".format(len(faces)))
                return ([], {}, []), []

        return self._add_faces(faces, face_clusters), indices

    def _add_faces(self, faces, face_clusters=None, n_clusters=None):
        '''
        For each face, ether assign it to an existing cluster (and keep track
        of that), or make a new cluster.

        This will be the core method that other functions - like ones accepting
        frames or feature vectors directly, will call.

        @faces: list of Face objects that have not been clustered yet.
        @face_clusters: dict of FaceCluster objects from before. Could be None
        if don't have previous clusters.

        @ret: ids: cluster identity assigned to each face_cluster
              face_clusters: dictionary of FaceCluster objects
              faces: 
        '''
        if n_clusters is None:
            n_clusters = self.n_clusters
        
        # Step 1: Cluster all the new faces.
        feature_vectors = np.array([face.features for face in faces])
        clusters = defaultdict(list)

        # TODO: More complicated clustering / combining clusters at this step.
        cluster_results = _sklearn_clustering(feature_vectors,
                        AgglomerativeClustering, n_clusters=n_clusters)
        
        # Convert it to dict format
        for i, label in enumerate(cluster_results.labels_):
            clusters[label].append(faces[i])

        # Generate FaceCluster objects
        new_face_clusters = {}
        for _, cluster in clusters.iteritems():
            # name of the cluster can't be its label as it has to be unique.
            name = cluster[0].name
            # use these negative features for training the svm in FaceCluster
            negative_features = self._get_negative_features(len(cluster))
            new_face_clusters[name] = FaceCluster(name, cluster,
                    negative_features=negative_features,
                    merge_threshold=self.merge_threshold)

            # also need to update each of the face objects in cluster with the
            # new name
            for face in cluster:
                face.cluster = name

        print('len of new face clusters = ', len(new_face_clusters))

        # Step 2: Merge the newly created clusters.
        new_face_clusters = self._merge_face_clusters(new_face_clusters)
        print('len of clusters after merging are: ', len(new_face_clusters))

        # Finish here if face_clusters was None
        if face_clusters is None:
            ids = [face.cluster for face in faces]
            return ids, new_face_clusters

        # Step 3: Merge with the clusters that already existed (face_clusters)
        # If any of the new clusters don't merge, just add them to the
        # face_clusters list.
        print('Step 3')
        print('orig len of face_clusters: ', len(face_clusters))
        added_clusters = {}
        for name, cluster in new_face_clusters.iteritems():
            if not self._try_merge_cluster(cluster, face_clusters):
                # Add cluster as a new cluster in face_clusters
                # FIXME: Deal with same images being added multiple times.
                assert not name in face_clusters, 'should not be possible'
                face_clusters[name] = cluster
                added_clusters[name] = cluster
                print('added new cluster to existing clusters!')

        print('final len of face clusters = ', len(face_clusters))
        ids = [face.cluster for face in faces]
        return ids, added_clusters, faces

    def _merge_face_clusters(self, face_clusters):
        '''
        @ret: A merged version of face clusters
        '''
        for name, cluster in face_clusters.iteritems():
            # Can this cluster be merged with any of the given clusters?
            if self._try_merge_cluster(cluster, face_clusters):
                # merged them!
                # delete previous cluster (set its name/svm to None)
                cluster.svm = None
                cluster.name = None

        # get rid of clusters which were merged (ie. there name was None)
        updated_clusters = {k:v for k,v in face_clusters.iteritems() if
                                v.name is not None}

        return updated_clusters

    def _try_merge_cluster(self, cluster_to_merge, face_clusters):
        '''
        Checks if the given FaceCluster can merge into any of the identities
        that existed from before. If it can, then it merges them and returns
        true.
        '''
        for _, new_cluster in face_clusters.iteritems():
            if new_cluster.name == cluster_to_merge.name:
                continue
            if new_cluster.svm is None:
                continue
            if new_cluster.check_merge(cluster_to_merge):
                new_cluster.merge(cluster_to_merge)
                # need to retrain the svm of the new cluster
                # First need the negative features
                features = [face.features for face in new_cluster.faces]
                negative_features = self._get_negative_features(len(features))
                new_cluster.train_svm(features, negative_features)
                return True

        return False
    
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
        if clusters is None:
            self._cluster()
            clusters = self.main_clusters

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

            if cohesion > 0.30 and score > 0.80:
                img_name = 'cohesion_cluster_' + str(k) + '_' + self.db_name
                save_cluster_image(faces, img_name)

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
                save_cluster_image(faces, img_name)

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


    def _gen_pickle_name(self, db_name, name):
        '''
        Use hash of file names + which classifier we're using
        '''
        final_name = name + '_' + db_name + '_' + self.torch_model
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
            if self.verbose:
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
        # k = self._find_best_k()
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
        
        if len(feature_vectors) < 5:
            return

        for alg in self.cluster_algs:
            if alg == 'AP':
                cluster_results['AP'] = _sklearn_clustering(feature_vectors,
                    AffinityPropagation, damping=0.5)
            elif alg == 'AC':
                cluster_results['AC'] = _sklearn_clustering(feature_vectors,
                    AgglomerativeClustering, n_clusters=self.n_clusters)

            elif alg == 'RO':
                rank_order = Rank_Order(feature_vectors,
                        num_neighbors=50, alg_type='approx')

                D = rank_order.compute_all_distances()
                cluster_results['RO'] = _sklearn_clustering(D, AgglomerativeClustering,
                        n_clusters=self.n_clusters, affinity='precomputed',
                        linkage='complete')

        for alg in cluster_results:
            for i, cluster in enumerate(cluster_results[alg].labels_):
                # self.faces[i].cluster = cluster
                # self.clusters[alg][cluster].append(self.faces[i])
                labeled_faces[i].cluster = cluster
                self.clusters[alg][cluster].append(labeled_faces[i])

        if self.svm_merge:
            self._merge_clusters()


    def _get_negative_features(self, num):
        '''
        Where should we get these from? I guess the idea would be that these
        are chosen at random from a bigger dataset.

        TODO: get a bunch of random faces to use for this.
        '''
        # feature_vectors = [face.features for face in self.faces]
        if num > len(self.negative_features):
            num = len(self.negative_features)

        return random.sample(self.negative_features, num)

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

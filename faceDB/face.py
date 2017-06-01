from collections import defaultdict
from util import *
from sklearn.svm import NuSVC, SVC, LinearSVC

class Face():

    def __init__(self, img_path=None, video_id = None, label = None,
            features=None, frame=None):
        '''
        Contains basic information about the face.

        @img_path: file path on disk
        @video_id: which video it is from. Can be useful as a feature, but we
        aren't using it right now.
        @label: If it is from a labelled dataset, then we can use these labels
        to analyze the quality of clustering etc.

        Can extend this to include more diverse features - such as additional
        audio/video features, other faces that it cannot be etc.
        '''
        self.img_path = img_path
        self.features = None

        self.video_id = video_id
        self.label = label

        if img_path is not None:
            self.name = img_path
        else:
            hashed_name = hashlib.sha1(str(features)).hexdigest()
            self.name = hashed_name[0:5]

        # Each of the clustering algorithms, like AC, AP etc can assign this
        # face to a different cluster, but here we only want to store the final
        # assignment after whatever algorithms we use to determine the cluster.
        self.cluster = None
        self.frame = frame

class FaceCluster():

    def __init__(self, name, faces, negative_features=None, svm=None,
            merge_threshold=1.00):
        '''
        Provides a list of face objects for this cluster.
        @name: string
        @faces: face objects
        '''
        self.name = name
        self.faces = faces

        # We will train an svm on the objects of this cluster
        self.merge_threshold = merge_threshold
        if svm is not None:
            self.svm = pickle.loads(svm)
        elif negative_features is not None:
            features = [face.features for face in self.faces]
            self.train_svm(features, negative_features)

        # Add cohesion to the clusters so we can iterate in the correct order
        # or drop clusters etc.
        self.cohesion_score = 0

    # TODO: Helper methods to train the svm etc.
    def check_merge(self, cluster):
        '''
        Checks if this cluster should be merged with another.
        '''
        features = [face.features for face in cluster.faces]
        results = self.svm.predict(features)
        result = sum(results) / float(len(results))
        if result >= self.merge_threshold:
            return True

        return False

    def merge(self, cluster):
        '''
        Eats up cluster into new cluster.
        Also, updates face.cluster for each of the faces in the cluster we are merging.
        '''
        for face in cluster.faces:
            face.cluster = self.name
            self.faces.append(face)

        # Update svm should happen from the caller - as we don't have access to
        # negative samples here. (Or provide negative samples here)

    def train_svm(self, features, negative_features):
        '''
        '''
        X, Y = mix_samples(features, negative_features)
        clf = LinearSVC()
        clf.fit(X, Y)
        self.svm = clf

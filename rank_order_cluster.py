from sklearn.neighbors import NearestNeighbors
import numpy as np
np.set_printoptions(2, suppress=True)
import time
import math
from util import *
import hashlib

class Rank_Order():

    def __init__(self, X, num_neighbors=None, Y = None, alg_type='approx',
            cluster_threshold=1.0):
        '''
        X are all the feature vectors.

        @type: approx vs exact
        '''
        self.X = X
        self.Y = Y
        self.alg_type = alg_type
        self.cluster_threshold = cluster_threshold

        if self.alg_type == 'exact':
            # assert num_neighbors is None, 'num neighbors just used for approx'
            self.num_neighbors = len(self.X)
        else:
            self.num_neighbors = num_neighbors

        # compute nearest neighbors fit
        nbrs = NearestNeighbors(n_neighbors=self.num_neighbors).fit(X)
        _, self.k_ranks = nbrs.kneighbors(X)
        
    def compute_all_distances(self):
        '''
        Computes the distance matrix D, between each pair of points, using the
        rank order algorithm described in ....
        '''
        name = self._gen_pickle_name()
        self.D = do_pickle(True, name, 1, self._compute_all_distances)

    def _gen_pickle_name(self):
        
        hashed_input1 = hashlib.sha1(str(self.X)).hexdigest() 
        hashed_input2 = hashlib.sha1(str(self.Y)).hexdigest()  
        hashed_input = hashed_input1[0:5] + hashed_input2[0:5]
         
        name = './pickle/bigD_' + hashed_input +'_'+ str(self.num_neighbors) \
                +'.pickle'
        return name

    def _compute_all_distances(self):

        start = time.time()
        N = len(self.X)

        D = np.zeros((N,N))

        for i in range(N):
            # compute the distance of ith element with each of the other
            # elements.
            for j in range(i, N, 1):
                D[i][j] = round(self._symmetric_distance(i, j), 2)
                D[j][i] = D[i][j]
 
        print('compute all distances took ', time.time() - start)
        distances_sanity_check(D)        
        return D

    def _symmetric_distance(self, a, b):
        '''
        @a, b: are indices into the face list.

        Identical to the D_m(a,b), equation 4, in Otto et al.
        '''
        d1 = self._distance(a,b)
        d2 = self._distance(b,a)

        # FIXME: check implementation of O(a,b) for the fixed case - or maybe
        # use the full distance matrix for this part?

        normalizer = min(self._O(a, b), self._O(b, a))
        
        # FIXME: More elegant solution to avoiding div by zero? This was not a
        # problem in the equations used in the paper because everything started
        # with 1
        normalizer += 1
        
        return float(d1 + d2) / normalizer

    def _distance(self, a,b):
        '''
        Identical to equation 3 in Otto et al.
        '''
        d_sum = 0
        # loop will not go beyon k
        
        if self.alg_type == 'approx':

            for i in range(min(self.num_neighbors, self._O(a,b))):

                rank = self._O(b, self._f(a,i)) 
                # Note: rank would always be = self.num_neighbors if it is not
                # present in the list.
                if rank < self.num_neighbors:
                    d_sum += 0
                else:
                    d_sum += 1

        elif self.alg_type == 'exact':
            for i in range(self._O(a,b)):
                d_sum += self._O(b, self._f(a,i))

        else: 
            assert False, 'only exact, approx rank order alg supported'
        return d_sum

    def _O(self, a, b):
        '''
        Gives rank of face b in a's neighbor list.

        FIXME: Should this be done with the full neighbor list computed?
        
        Test this function
        '''
        
        assert len(self.k_ranks[a]) == self.num_neighbors, 'k-neighbors for \
                each list'

        for i, face in enumerate(self.k_ranks[a]):
            if face == b:
                return i
        
        # I guess this is the most sensible option here?
        assert self.alg_type == 'approx', 'can only hapen for approx'

        return self.num_neighbors

    def _f(self, a, i):    
        '''
        @a,i: ints
        i-th face in neighbor list of a.

        @ret: face_index.
        '''
        return self.k_ranks[a][i]

    def cluster_threshold_ac(self, cluster_threshold=None, d_type='min'):
        '''
        Use a naive implementation of Agglomerative Clustering to compute the
        clusters which merges two clusters if the distance between them is
        below a threshold.

        ret: key : [1,2,...5]
        '''
        start = time.time()
        if cluster_threshold is not None:
            self.cluster_threshold = cluster_threshold

        clusters = {}

        # initialize each cluster to each feature 
        for i in range(len(self.X)):
            clusters[str(i)] = [i]
        
        # Use threshold to iteratively merge clusters

        found_cluster = True
        iteration = 0

        while found_cluster:
            iteration += 1
            print('clustering iteration = ', iteration) 

            found_cluster = False
            for i in clusters: 

                # try to merge it with any of the other clusters
                for j in clusters:
                    if i == j:
                        continue

                    d = self._cluster_distance(clusters, str(i), str(j),
                            d_type=d_type)

                    if d < self.cluster_threshold:
                        # append every guy from the jth cluster to i
                        for el in clusters[str(j)]:
                            clusters[str(i)].append(el) 
                        # remove the other guy from the clusters.
                        clusters[str(j)] = None
                        found_cluster = True
        
        self.clusters_ = {k:v for k,v in clusters.iteritems() if v is not None}

        print('clustering took ', time.time() - start)
        self.labels_ = self._convert_to_labels_()
        return self.clusters_                        
    
    def _convert_to_labels_(self):
        '''
        Converts the dict self.clusters_ to the labels_ format used in sklearn.
        '''
        labels = np.zeros(len(self.X))

        for cluster, vals in self.clusters_.iteritems():
            for val in vals:
                labels[val] = cluster
        
        return labels

    def _cluster_distance(self, clusters, c1, c2, d_type='min'):
        '''
        there are different possible ones here, will return the min of each
        pair of a,b where a \in i, b \in j as mentioned in the paper.

        Another obvious thing will be to just compare the means of the
        clusters.
        '''
        if clusters[c1] is None or clusters[c2] is None:
            return float("inf")
        
        distances = []

        for i in clusters[c1]:
            for j in clusters[c2]: 
                distances.append(self.D[i, j])
        
        if d_type == 'min':
            return min(distances)
        elif d_type == 'mean':
            return np.mean(np.array(distances))


def distances_sanity_check(D):
    '''
    '''
    # check if it is symmetric:
    assert (np.transpose(D) == D).all(), 'should be symmetric matrix'

    # basic check per row.
    for row in D:
        num_unique = len(set(row))
        assert num_unique > 5, 'should be unique d values'


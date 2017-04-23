FaceDB:
  
  - Performance of LFIW dataset
    - AP, AC both perform very poorly on F-Measure
    - Rank Order: Need to optimize it further before running it
    (ocean/halfmoon?)

  - Performance on a subset of VGG, (20K images from ~163 celebs) 
    - 5k images, from all 163 labels:
      - AP, AC have ~80% pairwise precision (AP > AC)
      - Low recall (< 30%)
      - SVM Merging: Improves recall, but decreases precision. Best case had
      both of them in the 50-60% range.
    
    - ~1k images, 10 labels (100 each):
      - AP, AC have >90% pairwise precision
      - Recall is low initially
      - SVM Merging: Improves recall considerably without decreasing precision.
        Haven't test different possible thresholds etc. but basically brings
        recall up to ~70%, and I guess could be improved further.

  - Labeled Videos (In progress):
    - Friends Season 7 Favorite scenes (youtube playlist)

    - Game of Thrones (......)

    - TVF show playlist, with recurring characters in different roles

TODO:
- Video based stuff - pipeline + labeling etc
- Add new files based on labels.
- Do a lot of different clusters (with different thresholding values for svm
    merge, different num clusters etc.), and then choose the best cluster based 
   on things like silhouette score / cohesion score (this can be expanded). Not
   as nice as some sort of cluster ensembling, but better than just one single
   run for a cluster.
- incremental approach with incorporating new videos into the faceDB.
- Bigger / other nn-models
- Rank Order on LFIW
- Read Ensemble clustering (and other cluster analysis stuff)
- centroid based clustering 
- get timings, audio associated with each frame/'face' object

Other TODOs:- 

-- torch to caffe:
  - cant get THPP which is a dependency to be installed because can't get
  thrift installed

-- rank-distance metric 
  - optimize exact
  - download lfiw dataset and test of that
  - implement F-score

-- what is resampling stuff? Check the dbscan library that supported
resampling?

Parallel version of vgg daces download:

Num clusters >> num faces:

  - Can solve this by training an SVM on clusters we are very sure about (not
      much std/variance)

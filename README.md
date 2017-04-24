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
      
        https://www.youtube.com/watch?v=gyp9C4gUKiA&list=PL4VjYMnxdYuVdqqAuuURtNHY5cfNFy606

        - TODO:  
            need to increase the quality of pictures for labeling - somehow the
            quality appears a lot less than when I was saving them in montages,
            and probably led to some mislabels.

        - friends1: Rachel's Erotic Book
          - Main identities: Rachel, Joey, Ross
          - phoebe pics seem low quality (?) not sure about all of them
          - face detect doesn't do a great job at multiple people in scene
          situations (but will need to test this empirically on more videos)
          - Took around ~45 minutes for 4800 images but this can probably be
          made quicker with better quality images etc. Perhaps should use full
          scenes with rectangles on faces while labeling? Not sure.
        
          - Few identities a lot of pictures is a very different scenario than
          the one we had tested so far.
          
          Results:
          - tldr; If we magically knew the right number of clusters, then we
          can get high accuracy with agglomerative clustering, but there was
          still one bad cluster (is this constant across videos or just this
              one?)

          Note: knowing the right number of clusters might not be such an
          outlanding idea because there has been a bunch of work about
          estimating the correct number of clusters based on various
          statistical measures. Also, just as a simple heuristic - the sum of
          squared errors (which appears to be a better metric than std/mean 
          for estimating quality of clusters) was quite high for the bad
          clusters - so we could literally just drop the worst cluster(s) like
          this and achieve very high accuracy. But need to check if this
          generalizes across videos. Had notices a similar thing in faces (with
              only 10 identities), but haven't tested this in situations with
          many ids (eg. 163 in the vgg dataset subset i'm using)

          - Main issue appears to be with merging clusters - If somehow we knew
          how many identities there are, then the performance is great. But I
          think we can do tuning for this using: 
            - do a lot of clusters and check silhouette score
            - do a lot of clusters and check cohesion scores of each  
            - gap statistic (I think it might be doing simialr things to the
                above 2)
            - other measures? Need to research this area more.
            - Might have better measures for ensemble clustering
            - Maybe take the worst cluster (assuming it has the worst cohesion
                score) and try to merge them into other clusters

          - Also possibly dropping small bounding boxes would result in much
          higher accuracy - as I found a lot of images with small bounding
          boxes fairly confusing.

          - Affinity Propogation might not be a great algorithm for such
          situations - throws up a huge number of clusters (~100), and while
          svm merging reduces it considerably, it was still quite high.

    - Game of Thrones (......)

    - TVF show playlist, with recurring characters in different roles:

TODO:
- Improve labeling process
  - quality of displayed images
  - might use full scenes with bounding boxes instead?

- rename files based on labels.
- Do a lot of different clusters (with different thresholding values for svm
    merge, different num clusters etc.), and then choose the best cluster based 
   on things like silhouette score / cohesion score (this can be expanded). Not
   as nice as some sort of cluster ensembling, but better than just one single
   run for a cluster.

- incremental approach with incorporating new videos into the faceDB. (should
    be straightforward)
- Torch model to caffe model
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

FaceDB:
  
  - Scoring metrics: Just defining a few metrics that I refer to repeatedly
  below.
    - pairwise precision: considers how precise identities are in a cluster,
    normalized to represent score across clusters.
    - pairwise recall: considers how distributed the identities are across all
    clusters.
    - 'score': just a simpler metric for each cluster, in which we take the identity dominating
    the cluster and divide by the size of the cluster. (basically gets us
        something like pairwise precision), could have also used pairwise
        precision here instead.
    
    - cohesion score: sum of squared errors of each element in the cluster from
    the cluster's mean. Not sure if it makes sense to normalize it by dividing
    with the number of elements or not? 

  - Deciding number of clusters, k:

    - So far the performance of the clustering algorithms seem great if we know
    an approximately good k (how many identities are there in this video?), so
    this is definitely an important topic.
    
    - Trying different k's and looking at overall cohesion scores did NOT
    work well - but I think this may be due to the fact that I was dividing by
    nunber of elements in cluster, so small k-values benefit because there
    number of k-values would be greater.

    - Check out 'gap' statistic: https://web.stanford.edu/~hastie/Papers/gap.pdf
            
    - Besides this, there is an approach of having a high k and then merging
    clusters, which doesn't seem bad (pairwise precision stays high) - but the clusters aren't merged down to
    the ideal levels so far, and the nice property - that the worst cluster has
    the highest cohesion does not seem to be as cleanly applicable (needs more
        testing). Here, I think there is definitely room for improvement in the
    way I'm merging clusters together - right now I have a very simple setup
    with a Linear SVC.

    - Use video metadata for heuristics: Since we don't need a precise number,
    but let's say +/- 10 seems to perform decently, so we can use things like:
      - subtitles
      - can analyze audio completely separately to identify how many different
      speakers were there
      - Maybe the title / type of video, along with its length can give us a
      good heuristic. 
  
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
        
        - 'top 10 moments' ~10 minutes
        - https://www.youtube.com/watch?v=qneehBzpJKg
        - skipped all images with len or width < 50
        - tldr; At first look, the results seem similar to the friends1 set.
        
        - comparitively few detected faces (only ~3.9k detected faces despite
            video being twice as long as friends1 - possibly because of messier
            game of thrones kind of video)
        - Many more identities, I labeled 20+ identities, although a few
        identities do dominate it (eg. arya stark)
        - took around ~20 minutes, despite slightly slow manually assigning
        labels to identities the first time an identitiy comes up
        
        - If I put the number of clusters in the 'right' range, ie.20-30, it
        performs quite well 
            - Pairwise precision score: 90+%
            - Pairwise recall score: ~30%
            - Only one really bad cluster - and just like in the friends1
            sample, it has the highest cohesion score - so if we continue with
            the heuristic of dropping the 'worst' cluster, we might do well.

        - If we can't find the right number of clusters, we can choose a
        conservatively high estimate, and merge them later using svm in a
        supervised learning fashion. 

          - With large number of clusters (around 100), we can merge clusters to
          bring it to around ~45 without losing on pairwise_precision, but
          pairwise_recall scores are pretty low still. 

          - With the higher number of clusters, it is less easy to single out
          based on cohesion scores, but for some bad clusters we could do:
            a. drop highest cohesion scores (gets rid of 2 bad clusters)
            b. drop clusters with too few elements (like < 30)
          But this doesn't deal with all the issues still, and there still
          remains bad clusters.

    - TVF show playlist, with recurring characters in different roles:

Major TODOs:
- face tracking (useful for labeling and possibly clustering etc)
- audio features integration (voice-id etc)
- Read more on Ensemble clustering (and other cluster analysis stuff)

Minor TODOs:
- Improve labeling process
  - quality of displayed images
  - might use full scenes with bounding boxes instead?

- rename files based on labels.
- Do a lot of different clusters (with different thresholding values for svm
    merge, different num clusters etc.), and then choose the best cluster based 
   on things like silhouette score / cohesion score (this can be expanded). Not
   as nice as some sort of cluster ensembling, but better than just one single
   run for a cluster.

- re-run on a bunch of different clustering algorithms - particularly, maybe
the cluster many times then choose best cluster based on heuristics like
silhouette score could make kmeans sufficiently good as compared to
agglomerative clustering - especially since kmeans is very sensitive to the
starting centroids etc. (they were close anyway before on still images). Kmeans
is probably a lot easier to scale up (using faiss etc) so this would be nice to
know

- incremental approach with incorporating new videos into the faceDB. (should
    be straightforward)
- Torch model to caffe model
- Bigger / other nn-models
- Rank Order on LFIW
- centroid based clustering 


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

-- Parallel version of vgg daces download (if we want to use the full dataset...)

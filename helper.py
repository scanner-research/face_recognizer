import argparse
import itertools
import openface
import glob
import os
import cv2
import dlib
import time

def str2bool(v):
  '''
  used to create special type for argparse
  '''
  return v.lower() in ("yes", "true", "t", "1", "True")


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

class ArgParser():
    '''
    Add other args that are needed here.
    '''

    def __init__(self):
        '''
        '''

        parser = argparse.ArgumentParser()

        parser.register('type', 'Bool', str2bool)

        # Just set the modelDir automatically and the rest are based on that.
        modelDir = '/Users/Parimarjann/openface/models/'
        
        # Other stuff from open face sample.
        dlibModelDir = os.path.join(modelDir, 'dlib')
        openfaceModelDir = os.path.join(modelDir, 'openface')

        parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                            default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
        parser.add_argument('--networkModel', type=str, help="Path to Torch network model.",
                            default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))

        parser.add_argument('--imgDim', type=int,help="Default image dimension.", default=96)

        parser.add_argument("-v", "--verbose", help="increase output \
                            verbosity",default=False,type='Bool')

        parser.add_argument("--ac", help="AgglomerativeClustering",
                            default=False, type='Bool')
        parser.add_argument("--kmeans", help="KMeans",
                            default=False, type='Bool')
        parser.add_argument("--ap", help="AffinityPropogation",
                            default=False, type='Bool')
        parser.add_argument("--dbscan", help="DBScan",
                            default=False, type='Bool')
        parser.add_argument("--birch", help="Birch",
                            default=False, type='Bool')
        parser.add_argument("--mean_shift", help="MeanShift",
                            default=False, type='Bool')

        parser.add_argument("--tsne", help="TSNE",
                            default=False, type='Bool')
        parser.add_argument("--tsne_pic_plot", help="TSNE + pic plot",
                            default=False, type='Bool')
        parser.add_argument("--tsne_color_plot", help="TSNE + color plot",
                            default=True, type='Bool')
        parser.add_argument("--normalize", help="normalize features",
                            default=False, type='Bool')
        parser.add_argument("--scale", help="scale features",
                            default=False, type='Bool')
        parser.add_argument("--pickle", help="to pickle or not pickle",
                            default=True, type='Bool')
        parser.add_argument("--tsne_pickle", help="tsne pickle",
                            default=True, type='Bool')

        parser.add_argument("--do_bb", help="do bounding box \
               with openface or treat full image as bb", 
                            default=False, type='Bool')

        parser.add_argument("--dataset", help="name of dir in the data/ \
                directory that we are going to use",
                default='vgg_face_dataset/dataset_images', type=str)

        parser.add_argument("--batch_size", help="num features to use in run",
                            default=0, type=int)
        parser.add_argument("--clusters", help="num clusters for diff algs",
                            default=8, type=int)


        parser.add_argument("--disp_pics", help="save label pictures",
                            default=False, type='Bool')

        parser.add_argument("--save_cluster_imgs", help="save each cluster in \
                            an nxn square ",
                            default=False, type='Bool')
 
        # Add movie folder etc here as well.

        # Classifiers
        parser.add_argument("--openface", "--of", help="use openface to gen features",default=True,type='Bool')

        parser.add_argument("--face_recognizer", "--fr", help="use fr \
                gen features",default=False,type='Bool')

        self.args = parser.parse_args()


class Open_Face_Helper():

    def __init__(self, args):

        self.align = openface.AlignDlib(args.dlibFacePredictor)
        self.net = openface.TorchNeuralNet(args.networkModel, args.imgDim)
        self.args = args

    def get_rep(self, imgPath, do_bb=False, new_dir=None):
        '''
        For a single image...

        @do_bb: providing a bounding box so don't have to do image detection at
        this step (Might want to keep image detection code for the future?)

        dlib.rectangle object
        '''
        if self.args.verbose:
            print("Processing {}.".format(imgPath))
        bgrImg = cv2.imread(imgPath)

        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.args.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        
        # We already have face bounding boxes so get rid of this step.
        if do_bb:
            bb = self.align.getLargestFaceBoundingBox(rgbImg)
        else:
            # treat the given image dimensions as the bb
            bounding_box = (0, 0, rgbImg.shape[0], rgbImg.shape[1]) 
            bb = _css_to_rect(bounding_box)

        # TODO: if do_bb, then we want to save the image based on the bounding
        # box as a new image - and return that name as well.
        
        start = time.time()
        alignedFace = self.align.align(self.args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if self.args.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))
        
        if do_bb:
            name = os.path.basename(imgPath)
            name = os.path.join('bb_faces_2', name)
            cv2.imwrite(name, alignedFace)
            print('saved image ', name)
        else:
            name = None

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.args.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")

        return rep, name


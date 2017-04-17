import itertools
import openface
import glob
import os
import cv2
import dlib
import time

def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])

class OpenFaceHelper():

    def __init__(self, model_dir, args=None):
        ''' 
        All defaul values. 
        TODO: Might want to experiment with these later.
        '''

        if args is None:
            self.verbose = False
            self.img_dim = 96

        dlibModelDir = os.path.join(model_dir, 'dlib')
        openfaceModelDir = os.path.join(model_dir, 'openface')

        network_model = os.path.join(openfaceModelDir, 'nn4.small2.v1.t7')
        self.align = openface.AlignDlib(os.path.join(dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

        self.net = openface.TorchNeuralNet(network_model, self.img_dim)


    def get_rep(self, imgPath, do_bb=False, new_dir=None):
        '''
        For a single image...

        @do_bb: providing a bounding box so don't have to do image detection at
        this step (Might want to keep image detection code for the future?)

        dlib.rectangle object
        '''
        if self.verbose:
            print("Processing {}.".format(imgPath))
        bgrImg = cv2.imread(imgPath)

        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
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
        alignedFace = self.align.align(self.img_dim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if self.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))
        
        if do_bb:
            name = os.path.basename(imgPath)
            name = os.path.join(new_dir, name)
            cv2.imwrite(name, alignedFace)
            print('saved image ', name)
        else:
            name = None

        start = time.time()
        rep = self.net.forward(alignedFace)
        if self.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")

        return rep, name


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

    def __init__(self, model_dir, torch_model='nn4.small2.v1.t7', args=None):
        ''' 
        All defaul values. 
        TODO: Might want to experiment with these later.
        '''
        if args is None:
            self.verbose = False
            self.img_dim = 96

        dlibModelDir = os.path.join(model_dir, 'dlib')
        openfaceModelDir = os.path.join(model_dir, 'openface')

        network_model = os.path.join(openfaceModelDir, torch_model)
        self.align = openface.AlignDlib(os.path.join(dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))

        self.net = openface.TorchNeuralNet(network_model, self.img_dim)

    def frame_to_faces(self, img_path, new_dir):
        '''
        @ret: [img_path1, img_path2,...]
        All the imgs have the same name as before, but we append a,b,c... to
        signify that these used to be the same frame.
        '''
        bgrImg = cv2.imread(img_path)

        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
 
        bbs = self.align.getAllFaceBoundingBoxes(rgbImg)
        if len(bbs) == 0:
            print('couldnt find multiple faces')
            # bb = self.align.getAllFaceBoundingBoxes(rgbImg)
            bb = self.align.getLargestFaceBoundingBox(rgbImg) 
            print(bb)
            raise Exception("Unable to find a face: {}".format(img_path))

        aligned_faces = []
        for bb in bbs:
            aligned_face = self.align.align(self.img_dim, rgbImg, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                raise Exception("Unable to align image: {}".format(img_path))

            aligned_faces.append((bb.center().x, aligned_face))

        aligned_faces = sorted(aligned_faces, key=lambda x: x[0])
        
        saved_names = []
        # Now write out the sorted aligned faces:
        for i, af in enumerate(aligned_faces):
            # save aligned face 
            af = af[1]
            name = os.path.basename(img_path)
            name = name.replace('.jpg', '')
            name += chr(ord('a') + i)
            name = os.path.join(new_dir, name)
            name += '.jpg'
            cv2.imwrite(name, aligned_face)
            saved_names.append(name)
            print('saved image ', name)

        return saved_names

    def get_rep(self, img_path, do_bb=False, new_dir=None):
        '''
        Slightly modified function from openface demos/classifier.py

        For a single image.
        @do_bb: if True, we treat full image as the bounding box of the face.
        else: Find all the faces in the image, and save those as new images.
        
        ret: [features], [names]
        '''
        if self.verbose:
            print("Processing {}.".format(img_path))
        bgrImg = cv2.imread(img_path)

        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        if self.verbose:
            print("  + Original size: {}".format(rgbImg.shape))

        start = time.time()
        
        if do_bb:
            # bbs = align.getAllFaceBoundingBoxes(rgbImg)
            bb = self.align.getLargestFaceBoundingBox(rgbImg)
        else:
            # treat the given image dimensions as the bb
            bounding_box = (0, 0, rgbImg.shape[0], rgbImg.shape[1]) 
            bb = _css_to_rect(bounding_box)
            bbs = [bb]

        start = time.time()
        aligned_face = self.align.align(self.img_dim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if aligned_face is None:
            raise Exception("Unable to align image: {}".format(img_path))
        if self.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))
        
        if do_bb:
            name = os.path.basename(img_path)
            name = os.path.join(new_dir, name)
            cv2.imwrite(name, aligned_face)
            print('saved image ', name)
        else:
            name = None

        start = time.time()
        rep = self.net.forward(aligned_face)
        if self.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))
            print("Representation:")
            print(rep)
            print("-----\n")

        return rep, name


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
            raise Exception("Unable to find a face: {}".format(img_path))

        orig_faces = []
        orig_img = cv2.imread(img_path)
        for bb in bbs:
            # if we can't align it, then don't waste everyones time by trying
            # to save it
            aligned_face = self.align.align(self.img_dim, rgbImg, bb,
                                      landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
            if aligned_face is None:
                continue

            crop_img = orig_img[bb.top():bb.bottom(),bb.left():bb.right()]
            orig_faces.append((bb.center().x, crop_img))

        orig_faces = sorted(orig_faces, key=lambda x: x[0])

        
        saved_names = []
        for i, af in enumerate(orig_faces):
            af = af[1]
            name = os.path.basename(img_path)
            name = name.replace('.jpg', '')
            name += chr(ord('a') + i)
            name = os.path.join(new_dir, name)
            name += '.jpg'
            cv2.imwrite(name, af)
            saved_names.append(name)
            print('saved image ', name)

        return saved_names

    def get_rep(self, img_path, do_bb=False, new_dir=None):
        '''
        Slightly modified function from openface demos/classifier.py

        For a single image.
        @do_bb: if True, we treat full image as the bounding box of the face.
        TODO: Just get rid of the do_bb option as I have separated the face
        detection stage. Just treat every input as the full image.
        
        ret: features, None
        '''
        if self.verbose:
            print("Processing {}.".format(img_path))
        bgrImg = cv2.imread(img_path)

        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(img_path))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        start = time.time()
        
        bb = self.align.getLargestFaceBoundingBox(rgbImg)

        start = time.time()
        aligned_face = self.align.align(self.img_dim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

        if aligned_face is None:
            raise Exception("Unable to align image: {}".format(img_path))
        if self.verbose:
            print("  + Face alignment took {} seconds.".format(time.time() - start))
        
        start = time.time()
        rep = self.net.forward(aligned_face)
        if self.verbose:
            print("  + OpenFace forward pass took {} seconds.".format(time.time() - start))

        return rep


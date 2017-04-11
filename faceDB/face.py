class Face():

    def __init__(self, img_path, video_id):
        '''
        Contains basic information about the face.

        @img_path: file path on disk
        @video_id: which video it is from. Can be useful as a feature, but we
        aren't using it right now.

        Can extend this to include more diverse features - such as additional
        audio/video features, other faces that it cannot be etc.
        '''

        self.img_path = img_path
        self.feature_vector = None
        self.video_id = video_id



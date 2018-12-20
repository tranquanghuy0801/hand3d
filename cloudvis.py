import json
import cv2
import base64
import numpy
import zmq

import numpy as np
import tensorflow as tf
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from mpl_toolkits.mplot3d import Axes3D
import operator
import pickle
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from pose.utils.FingerPoseEstimate import FingerPoseEstimate


#Use the math geometry to calculate the hand pose estimation

def predict_by_geometry(keypoint_coord3d_v, known_finger_poses, threshold):
    fingerPoseEstimate = FingerPoseEstimate(keypoint_coord3d_v)
    fingerPoseEstimate.calculate_positions_of_fingers(print_finger_info = True)
    obtained_positions = determine_position(fingerPoseEstimate.finger_curled, 
                                        fingerPoseEstimate.finger_position, known_finger_poses,
                                        threshold * 10)

    score_label = 'Undefined'
    if len(obtained_positions) > 0:
        max_pose_label = max(obtained_positions.items(), key=operator.itemgetter(1))[0]
        if obtained_positions[max_pose_label] >= threshold:
            score_label = max_pose_label
    
    print(obtained_positions)
    return score_label

#Predict the label for the image requested 

def predict_the_label(image_path,threshold):


    known_finger_poses = create_known_finger_poses()
    # network input
    image_tf = tf.placeholder(tf.float32, shape = (1, 240, 320, 3))
    hand_side_tf = tf.constant([[1.0, 1.0]])  # Both left and right hands included
    evaluation = tf.placeholder_with_default(True, shape = ())

     # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    net = ColorHandPose3DNetwork()

    hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
        keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)
    # initialize network
    net.init(sess)


    image_raw = image_path[:,:,:3]
    image_height, image_width = image_raw.shape[:2]
    if image_height < 240 or image_width < 320:
            image_raw = scipy.misc.imresize(image_raw,(2 * image_height,2 * image_width))
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)


    scale_v, center_v, keypoints_scoremap_v, \
                keypoint_coord3d_v = sess.run([scale_tf, center_tf, keypoints_scoremap_tf,\
                                            keypoint_coord3d_tf], feed_dict = {image_tf: image_v})

    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
    keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

    # post processing
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

    plot_hand_2d(coord_hw, image_raw)


    score_label = predict_by_geometry(keypoint_coord3d_v, known_finger_poses, threshold)

    label = score_label.split('-')[0]


    if label != 'Thumbs Up' or label == '':
        label = 'Not Thumbs Up'

    
    return label,image_raw

class NoDefault:
    pass

class Request:
    def __init__(self, data=str()):
        self.data = json.loads(data)

    def getValue(self, name, default=NoDefault()):
        if name in self.data:
            return self.data[name]

        if isinstance(default, NoDefault):
            raise Exception('Missing value for field: ' + name)

        return default

    def getValues(self):
        return self.data

    def getImage(self, name, default=NoDefault()):
        encoded = self.getValue(name, default)
        try:
            # probably in python2
            buf = base64.decodestring(encoded)
            flags = cv2.CV_LOAD_IMAGE_COLOR
        except TypeError:
            buf = base64.decodestring(str.encode(encoded))
            flags = cv2.IMREAD_ANYCOLOR

        
        return cv2.imdecode(numpy.frombuffer(buf, dtype=numpy.uint8),
                            flags)

class Response:
    def __init__(self):
        self.data = {}

    def addValue(self, name, value):
        self.data[name] = value

    def addValues(self, values):
        self.data.update(values)

    def addImage(self, name, image, extension=".jpg"):
        enim = cv2.imencode(extension, image)[1]
        encoded = base64.b64encode(enim)
        self.addValue(name, encoded.decode())

    def toJSON(self):
        return json.dumps(self.data)

class CloudVis:
    def __init__(self, port, host='cloudvis.qut.edu.au'):
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.connect("tcp://{0}:{1}".format(host, port))

    def __del__(self):
        self.socket.close()

    def run(self, process, data={}):
        print('Waiting')
        
        while True:
            try:
                req = Request(self.socket.recv())
                resp = Response()

                print('Request received')

                try:
                    process(req, resp, data)
                except Exception as e:
                    resp.addValues({'error': 1, 'message': str(e)})
                self.socket.send_string(resp.toJSON())
                print('Replied')

            except KeyboardInterrupt:
                print('Shutting down...')
                break

if __name__ == '__main__':
  def callback(req, res, data):
    # Get data
    img = req.getImage('input_image')

    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    label,image_processed = predict_the_label(img,0.45)


    if label == 'Thumbs Up':
        res.addValue('gesture', 'Thumbs Up')

    # Return data
    res.addValue('gesture', label)
    
    if req.getValue('render'):
        res.addImage('output_image', image_processed)


  cloudvis = CloudVis(port=6007)
  cloudvis.run(callback)
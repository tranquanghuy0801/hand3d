from __future__ import print_function, unicode_literals
import os 
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.image as mpimg
from werkzeug import secure_filename
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse
import operator
import pickle
import io
from PIL import Image
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from pose.utils.FingerPoseEstimate import FingerPoseEstimate
import flask

app = flask.Flask(__name__)



#Predict hand gesture by hand geometry

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

@app.route("/predict", methods = ['GET', 'POST'])
def upload():

    data = {"success": False}

    if flask.request.method == 'POST':
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)).convert('RGB')
            image = np.asarray(image)


            label, image_raw = predict_the_label(image,0.45)

            data["predictions"] = []

            r = {"label": label,"image": image_raw}

            data["predictions"].append(r)
            data["success"] =  True

    
    return flask.jsonify(data)



if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.run()
from __future__ import print_function, unicode_literals
import os 
import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from werkzeug import secure_filename
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse
import cv2
import operator
import pickle
import time
from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position, get_position_name_with_pose_id
from pose.utils.FingerPoseEstimate import FingerPoseEstimate

from flask import Flask, render_template, request, redirect, url_for

PROJECT_HOME = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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

    tf.reset_default_graph()

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


    image_raw = scipy.misc.imread(image_path)[:,:,:3]
    image_height, image_width = image_raw.shape[:2]
    if image_height < 240 or image_width < 320:
            image_raw = scipy.misc.imresize(image_raw,(2 * image_height,2 * image_width))
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    keypoint_coord3d_v = sess.run(keypoint_coord3d_tf, feed_dict = {image_tf: image_v})

    score_label = predict_by_geometry(keypoint_coord3d_v, known_finger_poses, threshold)

    label = score_label.split('-')[0]

    if label != 'Thumbs Up':
        label = 'Not Thumbs Up'
    return label

@app.route("/")
def template_test():
    return render_template('template.html', label='', image_name = 'teaser.png')

@app.route("/predict", methods = ['GET', 'POST'])
def upload():
    start = time.time()
    target = os.path.join(PROJECT_HOME, 'images/')
    # target = os.path.join(APP_ROOT, 'static/')
    print(target)
    if not os.path.isdir(target):
            os.mkdir(target)
    else:
        print("Couldn't create upload directory: {}".format(target))
    print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print(upload)
        print("{} is the file name".format(upload.filename))
        filename = upload.filename
        destination = "/".join([target, filename])
        print ("Accept incoming file:", filename)
        print ("Save it to:", destination)
        upload.save(destination)
        label = predict_the_label(destination,0.45)
    # return send_from_directory("images", filename, as_attachment=True)
    end = time.time()
    print(end-start)
    return render_template("template.html",label = label,image_name=filename)

    # return send_from_directory("images", filename, as_attachment=True)

from flask import send_from_directory
      
@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    app.run(port=4555, debug=True)
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from mpl_toolkits.mplot3d import Axes3D
import argparse
import cv2

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_2d, plot_hand_3d
from pose.DeterminePositions import create_known_finger_poses, determine_position
from pose.utils.FingerPoseEstimate import FingerPoseEstimate

def parse_args():
	parser = argparse.ArgumentParser(description='Detect objects in the video or still images')
	parser.add_argument('data_path', help = 'Path of folder containing images', type = str)
	parser.add_argument('--output-path', dest = 'output_path', type = str, default = None,
						help='Path of folder where to store the evaluation result')
	parser.add_argument('--plot-fingers', dest = 'plot_fingers', help = 'Should fingers be plotted. (1 = Yes, 0 = No)', 
						default = 1, type = int)
	parser.add_argument('--thresh', dest = 'threshold', help = 'Threshold of confidence level(0-1)', default = 0.6,
	                    type = float)
	args = parser.parse_args()
	return args

def prepare_input(data_path, output_path):
	data_path = os.path.abspath(data_path)
	data_files = os.listdir(data_path)
	data_files = [os.path.join(data_path, data_file) for data_file in data_files]

	if output_path is None:
		output_path = data_path
	else:
		output_path = os.path.abspath(output_path)

	return data_files, output_path

if __name__ == '__main__':
	args = parse_args()
	data_files, output_path = prepare_input(args.data_path, args.output_path)
	known_finger_poses = create_known_finger_poses()

	# network input
	image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
	hand_side_tf = tf.constant([[1.0, 1.0]])  # Both left and right hands included
	evaluation = tf.placeholder_with_default(True, shape=())

	# build network
	net = ColorHandPose3DNetwork()
	hand_scoremap_tf, image_crop_tf, scale_tf, center_tf,\
		keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

	# Start TF
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
	sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

	# initialize network
	net.init(sess)

	# Feed image list through network
	for img_name in data_files:
		image_raw = scipy.misc.imread(img_name)
		image_raw = scipy.misc.imresize(image_raw, (240, 320))
		image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

		if args.plot_fingers == 1:
			scale_v, center_v, keypoints_scoremap_v, \
				keypoint_coord3d_v = sess.run([scale_tf, center_tf, keypoints_scoremap_tf,\
											keypoint_coord3d_tf], feed_dict = {image_tf: image_v})

			#hand_scoremap_v = np.squeeze(hand_scoremap_v)
			#image_crop_v = np.squeeze(image_crop_v)
			keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
			keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

			# post processing
			#image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
			coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
			coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

			plot_hand_2d(coord_hw, image_raw)

		else:
			keypoint_coord3d_v = sess.run(keypoint_coord3d_tf, feed_dict = {image_tf: image_v})

		fingerPoseEstimate = FingerPoseEstimate(keypoint_coord3d_v)
		fingerPoseEstimate.calculate_positions_of_fingers(image_raw)
		obtained_positions = determine_position(fingerPoseEstimate.finger_curled, 
											fingerPoseEstimate.finger_position, known_finger_poses,
											args.threshold * 10)

		score_label = 'Undefined'
		max_score = 0.0
		for k, v in obtained_positions.items():
			if v > max_score:
				max_score = v
				score_label = '{}'.format(k)
				
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(image_raw, score_label, (10, 200), font, 1.0, (255, 0, 0), 2, cv2.LINE_AA)

		file_name = os.path.basename(img_name)
		file_name_comp = file_name.split('.')
		file_save_path = os.path.join(output_path, "{}_out.png".format(file_name_comp[0]))
		mpimg.imsave(file_save_path, image_raw)

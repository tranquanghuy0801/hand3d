# Hand Gesture Pose Classification Network

This is a network trained to classify various types of hand gesture pose images.

You can read the detailed post about the approach used in this project in my [Medium post](https://medium.com/@prasad.pai/classification-of-hand-gesture-pose-using-tensorflow-30e83064e0ed).

The base network of this project is the forked version of [Hand3d](https://github.com/lmb-freiburg/hand3d). 

The trained network at present runs for classifying 7 hand gesture poses. The selected hand gesture poses are Simple thumbs up, Victory, Spock, Okay, Pointing up, I love you and Thumbs up right. The evaluation of the trained network will be done on images present inside a folder. This is the result of the network shown below.

![Classified Poses](https://user-images.githubusercontent.com/13696749/36244856-0e72a27e-124e-11e8-9c06-52c04d027386.png)

This is an example of classification shown along side with plottings of landmarks.

![Classified Poses with Landmarks](https://user-images.githubusercontent.com/13696749/36244906-4a824134-124e-11e8-86cb-0b8494bba1b9.png)

## Instructions to run the code
For the detailed instructions on how to run Hand3d network on its own, please go through their [ReadMe](https://github.com/lmb-freiburg/hand3d) page. To run Hand Gesture Pose Classification, you will have to download the weights file. Download links are present in Hand3d ReadMe page.

You can run Hand Gesture Pose Classification Network using three methodologies. 

To run using geometry formed(details explained later) by fingers, run the below instruction.

	python evaluate_pose.py ./pose/test_data

To run using the neural network, run the below instruction.

	python evaluate_pose.py ./pose/test_data --solve-by=1 --pb-file=./pose/learned_models/graph.pb

To run using the Support Vector Machine, run the below instruction.

	python evaluate_pose.py ./pose/test_data --solve-by=2 --svc-file=./pose/learned_models/svc.pickle

### Explanation of options present
During execution of `evaluate_pose.py`, there are several [options](https://github.com/Prasad9/Classify-HandGesturePose/blob/master/evaluate_pose.py#L20) available for configuration.
- `data_path`: This is the only compulsory option. Submit your path of folder containing images which needs to be evaluated.
- `--output-path`: Provide the path where output needs to be stored. If this option is not provided, output will be stored in input folder.
- `--plot-fingers`: If in output you wish to plot the landmarks obtained from Hand3d, make this option as 1. Otherwise set it to 0. By default it is at 1.
- `--solve-by`: There are three ways in which network can be evaluated. 1 (default) = Makes use of information obtained from curl and directional orientations of finger. 2 = Neural Network. 3 = Support Vector Machine.
- `--thresh`: This option is used if you are solving by either geometrical formation(1) or neural network(2). The value in this corresponds to threshold of confidence level. Set value in range from 0 to 1. Default value is 0.45.
- `--pb-file`: If solving by neural network, set the path of your trained frozen model in this option.
- `--svc-file`: If solving by SVM, set path of your trained svc pickle file in this.

## Creating your own hand gesture pose
In order to create your own hand gesture pose, you will have to edit [DeterminePositions.py](https://github.com/Prasad9/Classify-HandGesturePose/blob/master/pose/DeterminePositions.py) file. For each of the five fingers present in human hand, you will have to give the information on curl and directional orientation. Check these below images to understand various types of curls and directional orientations.
![Curled types](https://user-images.githubusercontent.com/13696749/36244978-a8a1e06c-124e-11e8-887d-3e2b1c02d813.png)
![Directional types](https://user-images.githubusercontent.com/13696749/36244950-7f738c40-124e-11e8-8169-7033a8625c75.png)

Apart from the above information, you even need to give the confidence level for each of the pose and directional orientation. Set the confidence level from 0 to 1 for each of the pose and directional orientation. Lastly, you need to set a unique position_id for each of the pose. An example of one such pose is shown below:

    ####### 1 Simple Thumbs up
    simple_thumbs_up = FingerDataFormation()
    simple_thumbs_up.position_name = 'Simple Thumbs Up'
    simple_thumbs_up.curl_position = [
        [FingerCurled.NoCurl],   # Thumb
        [FingerCurled.FullCurl], # Index
        [FingerCurled.FullCurl], # Middle
        [FingerCurled.FullCurl], # Ring
        [FingerCurled.FullCurl]  # Little
    ]
    simple_thumbs_up.curl_position_confidence = [
        [1.0], # Thumb
        [1.0], # Index
        [1.0], # Middle
        [1.0], # Ring
        [1.0]  # Little
    ]
    simple_thumbs_up.finger_position = [
        [FingerPosition.VerticalUp, FingerPosition.DiagonalUpLeft, FingerPosition.DiagonalUpRight], # Thumb
        [FingerPosition.HorizontalLeft, FingerPosition.HorizontalRight], # Index
        [FingerPosition.HorizontalLeft, FingerPosition.HorizontalRight], # Middle
        [FingerPosition.HorizontalLeft, FingerPosition.HorizontalRight], # Ring
        [FingerPosition.HorizontalLeft, FingerPosition.HorizontalRight] # Little
    ]
    simple_thumbs_up.finger_position_confidence = [
        [1.0, 0.25, 0.25], # Thumb
        [1.0, 1.0], # Index
        [1.0, 1.0], # Middle
        [1.0, 1.0], # Ring
        [1.0, 1.0]  # Little
    ]
    simple_thumbs_up.position_id = 0
    known_finger_poses.append(simple_thumbs_up)

## Training your own network:
### 1) Geometry (Curl and directional orientations)
Add all your hand gesture poses in `DeterminePositions.py` file as explained in above section. The evaluation will happen based on determination of curls and directions of fingers and hence, there is no training involved in this method. 

### 2) Neural Network
If you don't have data with images of your required hand gesture pose, you can make use of the tool present in this repository. Create separate videos with individual type of poses lasting for about 10 to 12 seconds. Ensure to add small amounts of distortions in your pose. An example video has been [enclosed](https://github.com/Prasad9/Classify-HandGesturePose/blob/master/pose/video/okay.mp4). Next to process this video, I have written in OpenCV and MoviePy. However, the OpenCV method couldn't be tested by me as I am having certain installation issues, but the usage is very similar to that of MoviePy. Using MoviePy, you can run the code as:

	python pose/tools/ProcessFramesMoviePy.py ./pose/video/okay.mp4 5 

The options required to run this code are:
- `video_path`: (Required) Give the path where the video is present.
- `pose_no`: (Required) The `position_id` you have set for the position you are evaluating in `DeterminePositions.py` file. 
- `--output-path`: Path of folder where to store the output csv file and video containing the detected frames. Default location is the path of input video.
- `--thresh`: Threshold of the confidence level. Default is 0.45.
- `--save-video`: (Available in OpenCV method) Set 1 if you would like to save the output video. Default is no, set as 0.

The above tool evaluates based on the positions mentioned in `DeterminePositions.py` file and hence, before running this tool, you have to add the required hand gestures in that file.

With keypoints of Hand3d obtained in separate CSV files for each of the hand gesture pose, training on neural network can be done by running the below instruction as:

	python pose/training/NeuralNetwork.py './pose1.csv, ./pose2.csv'

Take care to pass the CSV files in the order in which `position_id` has been set in `DeterminePositions.py`. I encourage you to look into various options present in [NeuralNetwork.py](https://github.com/Prasad9/Classify-HandGesturePose/blob/master/pose/training/NeuralNetwork.py#L15) file and pass accordingly to suit your training.
 
### 3) Support Vector machine
The data with keypoints of Hand3d can also be used to train a support vector classifier. Run the below instruction

	python pose/training/SVM.py './pose1.csv, ./pose2.csv'

Take care to pass the CSV files in the order in which `position_id` has been set in `DeterminePositions.py`. I encourage you to look into various options present in [SVM.py](https://github.com/Prasad9/Classify-HandGesturePose/blob/master/pose/training/SVM.py#L11) file and pass accordingly to suit your training.

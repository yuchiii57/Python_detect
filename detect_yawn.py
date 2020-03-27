import sys
sys.path.append('c:/users/peter/anaconda3/lib/site-packages')
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

#2
def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the three sets of
	# vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[2], mouth[10])
        B = dist.euclidean(mouth[3], mouth[9])
        C = dist.euclidean(mouth[4], mouth[8])
	# compute the euclidean distance between the horizontal
	# mouth landmark (x, y)-coordinates
        D = dist.euclidean(mouth[0], mouth[6])
        
	# compute the mouth aspect ratio
        mar = (A + B + C) / (3.0 * D)

	# return the mouth aspect ratio
        return mar

#3
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

#4
# define two constants, one for the mouth aspect ratio to indicate
# yawn and then a second constant for the number of consecutive
# frames the mouth must be above the threshold
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 10
 
# initialize the frame counters and the total number of yawn and time
COUNTER = 0
TOTAL = 0
START_TIME=time.time()

#5
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

#6
# grab the indexes of the facial landmarks for the mouth, respectively
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

#main
# loop over frames from the video stream
while True:
	# if this is a file video stream, then we need to check if
	# there any more frames left in the buffer to process
	if fileStream and not vs.more():
		break
 
	# grab the frame from the threaded video file stream, resize
	# it, and convert it to grayscale
	# channels)
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio 
		mouth = shape[Start:End]
 
		# average the mouth aspect ratio for mouth
		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the mouth, then visualize mouth
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

		# check to see if the mouth aspect ratio is above the yawn
		# threshold, and if so, increment the yawn frame counter
		if mar > MOUTH_AR_THRESH:
			COUNTER += 1
 
		# otherwise, the mouth aspect ratio is not above the yawn
		# threshold
		else:
			# if the mouth was opened for a sufficient number of
			# then increment the total number of yawns
			if COUNTER >= MOUTH_AR_CONSEC_FRAMES:
				TOTAL += 1
 
			# reset the mouth frame counter
			COUNTER = 0

		test_time=time.time()-START_TIME
		# draw the total number of blinks on the frame along with
		# the computed mouth aspect ratio for the frame
		cv2.putText(frame, "Yawns: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Time: {:.3f}".format(test_time), (150, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()

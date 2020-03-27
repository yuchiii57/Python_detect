import sys
sys.path.append('c:/users/peter/anaconda3/lib/site-packages')
###1.import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from datetime import datetime
from keras.models import model_from_json
from keras.optimizers import SGD
from scipy.ndimage import zoom
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2
import xlsxwriter
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
###2.define function
def extract_face_features(gray, detected_face, offset_coefficients):
        (x, y, w, h) = detected_face
        #print x , y, w ,h
        horizontal_offset = np.int(np.floor(offset_coefficients[0] * w))
        vertical_offset = np.int(np.floor(offset_coefficients[1] * h))
        extracted_face = gray[y+vertical_offset:y+h, 
                          x+horizontal_offset:x-horizontal_offset+w]
        #print extracted_face.shape
        new_extracted_face = zoom(extracted_face, (48. / extracted_face.shape[0], 
                                               48. / extracted_face.shape[1]))
        new_extracted_face = new_extracted_face.astype(np.float32)
        new_extracted_face /= float(new_extracted_face.max())
        return new_extracted_face
def detect_face(frame):
        cascPath = "./models/haarcascade_frontalface_default.xml"
        faceCascade = cv2.CascadeClassifier(cascPath)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(48, 48),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        return gray, detected_faces
def sound_alarm(path="./alarm.wav"):
	# play an alarm sound
	playsound.playsound(path)

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
 
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])
 
	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)
 
	# return the eye aspect ratio
	return ear

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

###3.define models
model = model_from_json(open('./models/Face_model_architecture.json').read())
#model.load_weights('_model_weights.h5')
model.load_weights('./models/Face_model_weights.h5')
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

###4.define constants
# define filename
filename=datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f")
# define .xlsx
workbook = xlsxwriter.Workbook(filename+".xlsx")
# define sheet
sheet1 = workbook.add_worksheet("Sheet1")
sheet2 = workbook.add_worksheet("Sheet2")
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5
EYE_AR_CONSEC_FRAMES_DROWSINESS=48

# define two constants, one for the mouth aspect ratio to indicate
# yawn and then a second constant for the number of consecutive
# frames the mouth must be above the threshold
MOUTH_AR_THRESH = 0.75
MOUTH_AR_CONSEC_FRAMES = 10

# initialize time
START_TIME=time.time()

# initialize the frame counters and the total number of blinks and yawns 
COUNTER_BLINK = 0
TOTAL_BLINK = 0
COUNTER_YAWN = 0
TOTAL_YAWN = 0

#initialize a boolean used to indicate if the alarm is going off
ALARM_ON = False

###5.initialize dlib
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# grab the indexes of the facial landmarks for eyes and mouth, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(Start, End) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

###6.start video
# start the video stream thread
print("[INFO] starting video stream thread...")
#vs = FileVideoStream(args["video"]).start()
#fileStream = True
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
fileStream = False
time.sleep(1.0)

###7.main loop
# loop over frames from the video stream
i=0
j=0
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
	gray, detected_faces = detect_face(frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	face_index = 0
	text=""
	for face in detected_faces:
		(x, y, w, h) = face
		if w > 100:
            # draw rectangle around face 
			cv2.rectangle(frame, (x, y), (x+w, y+h), (255,0,0), 2)
            # extract features
			extracted_face = extract_face_features(gray, face, (0.075, 0.05)) #(0.075, 0.05)
            # predict smile
			prediction_result = model.predict_classes(extracted_face.reshape(1,48,48,1))
            # draw extracted face in the top right corner
			frame[face_index * 48: (face_index + 1) * 48, -49:-1, :] = cv2.cvtColor(extracted_face * 255, cv2.COLOR_GRAY2RGB)
            # annotate main image with a label
			if prediction_result == 3:
                                text="Happy"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 2)
			elif prediction_result == 0:
                                text="Angry"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			elif prediction_result == 1:
                                text="Disgust"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			elif prediction_result == 2:
                                text="Fear"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			elif prediction_result == 4:
                                text="Sad"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			elif prediction_result == 5:
                                text="Surprise"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			else :
                                text="Neutral"
                                cv2.putText(frame, text,(x,y), cv2.FONT_HERSHEY_DUPLEX, 1,  (255,255,255), 2)
			test_time=time.time()-START_TIME
			test_time=round(test_time,2)
			sheet1.write(i,0,test_time)
			#sheet1.write(i,0,datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f"))
			sheet1.write(i,1,prediction_result)
			sheet1.write(i,2,text)
			i+=1
			#with open(filename+'.txt', 'a') as f:
				#f.write('{},{}\n'.format(datetime.now().strftime("%Y-%m-%d %H-%M-%S-%f") ,prediction_result))
            # increment counter
			face_index += 1
	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
 
		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# extract the mouth coordinates, then use the
		# coordinates to compute the mouth aspect ratio 
		mouth = shape[Start:End]
		
		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# average the mouth aspect ratio for mouth
		mar = mouth_aspect_ratio(mouth)

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# compute the convex hull for the mouth, then visualize mouth
		mouthHull = cv2.convexHull(mouth)
		cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)

                # check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER_BLINK += 1

			if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES_DROWSINESS:
                                # if the alram is not on, turn it on
                                if not ALARM_ON:
                                        ALARM_ON = True
                                        # check to see if an alarm file was supplied, 
                                        # and if so, start a thread to have the alarm
                                        # sound played in the background
                                        t = Thread(target=sound_alarm)
                                        t.deamon = True
                                        t.start()
                                        
				# draw an alarm on the frame
                                cv2.putText(frame, "DROWSINESS ALERT!", (10, 260),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER_BLINK >= EYE_AR_CONSEC_FRAMES:
				TOTAL_BLINK += 1
			                                                
			# reset the eye frame counter
			COUNTER_BLINK = 0
			ALARM_ON = False

		# check to see if the mouth aspect ratio is above the yawn
		# threshold, and if so, increment the yawn frame counter
		if mar > MOUTH_AR_THRESH:
			COUNTER_YAWN += 1
 
		# otherwise, the mouth aspect ratio is not above the yawn
		# threshold
		else:
			# if the mouth was opened for a sufficient number of
			# then increment the total number of yawns
			if COUNTER_YAWN >= MOUTH_AR_CONSEC_FRAMES:
				TOTAL_YAWN += 1
				
			# reset the mouth frame counter
			COUNTER_YAWN = 0

		test_time=time.time()-START_TIME
		test_time=round(test_time,2)
		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		# draw the total number of yawns on the frame along with
		# the computed mouth aspect ratio for the frame
		cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINK), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Time: {:.2f}".format(test_time), (130, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "Yawns: {}".format(TOTAL_YAWN), (10, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 300),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		sheet2.write(j,0,test_time)
		sheet2.write(j,1,ear)
		sheet2.write(j,2,mar)
		sheet2.write(j,3,TOTAL_BLINK)
		sheet2.write(j,4,TOTAL_YAWN)
		j +=1
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
workbook.close()

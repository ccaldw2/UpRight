from scipy.spatial import distance
from imutils import face_utils
import numpy as np
import time
import dlib
import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A+B) / (2*C)
    return ear

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

face_rectangle_BGRT = (255,0,255,4)
eye_contours_BGR = (0,255,255)
eye_contours_T = 2

video_capture = cv2.VideoCapture(0)


EYE_ASPECT_RATIO_THRESHOLD = 0.25
EYE_ASPECT_RATIO_CONSEC_FRAMES = 10
DROWSY_LONGEST_FRAMES = 0
COUNTER = 0
DROWSY_CONSEC_DETECTIONS = 0
NEW_DETECTION = True

while(True):
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),face_rectangle_BGRT)

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, eye_contours_BGR, eye_contours_T)
        cv2.drawContours(frame, [rightEyeHull], -1, eye_contours_BGR, eye_contours_T)

        if(eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD):
            COUNTER += 1
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                cv2.putText(frame, "You're Tired!", (10,200), cv2.FONT_HERSHEY_TRIPLEX, 2.0, (0,0,255), 2)
                if COUNTER > DROWSY_LONGEST_FRAMES:
                    DROWSY_LONGEST_FRAMES = COUNTER

                if NEW_DETECTION:
                    DROWSY_CONSEC_DETECTIONS += 1;
                    NEW_DETECTION = False
        else:
            COUNTER = (COUNTER - 1) if (COUNTER != 0) else 0
            if COUNTER <= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                NEW_DETECTION = True

    #Show video feed
    detections_count_message = ('No. times drift detected: ' + str(DROWSY_CONSEC_DETECTIONS))
    detections_longest_message = ('longest drift time: ' +str(float(DROWSY_LONGEST_FRAMES)/30.0) + ' sec')
    cv2.putText(frame, (detections_count_message), (20,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
    cv2.putText(frame, (detections_longest_message), (20,440), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()

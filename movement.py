import numpy as np
import cv2
from playsound import playsound
import time
import datetime

alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

#Setting up a frame parametres
count = 1
prev_frame = None
cap = cv2.VideoCapture(0)
frame_size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
playsound("sound_on.mp3")
detection = False
timer_started = False
detection_stopped_time = 0
while True:
    #Time after which recording turns off if no next movements were detected
    time_to_record = 5
    count += 1
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if count % 2 == 0:
        prep_frame = cv2.GaussianBlur(grayImg, ksize=(5,5), sigmaX=0)
    if prev_frame is None:
        prev_frame = prep_frame
        continue
    #Face parametres
    face = haar_cascade.detectMultiScale(grayImg,1.3,4)
    #Difference between previous frame and current
    dif_frame = cv2.absdiff(prev_frame, prep_frame)
    #Setting previous frame to be current frame
    prev_frame = prep_frame
    #Preparing frame for better detection
    dif_frame = cv2.dilate(dif_frame, np.ones((5, 5)),1)
    thresh = cv2.threshold(dif_frame, 20, 255, type=cv2.THRESH_BINARY)[1]
    #Taking contours of the movements
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #Finding the contour with the biggest shape to draw only it
    val = 0
    (x, y, w, h) = (0,0,0,0)
    g=0
    for i in contours:
        max_val = cv2.contourArea(i)
        #Skip small objects
        if max_val < 500:
            continue
        g = 1
        if max_val>val:
            (x, y, w, h) = cv2.boundingRect(i)
            val = max_val
    cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
    #Drawing face
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x,y), ((x+w),(y+h)), (0,0,255), 2)

    #Check if movement/face was detected
    if g + len(face) > 0:
        #If movement was already detected before
        if detection:
            timer_started = False
        #If new movement was detected
        else:
            detection = True
            date = datetime.datetime.now().strftime("%d-%m-%Y-%M-%S")
            out = cv2.VideoWriter(f"{date}.mp4", fourcc, 20, frame_size)
            print("Recording...")
    #If no movement was detected but it was detected before
    elif detection:
        #If the recording already in proccess
        if timer_started:
            #If recording without a movement took more than 5 sec
            if time.time() - detection_stopped_time >= time_to_record:
                detection = False
                timer_started = False
                out.release()
                print("-------------\nFinished")
        #Start timer
        else:
            timer_started = True
            detection_stopped_time = time.time()

    if detection:
        out.write(frame)
    
    
    cv2.imshow("Window", frame)
    if cv2.waitKey(1) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
playsound("off.mp3")


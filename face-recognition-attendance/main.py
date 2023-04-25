import cv2
import numpy as np
import face_recognition
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

#load known faces
hkp_image = face_recognition.load_image_file("faces/hkp.jpeg")
hkp_encoding = face_recognition.face_encodings(hkp_image)[0]
tapas_image = face_recognition.load_image_file("faces/tapas.jpeg")
tapas_encoding = face_recognition.face_encodings(tapas_image)[0]

known_face_encoding = [hkp_encoding, tapas_encoding]
known_face_name = ["hkp", "tapas"]
#list of expected student
students = known_face_name.copy()

face_locations = []
face_encodings = []

#get the current date and time
now = datetime.now()
current_date= now.strftime("%m/%d/%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame=cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    #face recognize
    face_locations = face_recognition.face_locations(rgb_small_frame)

    face_encodings=face_recognition.face_encodings(rgb_small_frame, face_locations)
    for face_encodings in face_encodings :
        matches = face_recognition.compare_faces(known_face_encoding, face_encodings)
        face_distance = face_recognition.face_distance(known_face_encoding,face_encodings)
        best_match_index= np.argmin(face_distance)

        if(matches[best_match_index]):
            name= known_face_name[best_match_index]

            #add the text if the person is present
            if name in known_face_name:
                font= cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText =(10,100)
                fontScale = 1.5
                fontColor= (255, 0 , 0)
                thickness =3
                lineType = 2
                cv2.putText (frame, name + "present", bottomLeftCornerOfText ,font,fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

    cv2.imshow("attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllwindows()
f.close()

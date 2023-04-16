import face_recognition
import numpy as np
import cv2
import csv

from datetime import datetime


video_capture = cv2.VideoCapture(0)

#load known faces

ali_image = face_recognition.load_image_file('faces/ali.jpg')

ali_enconding = face_recognition.face_encodings(ali_image)[0]

rohan_image = face_recognition.load_image_file('faces/rohan.jpg')

rohan_encoding = face_recognition.face_encodings(rohan_image)[0]


known_face_encodings = [ali_enconding, rohan_encoding]

known_face_names = ['ali', 'rohan']


#list of expected student names

students = known_face_names.copy()


face_locations = []

face_encodings = []


#get the current date and time

now = datetime.now()
current_date = now.strftime('%Y-%m-%d')

#now csv module

f = open(f"{current_date}.csv",'w+', newline='')
lnwriter = csv.writer(f)


#now the magic begins

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)


    #recognize faces

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_matchIdx = np.argmin(face_distance)

        if(matches[best_matchIdx]):
            name = known_face_names[best_matchIdx]

        # add the text if person is present

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX

            bottom_left_corner_text = (10, 100)

            fontscale = 1.5

            fontColor = (255, 0, 0)

            thickness = 3

            lineType = 2

            cv2.putText(frame, name + " Present", bottom_left_corner_text, font, fontscale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime('%H-%M%S')
                lnwriter.writerow([name, current_time])

    cv2.imshow('attendance',frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
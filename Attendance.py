import csv

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# from PIL import ImageGrab

# Set the path to the folder containing images
path = 'Images'
images = []
classNames = []

# List all files in the specified path
myList = os.listdir(path)
print(myList)
# Loop through each file in the path
for cl in myList:
    # Read the image using OpenCV
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # Extract class names from file names (removing file extension)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find face encodings for a list of images
def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert image to RGB (face_recognition uses RGB format)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Get face encodings for the image
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


# def markAttendance(name):
#     with open('Attendance.csv', 'r+') as f:
#         myDataList = f.readlines()
#         nameList = []
#         for line in myDataList:
#             entry = line.split(',')
#             nameList.append(entry[0])
#         if name not in nameList:
#             now = datetime.now()
#             dtString = now.strftime('%H:%M:%S')
#             f.writelines(f'n{name},{dtString}')

# Function to mark attendance in a CSV file
def markAttendance(name):
    file_path = 'Attendance.csv'

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)

        # Check if the file is empty, and write header if needed
        if f.tell() == 0:
            writer.writerow(['Name', 'Date', 'Time'])

        now = datetime.now()
        date_string = now.strftime('%Y-%m-%d')
        time_string = now.strftime('%H:%M:%S')
        # Write the attendance record to the CSV file
        writer.writerow([name, date_string, time_string])


#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

# Find face encodings for known images
encodeListKnown = findEncodings(images)
print('Encoding Complete')
# Open a webcam
cap = cv2.VideoCapture(0)

# Main loop for face recognition
while True:
    success, img = cap.read()
    # img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the current frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop through each face in the current frame
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        # If a match is found, mark attendance and display information on the image
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) #Putting rectangle in the webcam
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2) #Putting text under the rectangle
            markAttendance(name)

    # Display the webcam feed with rectangles and text
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

    ### Pressing "q" to close the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

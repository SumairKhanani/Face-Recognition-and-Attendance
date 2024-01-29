import cv2
import numpy as np
import face_recognition

# Load the image of Bill Gates and convert it from BGR to RGB
imgBill = face_recognition.load_image_file('Images/Bill Gate.jpg')
imgBill = cv2.cvtColor(imgBill, cv2.COLOR_BGR2RGB)

# Load the test image of Bill Gates and convert it from BGR to RGB
imgTest = face_recognition.load_image_file('Images/Bill gate.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
q
# Face detection and encoding for the image of Bill Gates
faceLoc = face_recognition.face_locations(imgBill)[0]
encodeBill = face_recognition.face_encodings(imgBill)[0]
# Draw a rectangle around the detected face in the image of Bill Gates
cv2.rectangle(imgBill, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

# Face detection and encoding for the test image of Bill Gates
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
# Draw a rectangle around the detected face in the test image
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

# Compare the face encodings and calculate the face distance
results = face_recognition.compare_faces([encodeBill], encodeTest)
faceDis = face_recognition.face_distance([encodeBill], encodeTest)
print(results, faceDis)
# Display the results on the test image
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Elon Musk', imgBill)
cv2.imshow('Elon Test', imgTest)
cv2.waitKey(0)
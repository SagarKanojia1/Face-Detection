import numpy as np
import cv2
import os
import face_recognition as fr

print(fr)

# Give path to the image which you want to test
test_img = cv2.imread(r'testing_path')

faces_detected, gray_img = fr.faceDetection(test_img)
print("face Detected: ", faces_detected)

faces,faceID=fr.labels_for_training_data(r'tarin_images') #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)

# Give path of where trainingData.yml is saved
face_recognizer.save(r'trainingData_path')

# Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.
name = {0: "Sagar"}

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y+h, x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("Confidence :", confidence)
    print("label :", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img, (1000, 700))

cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

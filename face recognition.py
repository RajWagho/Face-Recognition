import cv2
import numpy as np
import face_recognition

img_harry = face_recognition.load_image_file('harry styles img 1.jpg')
img_harry = cv2.cvtColor(img_harry,cv2.COLOR_BGR2RGB)

img_harry_test = face_recognition.load_image_file('harry styles img 2.jpg')
img_harry_test = cv2.cvtColor(img_harry_test,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(img_harry)[0]
encodeharry = face_recognition.face_encodings(img_harry)[0]
cv2.rectangle(img_harry,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloc_test = face_recognition.face_locations(img_harry_test)[0]
encodeharry_test = face_recognition.face_encodings(img_harry_test)[0]
cv2.rectangle(img_harry_test,(faceloc_test[3],faceloc_test[0]),(faceloc_test[1],faceloc_test[2]),(25,0,255),2)

compare_results = face_recognition.compare_faces([encodeharry],encodeharry_test)
distance_results = face_recognition.face_distance([encodeharry],encodeharry_test)
cv2.putText(img_harry_test,f'{compare_results} {round(distance_results[0],2)} Harry Styles',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),2)
print(compare_results,distance_results)


cv2.imshow('Harry styles 1',img_harry)
cv2.imshow('Harry styles',img_harry_test)

cv2.waitKey(0)




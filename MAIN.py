#importing libraries.
import cv2
import mediapipe as mp
import tensorflow
from cvzone.ClassificationModule import Classifier
import numpy as np

#creating object
cap=cv2.VideoCapture(0)
mp_face_detect= mp.solutions.face_detection
faceDetect = mp_face_detect.FaceDetection(model_selection=1)
mp_draw = mp.solutions.drawing_utils

#creating classifier object
classifier=Classifier("model\keras_model.h5","model\labels.txt")

labels=["with mask" , "without mask"]

while True:

    try:
        success , img = cap.read()
        # img=cv2.resize(img,(244,244))
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        faces=faceDetect.process(imgRGB)
        # print(faces)

        if faces.detections:

            for id,face in enumerate(faces.detections):
                bbox=face.location_data.relative_bounding_box
                # print(bbox)
                height,width=img.shape[:2]

                x,y,w,h=bbox.xmin*width ,bbox.ymin*height ,bbox.width*width ,bbox.height*height
                # print(x,y,h,w)
                # print(id+1)

                img=cv2.rectangle(img,(int(x),int(y)),(int(x+w),int(y+h)),(0,234,0),1)

                img=cv2.putText(img,"id={}".format(id+1),(int(x),int(y-30)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,234,0),2)
                img=cv2.line(img,(int(x),int(y)),(int(x+25),int(y)),(0,234,0),4)
                img=cv2.line(img,(int(x),int(y)),(int(x),int(y+25)),(0,234,0),4)
                img=cv2.line(img,(int(x+w),int(y)),(int(x+w-25),int(y)),(0,234,0),4)
                img=cv2.line(img,(int(x+w),int(y)),(int(x+w),int(y+25)),(0,234,0),4)
                img=cv2.line(img,(int(x),int(y+h)),(int(x),int(y+h-25)),(0,234,0),4)
                img=cv2.line(img,(int(x),int(y+h)),(int(x+25),int(y+h)),(0,234,0),4)
                img=cv2.line(img,(int(x+w),int(y+h)),(int(x+w-25),int(y+h)),(0,234,0),4)
                img=cv2.line(img,(int(x+w),int(y+h)),(int(x+w),int(y+h-25)),(0,234,0),4)


                imgCrop=img[int(y-50):int(y+h+51),int(x-50):int(x+w+51)]

                imgCrop = cv2.resize(imgCrop,(224,224))

                prediction,index=classifier.getPrediction(imgCrop)


                color=[(0,234,0),(43,30,238)]
                img = cv2.putText(img,labels[index],(int(x),int(y+h+10)),cv2.
                FONT_HERSHEY_COMPLEX,1,color[index],2)

        else:
            cv2.putText(img,"No person" , (10,50),cv2.FONT_HERSHEY_SIMPLEX,1,(39,30,340),4)


        
        
    except Exception as e:
        print("Error occured {}".format(e))
    

    cv2.imshow("vedio",img)
    # cv2.imshow("vedio2",img2)
    # print(img2.shape)
    key=cv2.waitKey(1)
    if key==ord("q"):
        break


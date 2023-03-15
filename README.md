# Face-Mask-Recognition
 This is the code for face mask recognition.
 
 This program has 3 features -
    1. It tells wheather pearson is wearing a mask or not.
    2. It assign a unique Id to each person present in the view of camera
    3. If no person is present in View it shows msg " No Person "
 
 MAIN.py :- This is the runner file which runs above functionality.
 
 DATA :- This directory contains data used to train the cnn model. Data directory contains 2 folder -
    1. with_mask :- which contains the images of person wearing mask.
    2. without_mask :- person don't  wear any mask.
    
 all the images are of dimmension 224 by 224
 
 MODEL :- This directoy contains 2 files -
    1. keras_model.h5 file which is the model we trained on the given data set. This model is used for classifying the face with mask or without mask.
    2. label.txt file which contains the labels of prediction.
    
 requirements.txt file contains all the modules that is used .  
  
 

from statistics import mode

from keras.models import load_model
import cv2
import os
import numpy as np

#necessary functions
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    else:
        raise Exception('Invalid dataset name')
        
#def get_class_to_arg might be a useful func for emojis

#to add an image instead, can use load_image from inference to convert to pil image

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path) #loads an xml file pretrained with haar features
    return detection_model

def detect_faces(detection_model, gray_image_array):   #function to correctly use haar features to detect faces  
    #implement detectMultiScale function using scale factor of 1.3 and 5 minimumm neighbouring rectangles
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5) 

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

def run(frame, net, classes):
        
    height, width, _ = frame.shape   #height and width of the frame captured
        
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), (0, 0, 0), swapRB = True, crop = False)
    net.setInput(blob)
    
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    
    boxes = []       #stores the coordinates and measurements for the bounding box
    confidences = [] #Stores the confidence, i.e how much the object atches with a given class
    class_ids = []   #stores all the labels

    for output in layerOutputs:   #get ouput layers information
        for detection in output:  #extract information from each output (detection contains 85 parameters)
            
            scores = detection[5:] #prediction from all the classes, 6th element onwards
            
            class_id = np.argmax(scores) #extract location of the class with maximum confidence(index)
            confidence = scores[class_id] #extract the vaue of the confidence
            if confidence > 0.2:
                #these are normalised co-ordinates that is why we multiply them with heigth and width to
                #scale them back
                center_x = int(detection[0]*width) #the center x co-ordinate of the bounding box
                center_y = int(detection[1]*height) #the center y co-ordinate of the bounding box
                w = int(detection[2]*width)         #width of the bounding box
                h = int(detection[3]*height)        #height of the bounding box

                x = int(center_x - w/2)             #corner x co-ordinate
                y = int(center_y - h/2)             #corner y co-ordinate

                boxes.append([x, y, w, h])          #saves the co-ordinates and measurement in boxes[]
                confidences.append((float(confidence))) #saves the confidences of the classes
                class_ids.append(class_id)              #index of the classes detected
    
    #performs non-Max Supression on the classes with confidence greater then the threshold
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.2) 


    return indexes, boxes, class_ids, confidences

def boxing(frame, indexes, boxes, class_ids, confidences, classes, font):
    for i in indexes.flatten(): 
            x, y, w, h =  boxes[i] #co-ordinates if bounding boxes of final object after NMS
            label = str(classes[class_ids[i]]) #the name of the object detected
            confidence = str(round(confidences[i], 2)) #saves the confidence rounding it to 2 decimals
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3) #bounda a rectangle around the object
            #shows the confidence and object name at top left
            cv2.putText(frame, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 4)

    return frame

import cv2
import os
import numpy as np

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


# In[ ]:



        # Function to convert number into emoji
        # Switcher is dictionary data type here
def detect_emoji(argument):
                switcher = {	
                0: "\U0001f9D1",	#	  person
                1: "\U0001f6b2",	#	bicycle
                2:"\U0001f697",	#	car
                3:"\U0001f6f5",	#	motorbike
                4:"\U0002f708",	#	aeroplane 
                5:"\U0001f68c",	#	bus
                6:"\U0001f686",	#	train
                7:"\U0001f69a",	#	truck
                8:"\U0001f6e5",	#	boat
                9:"\U0001f6a6",	#	traffic light
                10:"\U0001f9ef",	#	fire hydrant
                11:"\U0001f6d1",	#	stop sign
                12:"\U0001f17f",	#	parking meter
                13:"\U0001f6cb",	#	bench
                14:"\U0001f985",	#	bird
                15:"\U0001f408",	#	cat
                16:"\U0001f415",	#	dog
                17:"\U0001f40e",	#	horse
                18:"\U0001f411",	#	sheep
                19:"\U0001f404",	#	cow
                20:"\U0001f418",	#	elephant
                21:"\U0001f43b",	#	bear
                22:"\U0001f993",	#	zebra
                23:"\U0001f992",	#	giraffe
                24:"\U0001f392",	#	backpack
                25:"\U0002f602",	#	umbrella
                26:"\U0001f45c",	#	handbag
                27:"\U0001f454",	#	tie
                28:"\U0001f6c4",	#	suitcase
                29:"\U0001f94f",	#	frisbee
                30:"\U0001f3bf",	#	skis
                31:"\U0001f3c2",	#	snowboard
                32:"\U0001f3c0",	#	sports ball
                33:"\U0001fa81",	#	kite
                34:"\U0002f6be",	#	baseball bat
                35:"\U0001f9e4",	#	baseball glove
                36:"\U0001f6f9",	#	skateboard
                37:"\U0001f3c4",	#	surfboard
                38:"\U0001f3be",	#	tennis racket
                39:"\U0001f9f4",	#	bottle
                40:"\U0001f377",	#	wine glass
                41:"\U0001f375",	#	cup
                42:"\U0001f374",	#	fork
                43:"\U0001f52a",	#	knife
                44:"\U0001f944",	#	spoon
                45:"\U0001f963",	#	bowl
                46:"\U0001f34c",	#	banana
                47:"\U0001f34e",	#	apple
                48:"\U0001f96a",	#	sandwich
                49:"\U0001f34a",	#	orange
                50:"\U0001f966",	#	broccoli
                51:"\U0001f955",	#	carrot
                52:"\U0001f32d",	#	hot dog
                53:"\U0001f355",	#	pizza
                54:"\U0001f369",	#	donut
                55:"\U0001f382",	#	cake
                56:"\U0001fa91",	#	chair
                57:"\U0001f6cb",	#	sofa
                58:"\U0001fab4",	#	pottedplant
                59:"\U0001f6cf",	#	bed
                60:"\U0001f37d",	#	diningtable
                61:"\U0001f6bd",	#	toilet
                62: "\U0001f4fa",	#	tvmonitor
                63:"\U0001f4bb",	#	laptop
                64:"\U0001f5b1",	#	mouse
                65:"\U0001f4f2",	#	remote
                66:"\U0002f328",	#	keyboard
                67: "\U0001f4F1",	#	cell phone
                68:"\U0001f958",	#	microwave
                69:"\U0001f950",	#	oven
                70:"\U0001f35e",	#	toaster
                71:"\U0001f6be",	#	sink
                72:"\U0001f9ca",	#	refrigerator
                73:"\U0001f4d3",	#	book
                74:"\U0002f3f2",	#	clock
                75:"\U0001f490",	#	vase
                76:"\U0002f702",	#	scissors
                77:"\U0001f9f8",	#	teddy bear
                78:"\U0001f4a8",	#	hair drier
                79:"\U0001faa5",	#	toothbrush
        }

        # get() method of dictionary data type returns
        # value of passed argument if it is present
        # in dictionary otherwise second argument will
        # be assigned as default value of passed argument
                return switcher.get(argument, "nothing")




def netClasses():
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
            
    classes = []
    with open('coco.txt', 'r') as f:
        classes = f.read().splitlines()
    
    return net,classes


def livecam():
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
            
    classes = []
    with open('coco.txt', 'r') as f:
        classes = f.read().splitlines()
    cap = cv2.VideoCapture(0)
    # output = input("Enter Recording output path: ")
    # output_path = os.path.abspath(output)
    ret = True                                       #creates a boolean 
    ret, old_frame = cap.read()                      #ret is true and the first frame of video saved in old_frame
    
    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    
    # # size = (frame_width, frame_height)
    # net,classes = netClasses()
    if not cap.isOpened():
        raise IOError("Cannot open webcam/Cannot read file")
    while ret:
        ret, frame = cap.read()          #saves the first frame of video in frame
        indexes = []
        boxes = []
        class_ids = []
        confidences = []
        indexes, boxes, class_ids, confidences = run(frame, net, classes)
        font = cv2.FONT_HERSHEY_PLAIN

        if len(indexes) <= 0:    #if no bounding box
            continue
        elif len(indexes) > 0:  #if bounding box is presrnt
            frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)
        max = confidences[0]
        index = 0
        for i in range(len(confidences)):
            if confidences[i]>max:
                max=confidences[i]
                index = i

        choice=class_ids[index]
        # return detect_emoji(choice)
        print (detect_emoji(choice))



        # rec_vid.write(frame)
        cv2.imshow('Input', frame)   #opens the webcam in a pop-up window
        old_frame = frame            #saves the vale of the new frame in old frame to be used later in the loop
        c = cv2.waitKey(400)           #new frame comes after () ms
        if cv2.waitKey(4) & 0xFF == ord('q'): #press q on keyboard to stop the webcam
            break

    cap.release()
    cv2.destroyAllWindows()          #Once out of the while loop, the pop-up window closes automatically

def image():  #input is image
    # net,classes = netClasses()
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
            
    classes = []
    with open('coco.txt', 'r') as f:
        classes = f.read().splitlines()
    frame_path = input("Enter the path of the image: ")          #takes in the image as a frame
    frame = cv2.imread(frame_path, cv2.IMREAD_COLOR)
    indexes = []
    boxes = []
    class_ids = []
    confidences = []
    indexes, boxes, class_ids, confidences = run(frame, net, classes)
    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) <= 0:    #if no bounding box
        print("No object to detect")
        exit(0)
    elif len(indexes) > 0:  #if bounding box is presrnt
        frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)
    max = confidences[0]
    index = 0
    for i in range(len(confidences)):
        if confidences[i]>max:
            max=confidences[i]
            index = i

    
    choice=class_ids[index]
    print (detect_emoji(choice))
    # return detect_emoji(choice)


    # rec_vid.write(frame)
    cv2.imshow('Input', frame)   #opens the webcam in a pop-up window

def video():
    # net,classes = netClasses()
    net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
            
    classes = []
    with open('coco.txt', 'r') as f:
        classes = f.read().splitlines()
    video = input("Enter Recording video path: ")
    video_path = os.path.abspath(video)
    cap = cv2.VideoCapture(video_path)
    ret = True                                       #creates a boolean 
    ret, old_frame = cap.read()                      #ret is true and the first frame of video saved in old_frame
    

    # frame_width = int(cap.get(3))
    # frame_height = int(cap.get(4))
    
    # size = (frame_width, frame_height)

    if not cap.isOpened():
        raise IOError("Cannot open webcam/Cannot read file")
    while ret:
        ret, frame = cap.read()          #saves the first frame of video in frame
        indexes = []
        boxes = []
        class_ids = []
        confidences = []
        indexes, boxes, class_ids, confidences = run(frame, net, classes)
        font = cv2.FONT_HERSHEY_PLAIN

        if len(indexes) <= 0:    #if no bounding box
            continue
        elif len(indexes) > 0:  #if bounding box is presrnt
            frame = boxing(frame, indexes, boxes, class_ids, confidences, classes, font)
        max = confidences[0]
        index = 0
        for i in range(len(confidences)):
            if confidences[i]>max:
                max=confidences[i]
                index = i

        choice=class_ids[index]
        print (detect_emoji(choice))
        # return detect_emoji(choice)



        # rec_vid.write(frame)
        cv2.imshow('Input', frame)   #opens the webcam in a pop-up window
        old_frame = frame            #saves the vale of the new frame in old frame to be used later in the loop
        c = cv2.waitKey(400)           #new frame comes after () ms
        if cv2.waitKey(4) & 0xFF == ord('q'): #press q on keyboard to stop the webcam
            break

    cap.release()
    cv2.destroyAllWindows()          #Once out of the while loop, the pop-up window closes automatically

#if input is image, mchoice = 1
#if input is video from path, mchoice = 2
#if input is live video, mchoice = 0

mchoice = int(input("Do you want to upload 1. Image 2. Video 0. Live Video? (0/1/2)"))
if(mchoice==0):
    print(livecam())
elif(mchoice==1):
    print(image())
elif(mchoice==2):
    print(video())

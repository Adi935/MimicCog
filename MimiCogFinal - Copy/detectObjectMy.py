## live cam function

from functions import *
flag = 0
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3.cfg')
        
classes = []
with open('coco.txt', 'r') as f:
    classes = f.read().splitlines()

def detectObj(frame):
    indexes = []
    boxes = []
    class_ids = []
    confidences = []
    indexes, boxes, class_ids, confidences = run(frame, net, classes)
    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) <= 0:    #if no object detected
            return
    max = confidences[0]
    index = 0
    for i in range(len(confidences)):
        if confidences[i]>max:
            max=confidences[i]
            index = i

    choice=class_ids[index]
    print("first detecttion",classes[class_ids[index]], confidences[i])    
    # print (detect_emoji(choice))
    if(choice==0):
        print("person detected, let's see your emotion")
        # parameters for loading data and images
        detection_model_path = 'haarcascade_frontalface_default.xml'
        emotion_model_path = 'fer2013_mini_XCEPTION.102-0.66.hdf5'
        emotion_labels = get_labels('fer2013')

        # hyper-parameters for bounding boxes shape
        frame_window = 10
        emotion_offsets = (20, 40)

        # loading models
        face_detection = load_detection_model(detection_model_path)
        emotion_classifier = load_model(emotion_model_path, compile=False)

        # getting input model shapes for inference
        emotion_target_size = emotion_classifier.input_shape[1:3]

        # starting lists for calculating modes
        emotion_window = []
        bgr_image = frame
        gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        faces = detect_faces(face_detection, gray_image)

        for face_coordinates in faces:

            x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
            gray_face = gray_image[y1:y2, x1:x2]
            try:
                gray_face = cv2.resize(gray_face, (emotion_target_size))
            except:
                continue

            gray_face = preprocess_input(gray_face, True)
            gray_face = np.expand_dims(gray_face, 0)
            gray_face = np.expand_dims(gray_face, -1)
            emotion_prediction = emotion_classifier.predict(gray_face)
            emotion_probability = np.max(emotion_prediction)
            emotion_label_arg = np.argmax(emotion_prediction)
            emotion_text = emotion_labels[emotion_label_arg]
            emotion_window.append(emotion_text)

            if len(emotion_window) > frame_window:
                emotion_window.pop(0)
            try:
                emotion_mode = mode(emotion_window)
            except:
                continue

            if emotion_mode == 'angry':
                emojiChoice = 81
            elif emotion_text == 'sad':
                emojiChoice = 86
            elif emotion_text == 'happy':
                emojiChoice = 91
            elif emotion_text == 'surprise':
                emojiChoice = 96
            else:
                emojiChoice = 101

            # color = color.astype(int)
            # color = color.tolist()

            # draw_bounding_box(face_coordinates, rgb_image, color)
            # draw_text(face_coordinates, rgb_image, emotion_mode,
            #         color, 0, -45, 1, 1)

                ## let us have five emojis for each emotion, labelling them rn as 1,2,3,4,5
            ## where for eg. happy5 is extremely happy and happy1 is least happy
            if(emotion_probability<0.2):
                emojiNo=1
            elif(emotion_probability<0.4): 
                emojiNo=2
            elif(emotion_probability<0.6): 
                emojiNo=3
            elif(emotion_probability<0.8): 
                emojiNo=4
            elif(emotion_probability<1): 
                emojiNo=5

            # bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            # choiceE=emotion_label_arg
            print(emotion_mode, emotion_probability,emojiNo)
            emotionFinal = emojiChoice+emojiNo
            return emotionFinal
        else:
            print("out of if statement",classes[class_ids[index]], confidences[i]) 
            return class_ids[index]



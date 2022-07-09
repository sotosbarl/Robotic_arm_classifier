import object_detection
import os
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
import requests
import imutils
import serial
import pygame

ser = serial.Serial('COM8', 9600)

# Replace the below URL with your own. Make sure to add "/shot.jpg" at last.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


CUSTOM_MODEL_NAME = 'last_ping'
PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
# PRETRAINED_MODEL_NAME='ping_base'
PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
LABEL_MAP_NAME = 'label_map.pbtxt'
paths = {
    'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
    'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
    'APIMODEL_PATH': os.path.join('Tensorflow','models'),
    'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
    'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
    'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
    'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
    'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME),
    'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'),
    'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'),
    'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'),
    'PROTOC_PATH':os.path.join('Tensorflow','protoc')
 }
files = {
    'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
    'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME),
    'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
}


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-7')).expect_partial()

@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

import cv2
import numpy as np
from matplotlib import pyplot as plt
category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

##ΤEST WITH VIDEO FROM WEBCAM
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
running = True

array=[];
while cap.isOpened():
    ret, frame = cap.read()
    image_np = np.array(frame)


    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()
    #lock.acquire()

    img, my_variable=viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=1,
                min_score_thresh=0.91,
                agnostic_mode=False)
    array.insert(0,my_variable)
    if len(array)>9:
        array.pop()

# 1 is for the white ball, 2 is for the red ball and 0 is for no ball

    cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
    print('opencv variable is {}'.format(my_variable))

    flag=0
    while ser.in_waiting:

        test_variable=ser.read()

        if test_variable.decode('ascii')=='8':
            for i in array:
                if i==1:
                    ser.write(str.encode('1'))
                    flag=1
                    break #ball is white


            if flag!=1:
                ser.write(str.encode('0')) #ball is red
                flag=0


    ###########PYGAME BEGINS HERE

    from pygame.locals import (
        K_UP,
        K_DOWN,
        K_LEFT,
        K_RIGHT,
        K_ESCAPE,
        K_KP1,
        K_KP4,
        K_KP2,
        K_KP5,
        K_KP3,
        K_KP6,
        KEYDOWN,
        QUIT,
        K_KP_9,
    )

    # Initialize pygame
    pygame.init()

    # Define constants for the screen width and height
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600

    # Create the screen object
    # The size is determined by the constant SCREEN_WIDTH and SCREEN_HEIGHT
    #screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    # Variable to keep the main loop running
    white = (255, 255, 255)
    green = (0, 255, 0)
    blue = (0, 0, 128)

    # assigning values to X and Y variable
    X = SCREEN_WIDTH
    Y = SCREEN_HEIGHT

    # create the display surface object
    # of specific dimension..e(X, Y).
    surface = pygame.display.set_mode((X, Y))

    # set the pygame window name
    pygame.display.set_caption('Show Text')

    # create a font object.
    # 1st parameter is the font file
    # which is present in pygame.
    # 2nd parameter is size of the font
    #font = pygame.font.Font('freesansbold.ttf', 32)

    # create a text surface object,
    # on which text is drawn on it.
    #text = font.render('GeeksForGeeks', 0, green, blue)

    # create a rectangular object for the
    # text surface object
    # textRect = text.get_rect()
    #
    # # set the center of the rectangular object.
    # textRect.center = (X // 2, Y // 2)

    screen = pygame.display.get_surface()
    font = pygame.font.Font(None, 40)

    font_surface = font.render("original", True, pygame.Color("white"));

    count=0

    # Main loop
    while running:
        # Look at every event in the queue
        surface.fill(white)
        screen.blit(surface, (0, 0))


        #display_surface.blit(text, textRect)
        pygame.display.update()
        for event in pygame.event.get():
            # Did the user hit a key?
            #
            if event.type == KEYDOWN:
                # Was it the Escape key? If so, stop the loop.
                if event.key == K_ESCAPE:
                    running = False
                if event.key == K_KP1:
                    ser.write(str.encode('1'))
                if event.key == K_KP4:
                    ser.write(str.encode('4'))
                if event.key == K_KP2:
                    ser.write(str.encode('2'))
                if event.key == K_KP5:
                    ser.write(str.encode('5'))
                if event.key == K_KP3:
                    ser.write(str.encode('3'))
                if event.key == K_KP6:
                    ser.write(str.encode('6'))
                if event.key == K_KP_9:
                    ser.write(str.encode('9'))
                    count=count+1
                    if count==5:
                        running=0


                    # text = font.render('Now select lalala', 0, green, blue)
                    # textRect = text.get_rect()
                    # # set the center of the rectangular object.
                    # textRect.center = (X // 2, Y // 2)
                    # display_surface.blit(text, textRect)
                    # pygame.display.flip()
                    #
                    #





            # Did the user click the window close button? If so, stop the loop.
            elif event.type == QUIT:
                running = False
    ################PYGAME ENDS HERE
    if cv2.waitKey(10) & 0xFF == ord('q'):
     # cap.release()
      #cv2.destroyAllWindows()
      break

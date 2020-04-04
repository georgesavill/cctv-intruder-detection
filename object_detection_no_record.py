import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from datetime import datetime
from utils import label_map_util
from utils import visualization_utils as vis_util
from grabscreen import grab_screen


VIDEO_STREAM_URL = "http://username:password@192.168.1.93:8080/video"
PATH_TO_GRAPH = "./model/frozen_inference_graph.pb"
PATH_TO_LABELS = "./model/labelmap.pbtxt"
NUM_CLASSES = 3

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

object_detection_graph = tf.Graph()
with object_detection_graph.as_default():
    object_detection_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        object_detection_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(object_detection_graph_def, name='')

    sess = tf.Session(graph=object_detection_graph)

image_tensor = object_detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = object_detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = object_detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = object_detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = object_detection_graph.get_tensor_by_name('num_detections:0')

video_stream = cv2.VideoCapture(VIDEO_STREAM_URL)

analyse_frame = False

while(video_stream.isOpened()):
    frame_recieved, frame = video_stream.read()

    frame = cv2.resize(frame, (480, 270))

    if frame_recieved:

        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        squeezed_scores = np.squeeze(scores)
        squeezed_classes = np.squeeze(classes).astype(np.int32)



        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            squeezed_classes,
            squeezed_scores,
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=0.95)

        screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
        frame = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        cv2.imshow('Object detector',frame)


    else:
        print("frame skipped")

    if cv2.waitKey(1) == ord('q'):
        break

video_stream.release()
cv2.destroyAllWindows()
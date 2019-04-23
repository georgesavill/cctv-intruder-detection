import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from datetime import datetime
from utils import label_map_util
from utils import visualization_utils as vis_util


def getcurrenttime():
    now = datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    
    return(year + "-" + month + "-" + day + "-" + hour + "-" + minute + "-" + second)


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


fourcc = cv2.VideoWriter_fourcc(*'MJPG')
# output_video = cv2.VideoWriter(getcurrenttime() + '.avi', fourcc, 20.0, (720,720))

empty_frame_count = 10

while(video_stream.isOpened()):
    frame_recieved, frame = video_stream.read()
    # frame = frame[350:900, 538:1088]

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

        for i in range (0, 5):
            if squeezed_scores[i] > 0.95:
                if squeezed_classes[i] in category_index.keys():
                    class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name']
                    if (class_name == "person"):
                        if (empty_frame_count > 5):
                            output_video = cv2.VideoWriter(getcurrenttime() + '.avi', fourcc, 20.0, (720,720))
                            output_video.write(frame)
                        else:
                            output_video.write(frame)

                        empty_frame_count = 0 

        empty_frame_count += 1


        cv2.imshow('Object detector', frame)
        print(empty_frame_count)
    else:
        print("frame skipped")

    if cv2.waitKey(1) == ord('q'):
        break

output_video.release()
video_stream.release()
cv2.destroyAllWindows()
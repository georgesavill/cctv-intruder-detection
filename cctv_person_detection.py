import cv2
from datetime import datetime
import io
import numpy as np
from tflite_runtime.interpreter import Interpreter

VIDEO_STREAM_URL = "rtsp://admin:password@192.168.1.1:554/Streaming/Channels/102/"

def get_current_time():
  now = datetime.now()
  year = str(now.year)
  month = str(now.month).zfill(2)
  day = str(now.day).zfill(2)
  hour = str(now.hour).zfill(2)
  minute = str(now.minute).zfill(2)
  second = str(now.second).zfill(2)
  return(year + "-" + month + "-" + day + "-" + hour + "-" + minute + "-" + second)

def set_input_tensor(interpreter, image):
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image
def get_output_tensor(interpreter, index):
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor

def detect_objects(interpreter, image, threshold):
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))
  
  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results

def main():
  interpreter = Interpreter("/tmp/detect.tflite")
  interpreter.allocate_tensors()
  video_feed = cv2.VideoCapture(VIDEO_STREAM_URL)

  while (video_feed.isOpened()):
    feed_opened, frame = video_feed.read()

    if not feed_opened:
      break

    cropped_frame = frame[60:360, 200:500]

    results = detect_objects(interpreter, cropped_frame, 0.6)
    for result in results:
        if result["class_id"] == 0:
            filename = "./images/" + get_current_time() + ".jpg"
            cv2.imwrite(filename,frame)

  video_feed.release()

if __name__ == '__main__':
  main()
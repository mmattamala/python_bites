#!/usr/bin/env python

import rospy
import rostopic

from ultralytics import YOLO
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator


class YoloRos:
    def __init__(self):
        # Load a COCO-pretrained YOLOv8n model
        self._model = YOLO("yolov8n.pt")
        # self._model = YOLO("yolov8n-seg.pt")
        self._model = self._model.to("cpu")

        # Prepare subscriber
        input_topic = rospy.get_param(
            "~input_image_topic", "/alphasense_driver_ros/cam0/color/image/compressed"
        )
        input_image_type, input_image_topic, _ = rostopic.get_topic_type(
            input_topic, blocking=True
        )
        self._is_compressed = (input_image_type == "sensor_msgs/CompressedImage")

        if self._is_compressed:
            self._sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=10
            )
        else:
            self._sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=10
            )
 
        # Prepare publisher
        self._pub = rospy.Publisher("~output_image", Image, queue_size=10)

        # Prepare cv bridge
        self._bridge = CvBridge()

    def callback(self, msg):
        # convert image + predict
        if self._is_compressed:
            img = self._bridge.compressed_imgmsg_to_cv2(msg)
        else:
            img = self._bridge.imgmsg_to_cv2(msg)

        # Run inference with the YOLOv8n model on the 'bus.jpg' image
        results = self._model(img)[0]

        # Anotate results
        annotator = Annotator(img)
        boxes = results.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, self._model.names[int(c)])
            
        annotated_img = annotator.result()
        out_msg = self._bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
        out_msg.header = msg.header
        self._pub.publish(out_msg)

if __name__ == "__main__":
    rospy.init_node("yolo_ros")
    yolo = YoloRos()
    rospy.spin()

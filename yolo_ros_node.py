#!/usr/bin/env python
# author: Matias Mattamala
#
# Dependencies (assuming ROS installation on base system)
#  - No dependencies, just pure ROS libs

import rospy
import rostopic
import numpy as np
import cv2

from ultralytics import YOLO
from ultralytics.utils.ops import scale_image
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors


class YoloRos:
    def __init__(self):
        # Read ROS parameters
        input_topic = rospy.get_param("~input_image_topic", "/camera/image")

        # for YOLO only "yolov8n.pt"
        self._model_name = rospy.get_param("~model", "yolov8n-seg.pt")

        # Device where the model runs
        self._device = rospy.get_param("~device", "cpu")

        # ALpha value for segmentation masks
        self._alpha = rospy.get_param("~alpha", 0.3)

        # Filter visualizations by confidence
        self._conf = rospy.get_param("~valid_conf", 0.7)

        # Load model
        self._model = YOLO(self._model_name)
        self._model = self._model.to(self._device)

        # Configure ROS
        input_image_type, input_image_topic, _ = rostopic.get_topic_type(
            input_topic, blocking=True
        )
        self._is_compressed = input_image_type == "sensor_msgs/CompressedImage"

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
        # Convert to numpy image
        if self._is_compressed:
            img = self._bridge.compressed_imgmsg_to_cv2(msg)
        else:
            img = self._bridge.imgmsg_to_cv2(msg)

        # Run inference
        result = self._model(img)[0]

        # Anotate results
        # Show bounding boxes
        annotator = Annotator(img)
        boxes = result.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls  # class
            p = box.conf.item()  # confidence
            if p > self._conf:
                annotator.box_label(
                    b,
                    f"{self._model.names[int(c)]} ({p * 100:.2f}%)",
                    color=colors(c, True),
                )
        out_img = annotator.result()

        # Show segmentation
        # Adapted from https://github.com/ultralytics/ultralytics/issues/561#issuecomment-1403079910
        # result.boxes.xyxy   # box with xyxy format, (N, 4)
        cls = result.boxes.cls.cpu().numpy()  # cls, (N, 1)
        probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
        boxes = result.boxes.xyxy.cpu().numpy()  # box with xyxy format, (N, 4)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # masks, (N, H, W)
            masks = np.moveaxis(masks, 0, -1)  # masks, (H, W, N)
            # rescale masks to original image
            masks = scale_image(masks, result.masks.orig_shape)
            masks = np.moveaxis(masks, -1, 0)  # masks, (N, H, W)

            for mask, c, p in zip(masks, cls, probs):
                if p > self._conf:
                    color = colors(c, True)
                    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
                    colored_mask = np.moveaxis(colored_mask, 0, -1)
                    masked = np.ma.MaskedArray(
                        out_img, mask=colored_mask, fill_value=color
                    )
                    overlay_img = masked.filled()
                    out_img = cv2.addWeighted(
                        out_img, 1 - self._alpha, overlay_img, self._alpha, 0
                    )

        # Publish
        out_msg = self._bridge.cv2_to_imgmsg(out_img, encoding="bgr8")
        out_msg.header = msg.header
        self._pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("yolo_ros_node")
    yolo = YoloRos()
    rospy.spin()

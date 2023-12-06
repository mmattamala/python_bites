#!/usr/bin/env python

import rospy
import cv2

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

if __name__ == "__main__":
    rospy.init_node("camera")

    pub = rospy.Publisher("~image", Image, queue_size=10)
    bridge = CvBridge()

    # define a video capture object
    vid = cv2.VideoCapture(0)

    rate = rospy.Rate(30)
    while True:
        # read frame
        ret, frame = vid.read()

        # publish
        out_msg = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out_msg.header.stamp = rospy.Time.now()
        pub.publish(out_msg)
        rate.sleep()

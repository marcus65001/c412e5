#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
import PIL

class TestNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(TestNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # subscriber
        self.sub_roi = rospy.Subscriber('~digit', Int8, self.cb_digit)  # image topic

        # publisher
        self.pub = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.VISUALIZATION,
            dt_help="The stream of JPEG compressed images from the modified camera feed",
        )

        # services
        # self.srvp_led_emitter = rospy.ServiceProxy(
        #     "~set_pattern", ChangePattern
        # )

        # parameters and internal objects
        self.image = None
        self._bridge = CvBridge()

    def read_image(self):
        img_cv=cv2.imread("/home/marcus/test.png")
        self.image=self._bridge.cv2_to_compressed_imgmsg(img_cv, dst_format="jpeg")
        self.log("image loaded")

    def cb_digit(self,msg):
        if msg.data:
            self.log("digit: {}".format(msg.data))

    def run(self):
        self.read_image()
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.pub.publish(self.image)
                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = TestNode(node_name='test_node')
    # keep spinning
    node.run()
    rospy.spin()

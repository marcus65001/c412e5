#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
import math
from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped
from duckietown_msgs.srv import ChangePattern, ChangePatternResponse
from std_msgs.msg import String, Int32

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(0, 70, 50), (10, 255, 255)]
STOP_LINE_MASK2 = [(170, 70, 50), (180, 255, 255)]
NUMBER_MASK = [(60, 122, 102), (134, 253, 162)]
LEFT_MASK = np.zeros((480,640), dtype=np.uint8)
LEFT_MASK[ :, :-540] = 1
RIGHT_MASK = np.zeros((480,640), dtype=np.uint8)
RIGHT_MASK[ :, 100:] = 1
DEBUG = True
ENGLISH = False

class LaneFollowNode(DTROS):

    def __init__(self, node_name):
        super(LaneFollowNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")



        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        self.timer = None
        self.stop_detection = False

        self.loginfo("timer called")

        self.num_of_detections = 0
        self.tag_to_action = {
            62: 'Left',
            58: 'Right',
            162: 'Straight',
            133: 'Right',
            169: 'Left',
            153: 'Straight',
            None: 'Straight'
        }
        self.tag_to_mask = {
            62: RIGHT_MASK,
            58: LEFT_MASK,
            162: None,
            133: LEFT_MASK,
            169: RIGHT_MASK,
            153: None,
            None: None
        }
        self.tagid = None
        self.mask=LEFT_MASK

        # PID Variables
        self.proportional = None
        self.proportional_stop = 0.0
        if ENGLISH:
            self.offset = -220
        else:
            self.offset = 220
        self.velocity = 0.4
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.049
        self.D = -0.004
        self.last_error = 0
        self.last_time = rospy.get_time()

        self.P_2 = (4/1600)
        self.stop_ofs = 0.0
        self.stop_times_up = False
        self.STOP = False

        # Image variables
        self.width = 640
        self.height = 480

        self.start_detect_shutoff_time = None
        self.current_detect_shutoff_time = None

        # self.point_positon = None
        # self.angular_position = None
        # self.angle_of_rotation = None
        # self.x = None

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.tagid_sub = rospy.Subscriber("~tagid",
                                          Int32,
                                          self.cb_tagid_detect,
                                          queue_size=1)

        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    # def cb_odometry(self, msg):

    #     self.point_positon = msg.pose.pose.position
    #     self.angular_position = msg.pose.pose.orientation
    #     self.loginfo(self.point_positon)
    #     self.loginfo(self.angular_position)

    #     # Setting the quaternarian x axis of rotation into euler
    #     self.x = self.point_positon.x
    #     sinr_cosp = 2 * (self.angular_position.w * self.angular_position.x + self.angular_position.y * self.angular_position.z)
    #     cosr_cosp = 1 - 2 * (self.angular_position.x * self.angular_position.x + self.angular_position.y * self.angular_position.y)
    #     self.angle_of_rotation = math.atan2(sinr_cosp, cosr_cosp)

    #Calculates the midpoint of the contoured object

    # def perform_action (self):

    #     if self.detected_apriltag[self.num_of_detections] in self.avail_tags["inter_tags"]:
          
    def midpoint (self, x, y, w, h):
        mid_x = int(x + (((x+w) - x)/2))
        mid_y = int(y + (((y+h) - y)))
        return (mid_x, mid_y)
    
    
    # def intersection_action(self):
    #
    #     if self.tag_to_action[self.tagid] == 'Left':
    #         self.loginfo("Turning Left")
    #         self.twist.v = 0.3
    #         self.twist.omega = 0.5
    #     elif self.tag_to_action[self.tagid] == 'Right':
    #         self.loginfo("Turning Right")
    #         self.twist.v = 0.3
    #         self.twist.omega = - 0.5
    #     else:
    #         self.twist.v = 0.4
    #         self.twist.omega = 0.0
        
    
    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        img = cv2.bitwise_and(img, img, mask=self.mask)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.proportional = cx - int(crop_width / 2) + self.offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.proportional = None

        if not self.stop_detection and not self.STOP:
            # Search for stop line
            img2 = self.jpeg.decode(msg.data)
            crop2 = img2[400:,:,:]

            cv2.line (crop2, (320, 240), (0,240), (255,0,0), 1)

            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
            maskl = cv2.inRange(hsv2, STOP_LINE_MASK[0], STOP_LINE_MASK[1])
            maskh=cv2.inRange(hsv2, STOP_LINE_MASK2[0], STOP_LINE_MASK2[1])
            mask2=maskh+maskl
            contours, hierarchy = cv2.findContours(mask2,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            if DEBUG:
                rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop2))
                self.pub.publish(rect_img_msg)
            if len(contours) != 0: 
                max_contour = max(contours, key=cv2.contourArea)
                """
                # Generates the size and the cordinantes of the bounding box and draw
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(crop2,(x,y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(crop2, self.midpoint(x,y,w,h), 2, (63, 127, 0), -1)
                # Calculate the pixel distance from the middle of the frame
                pixel_distance = math.sqrt(math.pow((160 - self.midpoint(x,y,w,h)[1]),2))
                cv2.line (crop2, self.midpoint(x,y,w,h), (self.midpoint(x,y,w,h)[0], 240), (255,0,0), 1)
                self.proportional_stop = pixel_distance
                self.STOP = True
                """
                self.loginfo("stop")
                self.STOP=True
                timer=rospy.Timer(rospy.Duration(3.),self.cb_stop,oneshot=True)
            """
            else:
                self.proportional_stop = 0.0
                self.STOP = False
                """



    def cb_stop(self,t):
        self.loginfo("stop timer started")
        self.STOP=False
        self.stop_detection=True
        tmr=rospy.Timer(rospy.Duration(4), self.cb_timer, oneshot=True)

    
    def cb_tagid_detect (self, msg):
        # self.loginfo(msg.data)
        if msg.data > 0:
            self.loginfo("Updated")
            self.tagid = msg.data
            self.mask=self.tag_to_mask[self.tagid]

    def drive(self):
        if self.STOP:
            self.twist.omega=0
            self.twist.v=0
        else:
            if self.proportional is None:
                self.twist.omega = 0
            else:
                # P Term
                P = - self.proportional * self.P

                # D Term
                d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D

                self.twist.v = self.velocity
                self.twist.omega = P + D
        # self.loginfo(self.twist)
        self.vel_pub.publish(self.twist)
    """
    def drive2(self):

        #Start driving when I don't detect a stopline or when I finsihed being stopped at one
        if not self.STOP:

            if self.proportional is None:
                self.twist.omega = 0
            else:
                # P Term
                P = - self.proportional * self.P

                # D Term
                d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time)
                self.last_error = self.proportional
                self.last_time = rospy.get_time()
                D = d_error * self.D

                self.twist.v = self.velocity
                self.twist.omega = P + D
                # if DEBUG:
                #     self.loginfo([self.proportional, P, D, self.twist.omega, self.twist.v])
        else:
            # P Term

            P_2 = (self.proportional_stop - self.stop_ofs) * self.P_2
            # print(P_2)

            # This makes sures the values outputted by P_2 stays within 0.0 and 0.4
            self.twist.v = np.clip(P_2, 0.0, 0.4)
            self.twist.omega = 0.0

            # When stopped wait 1 second then start moving
            if self.twist.v == 0.0:
                rospy.sleep(2)
                self.stop_detection = True
                self.loginfo("Timer set")
                if self.timer != None:
                    self.timer = rospy.Timer(rospy.Duration(4), self.cb_timer, oneshot= True)
                self.intersection_action()

                # self.stop_times_up = True


            # if DEBUG:
            #   self.loginfo([self.proportional, P, D, self.twist.omega, self.twist.v])

        self.vel_pub.publish(self.twist)
    """

    def cb_timer(self, te):
        self.loginfo("stop det timer")
        self.stop_detection = False
        self.timer = None

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = LaneFollowNode("lanefollow_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()
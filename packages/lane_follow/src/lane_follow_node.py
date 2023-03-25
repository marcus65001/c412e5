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
from std_msgs.msg import Int32, Bool

ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(-5, 130, 178), (10, 255, 255)]
NUMBER_MASK = [(60, 122, 102), (134, 253, 162)]
DEBUG = False
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
            58: 'Straight',
            162: 'Right',
            133: 'Right',
            169: 'Left',
            153: 'Straight',
            None: 'Straight'
        }
        self.action_num = 0
        self.tagid = None

        # PID Variables
        self.proportional = None
        self.proportional_stop = 0.0
        if ENGLISH:
            self.offset = -200
        else:
            self.offset = 200
        self.velocity = 0.3
        self.twist = Twist2DStamped(v=self.velocity, omega=0)

        self.P = 0.041
        self.D = -0.0025
        self.last_error = 0
        self.last_time = rospy.get_time()

        self.P_2 = (4/1600)
        self.stop_ofs = 0.0
        self.stop_times_up = False
        self.STOP = False

        # Image variables
        self.width = 640
        self.height = 480
        self.add_patch = True

        self.start_detect_shutoff_time = None
        self.current_detect_shutoff_time = None

        # Publishers & Subscribers
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        
        self.tagid_sub = rospy.Subscriber("~tagid",
                                    Int32,
                                    self.cb_tagid_detect,
                                    queue_size=1)
        
        self.vel_shutdown = rospy.Subscriber('~shutdown', 
                                        Bool,
                                        self.cb_shutdown,
                                        queue_size=1)
        
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        # Wait a little while before sending motor commands
        rospy.Rate(0.20).sleep()

        # self.point_positon = None
        # self.angular_position = None
        # self.angle_of_rotation = None
        # self.x = None
        

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
    
    
    def intersection_action(self):

        self.loginfo("TAG DETECTED: " + str(self.tagid))
        if self.tag_to_action.get(self.tagid) == 'Left':
            self.loginfo("Turning Left")
            self.twist.v = 0.3
            self.twist.omega = 0.5
        elif self.tag_to_action.get(self.tagid) == 'Right':
            self.loginfo("Turning Right")
            self.add_patch = False
            self.twist.v = 0.35
            self.twist.omega = - 0.7
        else:
            self.twist.omega = 0
            self.twist.v = self.velocity
            self.last_error = 0   
    
    def callback(self, msg):

        img = self.jpeg.decode(msg.data)
        if self.add_patch:
            m_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            m_mask[ :, :-280] = 1
            img = cv2.bitwise_and(img, img, mask=m_mask)
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

        # Search for stop line
        img2 = self.jpeg.decode(msg.data)
        crop2 = img2[320:480,300:640,:]

        cv2.line (crop2, (320, 240), (0,240), (255,0,0), 1)

        if not self.stop_detection:

            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, STOP_LINE_MASK[0], STOP_LINE_MASK[1])
            contours, hierarchy = cv2.findContours(mask2,
                                                cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) != 0: 
                max_contour = max(contours, key=cv2.contourArea)
                # Generates the size and the cordinantes of the bounding box and draw
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(crop2,(x,y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(crop2, self.midpoint(x,y,w,h), 2, (63, 127, 0), -1)
                # Calculate the pixel distance from the middle of the frame
                pixel_distance = math.sqrt(math.pow((160 - self.midpoint(x,y,w,h)[1]),2))
                cv2.line (crop2, self.midpoint(x,y,w,h), (self.midpoint(x,y,w,h)[0], 240), (255,0,0), 1)
                self.proportional_stop = pixel_distance
                self.STOP = True
            else: 
                self.proportional_stop = 0.0
                self.STOP = False
        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)
    
    def cb_tagid_detect (self, msg):
        
        if msg.data in self.tag_to_action:
            self.loginfo("Updated")
            self.tagid = msg.data

    def drive(self):

        #Start driving when I don't detect a stopline or when I finsihed being stopped at one
        if not self.STOP:

            if self.proportional is None:
                # self.twist.omega = 0
                # self.last_error = 0
                self.intersection_action()
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
                if self.timer is None:
                    self.stop_detection = True
                    self.loginfo("Timer set")
                    self.timer = rospy.Timer(rospy.Duration(5), self.cb_timer, oneshot= True)
            # self.intersection_action()
                # self.stop_times_up = True
            self.STOP = False

        # if DEBUG:
            # self.loginfo([self.proportional, P, D, self.twist.omega, self.twist.v])
            # self.loginfo(self.twist)
        # self.loginfo("STOP DETECTION STATUS: " + str(self.stop_detection))
        # self.loginfo([self.twist.v, self.twist.omega])
        self.vel_pub.publish(self.twist)

    def cb_timer(self, te):
        self.loginfo("Timer Up")
        self.stop_detection = False
        self.timer = None
        self.add_patch = True

    def cb_shutdown(self,msg):
        if msg.data == True:
            self.loginfo("shutdown signal recieved")
            rospy.signal_shutdown("Finish")

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
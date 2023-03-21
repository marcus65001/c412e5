#!/usr/bin/env python3
import rospy
from duckietown.dtros import DTROS, NodeType, TopicType
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Int8
# nn
from torch import nn
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.models import resnet18
import PIL


class ResNetMNIST(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_no):
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.parameters(), lr=0.005)


class InferenceNode(DTROS):

    def __init__(self, node_name):
        # initialize the DTROS parent class
        super(InferenceNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)

        self.veh = rospy.get_param("~veh")

        # subscriber
        self.sub_roi = rospy.Subscriber('~cam_roi', CompressedImage, self.cb_img)  # image topic

        # publisher
        self.pub_digit = rospy.Publisher(
            "~digit",
            Int8,
            queue_size=1,
        )

        # services
        # self.srvp_led_emitter = rospy.ServiceProxy(
        #     "~set_pattern", ChangePattern
        # )

        # parameters and internal objects
        self.image = None
        self._bridge = CvBridge()

        # nn
        self._model = ResNetMNIST.load_from_checkpoint("/data/resnet18_mnist.pt")
        self._transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def read_image(self, msg):
        try:
            img = self._bridge.compressed_imgmsg_to_cv2(msg)
            if (img is not None) and (self.image is None):
                self.log("first image msg")
            return img
        except Exception as e:
            self.log(e)
            return np.array([])

    def cb_img(self, msg):
        # image callback
        if self._bridge:
            t_img=self.read_image(msg)
            self.image = self._transforms(PIL.Image.fromarray(t_img))

    def get_prediction(self, x: pl.LightningModule):
        self._model.freeze()  # prepares model for predicting
        probabilities = torch.softmax(self._model(x), dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class, probabilities

    def inference(self):
        img = self.image
        self.image = None
        pred,prob=self.get_prediction(img)
        msg=Integer()
        msg.data=-1
        if prob[pred]>0.5:
            msg.data = pred
        self.pub_digit.publish(msg)
        return


    def run(self):
        rate = rospy.Rate(2)
        while not rospy.is_shutdown():
            if self.image is not None:
                self.inference()
                rate.sleep()


if __name__ == '__main__':
    # create the node
    node = InferenceNode(node_name='inference_node')
    # keep spinning
    node.run()
    rospy.spin()

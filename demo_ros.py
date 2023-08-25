#! /usr/bin/env python3

import rospy
import message_filters
from std_msgs.msg import String
from sensor_msgs.msg import Image
import cv2
import numpy as np
import os
import time
from cv_bridge import CvBridge, CvBridgeError

import argparse
import torch

from torchvision import transforms
from models.fast_scnn import get_fast_scnn
from PIL import Image as pilimg
from utils.visualize import get_color_pallete

class SegmentationDemo():
    def __init__(self):
        rospy.init_node('segmentation_node', anonymous=True)
        self.init_finished = False
        self.bridge = CvBridge()
        self.sub_rgb_image = message_filters.Subscriber("/row_detection/color/image_raw",Image)
        self.sub_depth_image = message_filters.Subscriber("/row_detection/aligned_depth_to_color/image_raw",Image)
        self.sub_syncrhonizer = message_filters.ApproximateTimeSynchronizer(
                                                    [self.sub_rgb_image, self.sub_depth_image], 
                                                    1, 0.5, allow_headerless=True)
        self.sub_syncrhonizer.registerCallback(self.image_process)


        # init publisher
        self.pub_image = rospy.Publisher("/row_segmentation/output_rgb", Image, queue_size=10)

        self.arg_parser()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # image transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        self.model = get_fast_scnn(self.args.dataset, pretrained=True, root=self.args.weights_folder, map_cpu=self.args.cpu, weight_name=self.args.weight_name).to(self.device)
        print('Finished loading model!')
        self.model.eval()
        self.init_finished = True

    def arg_parser(self):
        parser = argparse.ArgumentParser(
            description='Predict segmentation result from a given image')
        parser.add_argument('--model', type=str, default='fast_scnn',
                            help='model name (default: fast_scnn)')
        parser.add_argument('--dataset', type=str, default='citys',
                            help='dataset name (default: citys)')
        parser.add_argument('--weights-folder', default='./weights',
                            help='Directory for saving checkpoint models')
        parser.add_argument('--input-pic', type=str,
                            default='./datasets/citys/leftImg8bit/test/berlin/berlin_000000_000019_leftImg8bit.png',
                            help='path to the input picture')
        parser.add_argument('--input-pic-list', type=str,
                            default="",
                            help='path to the input picture list')
        parser.add_argument('--outdir', default='./test_result', type=str,
                            help='path to save the predict result')
        parser.add_argument('--weight-name', type=str,
                            default="",
                            help='custom weight name')

        parser.add_argument('--cpu', dest='cpu', action='store_true')
        parser.set_defaults(cpu=False)

        self.args = parser.parse_args()

    def inference(self, image_cv):
        start = time.time()
        
        image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        image = cv2.resize(image , (960,540))
        image_transformed = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image_transformed)
        pred = torch.argmax(outputs[0], 1).squeeze(0).cpu().data.numpy()
        mask = get_color_pallete(pred, self.args.dataset)

        pil_image = mask.convert('RGB') 
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 

        end = time.time()
        print("Time :",(end-start) * 10**3, "ms")
        return open_cv_image
        
    def image_process(self, image_ros, image_depth_ros):
        print("get image")
        image_cv = np.zeros((1,1,3), np.uint8)

        try:
            image_cv = self.bridge.imgmsg_to_cv2(image_ros, "bgr8")
        except CvBridgeError as e:
            print(e)
        
        try:
            image_depth_cv = self.bridge.imgmsg_to_cv2(image_depth_ros, "32FC1")
            image_depth_cv_array = np.array(image_depth_cv, dtype = np.dtype('f8'))
            image_depth_cv_norm = cv2.normalize(image_depth_cv_array, image_depth_cv_array, 0, 255, cv2.NORM_MINMAX)
        except CvBridgeError as e:
            print(e)
        
        if image_cv is not None and self.init_finished:
            image_out_cv = self.inference(image_cv)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(image_out_cv, "bgr8"))
            # self.pub_image.publish(self.bridge.cv2_to_imgmsg(image_cv, "bgr8"))
            print("inference image")

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    row_segmentation = SegmentationDemo()
    row_segmentation.run()

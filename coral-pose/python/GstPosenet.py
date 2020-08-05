import sys
sys.path.append('~/coral/project-posenet')
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
import cairo
import cv2
import time
from gi.repository import Gst, GObject, GstBase, GstVideo
from contextlib import contextmanager
from ctypes import *
from typing import Tuple
import os
from fractions import Fraction

import numpy as np
import math
from PIL import Image
from project.pose_engine import PoseEngine


Gst.init(None)
FIXED_CAPS = Gst.Caps.from_string('video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]')



class GstPosenet(GstBase.BaseTransform):

    def __init__(self):
        print("init")
        self.Q = None
        self.width = -1
        self.height = -1
        self.calibration = None
        self.engine = PoseEngine('python/project/models/mobilenet/posenet_mobilenet_v1_075_721_1281_quant_decoder_edgetpu.tflite')
        self.max_pose_detections = 20
        self.lcameramtx = None
        self.ldist = None
        self.lrectification = None
        self.lprojection = None
        self.rcameramtx = None
        self.rdist = None
        self.rrectification = None
        self.rprojection = None
    
    __gstmetadata__ = ('GstPosenet Python','Transform',
                      'example gst-python element that can modify the buffer gst-launch-1.0 videotestsrc ! TestTransform2 ! videoconvert ! xvimagesink', 'dkl')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS))
    __gproperties__ = {
        "calibration": (str,
                  "calibration",
                  "Calibration file",
                  None,
                  GObject.ParamFlags.READWRITE)
    }
    
                
    def calculate_height(self, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):

        points2d = np.array([[lx1, ly1, lx1-rx1, 1], [lx2, ly2, lx2-rx2, 1]], dtype=np.float32).T
        points3d = self.Q.dot(points2d)
        x1 = points3d[0][0]/points3d[3][0]
        y1 = points3d[1][0]/points3d[3][0]
        z1 = points3d[2][0]/points3d[3][0]

        x2 = points3d[0][1]/points3d[3][1]
        y2 = points3d[1][1]/points3d[3][1]
        z2 = points3d[2][1]/points3d[3][1]

        #print(points2d)
       # print(points3d)

        #print((x1, y1, z1), (x2, y2, z2))


        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return str(distance)
    
    def pose(self, img):
        print('pose')
        newimg = Image.fromarray(img, 'RGB')
        newimg = newimg.resize((1281, 721), Image.NEAREST)
        lefteye = None
        leftankle = None
        poses, inference_time = self.engine.DetectPosesInImage(np.uint8(newimg))
        print('Inference time: %.fms' % inference_time)

        for pose in poses:
            if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                #print(' %-20s x=%-4d y=%-4d score=%.1f' %
                   # (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
                cv2.circle(img, (int(keypoint.yx[1]/2), int(keypoint.yx[0]*2)), 1, 255, 3)
                if label == 'left eye':
                    print('leftEye')
                    lefteye = keypoint.yx
                if label == 'left ankle':
                    print('leftAnkle')
                    leftankle = keypoint.yx
        
        return (lefteye, leftankle)


        

        
    
    def do_set_property(self, prop, value):
        if prop.name == 'calibration':
            self.calibration = value
            self.Q = np.load(self.calibration)["Q"]
            self.lcameramtx = np.load(self.calibration)["lcameramtx"]
            self.ldist = np.load(self.calibration)["ldist"]
            self.lrectification = np.load(self.calibration)["lrectification"]
            self.lprojection = np.load(self.calibration)["lprojection"]
            self.rcameramtx = np.load(self.calibration)["rcameramtx"]
            self.rdist = np.load(self.calibration)["rdist"]
            self.rrectification = np.load(self.calibration)["rrectification"]
            self.rprojection = np.load(self.calibration)["rprojection"]


    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True



    def do_transform_ip(self, buf):
        try:
            #print(type(buf))
            (_, info) = buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)
            #print(type(info))
            A = np.ndarray(buf.get_size(), dtype = np.uint8, buffer = info.data)
            A = A.reshape(self.height, self.width, 3).squeeze()
            limg = A[:, :self.width//2 - 1]
            rimg = A[:, self.width//2:]
            img = A
            lpose = self.pose(limg)
            rpose = self.pose(rimg)
            if lpose[0] is not None:
                print(lpose)
                ([ly1, lx1], [ly2, lx2]) = lpose
            else:
                #print(lpose)
                #print('lpose')
                ly1 = lx1 = ly2 = lx2 = 0
            if rpose[0] is not None:
                ([ry1, rx1], [ry2, rx2]) = rpose
            else:
                ry1 = ry2 = rx1 = rx2 = 0
            lpoints = np.array([[lx1/2, ly1*2], [lx2/2, ly2*2]], dtype=np.float32)
            rpoints = np.array([[rx1/2, ry1*2], [rx2/2, ry2*2]], dtype=np.float32)
            #print(self.lcameramtx)
            #print(self.ldist)
            #print(self.lrectification)
            #print(self.lprojection)
            #print(lpoints)
            lpoints = cv2.undistortPoints(lpoints, self.lcameramtx, self.ldist, None, self.lrectification, self.lprojection)
            rpoints = cv2.undistortPoints(rpoints, self.rcameramtx, self.rdist, None, self.rrectification, self.rprojection)
            lx1 = lpoints[0][0][0]
            ly1 = lpoints[0][0][1]
            lx2 = lpoints[1][0][0]
            ly2 = lpoints[1][0][1]
            rx1 = rpoints[0][0][0]
            ry1 = rpoints[0][0][1]
            rx2 = rpoints[1][0][0]
            ry2 = rpoints[1][0][1]
            print(self.calculate_height(lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2))
            return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR







GObject.type_register(GstPosenet)
__gstelementfactory__ = ("GstPosenet", Gst.Rank.NONE, GstPosenet)


import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstBase', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('Rsvg', '2.0')
import cairo
import cv2
import time
from gi.repository import Gst, GObject, GstBase, GstVideo, Rsvg
from contextlib import contextmanager
from ctypes import *
from typing import Tuple
import os
from fractions import Fraction
import tensorflow as tf
import posenet
from posenet.posenet_factory import load_model
from posenet.utils import draw_skel_and_kp
import posenet.converter.config as config 
import numpy as np
import math


Gst.init(None)
FIXED_CAPS = Gst.Caps.from_string('video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]')



class GstPosenet(GstBase.BaseTransform):

    def __init__(self):
        self.weightsPath = "yolo/yolov3.weights"
        self.configPath = "yolo/yolov3.cfg"
        self.LABELS = open("yolo/coco.names").read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)

        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.Q = None
        self.width = -1
        self.height = -1
        self.calibration = None
        self.pnet = load_model(config.RESNET50_MODEL, 16, 4, 1.0)
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
    def object_detection(self, img):
        blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                        swapRB=True, crop=False)
        self.net.setInput(blob)
        start = time.time()
        layerOutputs = self.net.forward(self.ln)
        end = time.time()

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:

            for detection in output:

                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > 0.5:

                    box = detection[0:4] * np.array([self.width//2, self.height, self.width//2, self.height])
                    (centerX, centerY, w, g) = box.astype("int")

                    x = int(centerX - (w / 2))
                    y = int(centerY - (g / 2))

                    boxes.append([x, y, int(w), int(g)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
                    

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
            0.3)



        i = confidences.index(max(confidences))
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
        text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
        cv2.putText(img, text, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)
        
        return (x, y, w, h)

                
    def calculate_height(self, lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2):
        #lx1 = lx2 = lx + lw//2
        #ly1 = ry1 = ly
        #ly2 = ry2 = ly + lh
        #rx1 = rx2 = rx + rw//2


        points2d = np.array([[lx1, ly1, lx1-rx1, 1], [lx2, ly2, lx2-rx2, 1]], dtype=np.float32).T

        points3d = self.Q.dot(points2d)
        x1 = points3d[0][0]/points3d[3][0]
        y1 = points3d[1][0]/points3d[3][0]
        z1 = points3d[2][0]/points3d[3][0]

        x2 = points3d[0][1]/points3d[3][1]
        y2 = points3d[1][1]/points3d[3][1]
        z2 = points3d[2][1]/points3d[3][1]




        distance = np.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
        return str(distance)
    
    def pose(self, img):
        lefteye = None
        leftankle = None
        pose_scores, keypoint_scores, keypoint_coords = self.pnet.estimate_multiple_poses(img, self.max_pose_detections)
        keypoint_coords_upscaled = keypoint_coords / 1.0
        overlay_frame = draw_skel_and_kp(
            img, pose_scores, keypoint_scores, keypoint_coords_upscaled,
            min_pose_score=0.15, min_part_score=0.1)
        
        for pi in range(len(pose_scores)):
            if pose_scores[pi] == 0.:
                break
            print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
            for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                #print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                cv2.circle(img, (int(c[1]), int(c[0])), 1, 255, 3)
                if posenet.PART_NAMES[ki] == 'leftEye':

                    lefteye = c
                if posenet.PART_NAMES[ki] == 'leftAnkle':

                    leftankle = c
        cv2.imwrite('file.jpg', img)
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

            info = buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE)

            A = np.ndarray(buf.get_size(), dtype = np.uint8, buffer = info.data)
            A = A.reshape(self.height, self.width, 3).squeeze()
            limg = A[:, :self.width//2 - 1]
            rimg = A[:, self.width//2:]
            img = A

            cv2.imwrite('asdf.jpg', rimg)
            #(lx, ly, lw, lh) = self.object_detection(limg)
            #(rx, ry, rw, rh) = self.object_detection(rimg)
            lr = np.rot90(limg)
            rr = np.rot90(rimg)
            cv2.imwrite('r.jpg', lr)
            lpose = self.pose(lr)
            rpose = self.pose(rr)
            if lpose[0] is not None:

                ([lx1, ly1], [lx2, ly2]) = lpose
            else:
                ly1 = lx1 = ly2 = lx2 = 0
            if rpose[0] is not None:
                ([rx1, ry1], [rx2, ry2]) = rpose
            else:
                ry1 = ry2 = rx1 = rx2 = 0

            lpoints = np.array([[lx1, ly1], [lx2, ly2]], dtype=np.float32)
            rpoints = np.array([[rx1, ry1], [rx2, ry2]], dtype=np.float32)
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


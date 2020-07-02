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

import numpy as np
import math

Gst.init(None)
FIXED_CAPS = Gst.Caps.from_string('video/x-raw,format=RGB,width=[1,2147483647],height=[1,2147483647]')



class ExampleTransform(GstBase.BaseTransform):

    def __init__(self):
        print("init")
        self.weightsPath = "yolo/yolov3.weights"
        self.configPath = "yolo/yolov3.cfg"
        self.LABELS = open("yolo/coco.names").read().strip().split("\n")
        self.net = cv2.dnn.readNetFromDarknet(self.configPath, self.weightsPath)
        print(self.net)
        print(type(self.net))
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        self.writer = None
    
    __gstmetadata__ = ('TestTransform2 Python','Transform',
                      'example gst-python element that can modify the buffer gst-launch-1.0 videotestsrc ! TestTransform2 ! videoconvert ! xvimagesink', 'dkl')

    __gsttemplates__ = (Gst.PadTemplate.new("src",
                                           Gst.PadDirection.SRC,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS),
                       Gst.PadTemplate.new("sink",
                                           Gst.PadDirection.SINK,
                                           Gst.PadPresence.ALWAYS,
                                           FIXED_CAPS))

    def do_set_caps(self, incaps, outcaps):
        struct = incaps.get_structure(0)
        self.width = struct.get_int("width").value
        self.height = struct.get_int("height").value
        return True



    def do_transform_ip(self, buf):
        try:
            with buf.map(Gst.MapFlags.READ | Gst.MapFlags.WRITE) as info:

                A = np.ndarray(buf.get_size(), dtype = np.uint8, buffer = info.data)
                A = A.reshape(360, 640, 3).squeeze()

                img = A
                print(self.height)
                cv2.imwrite('asdf.jpg', img)
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

                            box = detection[0:4] * np.array([640, 360, 640, 360])
                            (centerX, centerY, width, height) = box.astype("int")

                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            classIDs.append(classID)
                            cv2.imwrite('img.jpg', img)

                idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                    0.3)

                if len(idxs) > 0:
                    for i in idxs.flatten():
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])

                        cv2.rectangle(img, (x, y), (x + w, y + h), 0, 2)
                        text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                            confidences[i])
                        cv2.putText(img, text, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)

                if self.writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                    self.writer = cv2.VideoWriter("output.avi", fourcc, 30,
                        (img.shape[1], img.shape[0]), True)

                self.writer.write(img)

                return Gst.FlowReturn.OK
        except Gst.MapError as e:
            Gst.error("Mapping error: %s" % e)
            return Gst.FlowReturn.ERROR



GObject.type_register(ExampleTransform)
__gstelementfactory__ = ("TestTransform2", Gst.Rank.NONE, ExampleTransform)

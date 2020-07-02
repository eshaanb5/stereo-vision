# Test to see how well the calibration worked.
# Arguments are calibration output file, path to left image, and path to right image

import sys
import numpy
import cv2

cachefile = sys.argv[1]
limgpath = sys.argv[2]
rimgpath = sys.argv[3]
cache = numpy.load(cachefile)
limg = cv2.imread(limgpath)
rimg = cv2.imread(rimgpath)
lmapx = cache["lmapx"]
lmapy = cache["lmapy"]
rmapx = cache["rmapx"]
rmapy = cache["rmapy"]


stereoMatcher = cv2.StereoSGBM_create()
stereoMatcher.setMinDisparity(4)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(21)
stereoMatcher.setSpeckleRange(16)
stereoMatcher.setSpeckleWindowSize(45)



fixedLeft = cv2.remap(limg, lmapx, lmapy, cv2.INTER_NEAREST)
fixedRight = cv2.remap(rimg, rmapx, rmapy, cv2.INTER_NEAREST)
#fixedRight = fixedRight[300:2000, 200: ]
#fixedLeft = fixedLeft[300:2000, 200: ]



cv2.imshow('fixedleft', fixedLeft)
cv2.waitKey(0)
cv2.imshow('fixedright', fixedRight)
cv2.waitKey(0)
grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
depth = stereoMatcher.compute(grayLeft, grayRight)
cv2.imshow('depth', depth/2048)
cv2.waitKey(0)
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
Q = cache["Q"]


lx1, ly1, rx1, ry1, lx2, ly2, rx2, ry2 = -1, -1, -1, -1, -1, -1, -1, -1
lc, rc = 0, 0

def left_coordinates(event, x, y, flags, param):
    global lx1, ly1, lx2, ly2, lc
    if event == cv2.EVENT_LBUTTONDOWN:
        if lc == 0:
            lx1 = float(x)
            ly1 = float(y)
        elif lc == 1:
            lx2 = float(x)
            ly2 = float(y)
        lc += 1
        print((x, y))
def right_coordinates(event, x, y, flags, param):
    global rx1, ry1, rx2, ry2, rc
    if event == cv2.EVENT_LBUTTONDOWN:
        if rc == 0:
            rx1 = float(x)
            ry1 = float(y)
        elif rc == 1:
            rx2 = float(x)
            ry2 = float(y)
            cv2.destroyWindow('right')
        rc += 1
        print((x, y))


cv2.namedWindow('left')
cv2.namedWindow('right')
cv2.setMouseCallback('left', left_coordinates)
cv2.setMouseCallback('right', right_coordinates)
fixedLeft = cv2.remap(limg, lmapx, lmapy, cv2.INTER_NEAREST)
fixedRight = cv2.remap(rimg, rmapx, rmapy, cv2.INTER_NEAREST)
cv2.imshow('left', fixedLeft)
cv2.waitKey(0)
cv2.destroyWindow('left')
cv2.imshow('right', fixedRight)
cv2.waitKey(0)


points2d = numpy.array([[lx1, ly1, lx1-rx1, 1], [lx2, ly2, lx2-rx2, 1]], dtype=numpy.float32).T

points3d = Q.dot(points2d)
x1 = points3d[0][0]/points3d[3][0]
y1 = points3d[1][0]/points3d[3][0]
z1 = points3d[2][0]/points3d[3][0]

x2 = points3d[0][1]/points3d[3][1]
y2 = points3d[1][1]/points3d[3][1]
z2 = points3d[2][1]/points3d[3][1]


distance = numpy.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
distance = distance
print(distance)
cv2.destroyAllWindows()
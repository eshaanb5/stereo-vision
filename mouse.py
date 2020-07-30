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

lcameramtx = cache["lcameramtx"]
rcameramtx = cache["rcameramtx"]
ldist = cache["ldist"]
rdist = cache["rdist"]
lrectification = cache["lrectification"]
rrectification = cache["rrectification"]
lprojection = cache["lprojection"]
rprojection = cache["rprojection"]






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
        rc += 1
        print((x, y))


print(Q)



#Q = numpy.array([[1., 0., 0., -1228.47],[0., 1., 0., -451.838],[0., 0., 0., 1390.31],[0., 0., 7.69789, 0.]])
print(Q)
cv2.namedWindow('left')
cv2.namedWindow('right')
cv2.setMouseCallback('left', left_coordinates)
cv2.setMouseCallback('right', right_coordinates)
print(limg)
print('lmapx')
print(lmapx)
print(lmapy)
fixedLeft = limg #cv2.remap(limg, lmapx, lmapy, cv2.INTER_NEAREST)
print(fixedLeft)
fixedRight = rimg #cv2.remap(rimg, rmapx, rmapy, cv2.INTER_NEAREST)
cv2.imshow('left', fixedLeft)
cv2.waitKey(0)
cv2.destroyWindow('left')
cv2.imshow('right', fixedRight)
cv2.waitKey(0)
cv2.destroyWindow('right')

lpoints = numpy.array([[lx1, ly1], [lx2, ly2]], dtype=numpy.float32)
rpoints = numpy.array([[rx1, ry1], [rx2, ry2]], dtype=numpy.float32)
lpoints = cv2.undistortPoints(lpoints, lcameramtx, ldist, None, lrectification, lprojection)
rpoints = cv2.undistortPoints(rpoints, rcameramtx, rdist, None, rrectification, rprojection)
print(lpoints)

lx1 = lpoints[0][0][0]
ly1 = lpoints[0][0][1]
lx2 = lpoints[1][0][0]
ly2 = lpoints[1][0][1]
rx1 = rpoints[0][0][0]
ry1 = rpoints[0][0][1]
rx2 = rpoints[1][0][0]
ry2 = rpoints[1][0][1]



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
print([[x1, y1, z1],[x2, y2, z2]])
points2d = numpy.array([[lx1, ly1, lx1-rx1], [lx2, ly2, lx2-rx2]], dtype=numpy.float32)


points3d = cv2.perspectiveTransform(points2d[None, :, :], Q)
x1 = points3d[0][0][0]
y1 = points3d[0][0][1]
z1 = points3d[0][0][2]

x2 = points3d[0][1][0]
y2 = points3d[0][1][1]
z2 = points3d[0][1][2]


distance = numpy.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)
distance = distance
print(distance)
print(points3d)

cv2.destroyAllWindows()

import cv2
import numpy as np

K = np.array([[657.7530732583206, 0.0, 495.51785347349863], [0.0, 660.9636878760934, 271.60650499453703], [0.0, 0.0, 1.0]])
D = np.array([[-0.10638295935397933], [-0.2437635745576714], [0.705424472804052], [-0.6260109194284841]])


img = cv2.imread('./chess_markers/5.jpg')
h, w = img.shape[:2]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, (960, 540), cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



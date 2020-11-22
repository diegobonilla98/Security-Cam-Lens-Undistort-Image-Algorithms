import cv2
import numpy as np
import matplotlib.pyplot as plt

# camera matrix K. The real one (for some reason) is the transpose. [[fx, 0, 0], [s, fy, 0], [cx, cy, 1]]
K = np.array([[1283.2471531112978, 0.0,                961.3034904651593],
              [0.0,                1288.6515127343948, 517.3701696454254],
              [0.0,                0.0,                1.0]])

# distortion coefficients D. ¿¿ k1, k2, p1, p2 ??
D = np.array([[-0.12775519839463853], [-0.02714832019687175], [0.5168356372126586], [-0.0076847044003239]])

cam = cv2.VideoCapture('rtsp://link')
cv2.namedWindow("Result", cv2.WINDOW_NORMAL)

DIM = (1920, 1080)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)

while True:
    ret, frame = cam.read()

    frame = cv2.resize(frame, DIM)

    h, w = frame.shape[:2]

    undistorted_img = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

    result = np.hstack([frame, undistorted_img])
    cv2.imshow("Result", result)
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
cam.release()

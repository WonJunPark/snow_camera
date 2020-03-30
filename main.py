import cv2, dlib, sys
import numpy as np
from overlay_pre import overlay_processing

scaler = 0.3

detector = dlib.get_frontal_face_detector()
# 68개의 얼굴 특징점 추출
predictor = dlib.shape_predictor('face_landmark.dat')
# load video
cap = cv2.VideoCapture('samples/girl.mp4')
# load overlay image
overlay = cv2.imread('samples/Apeach.png', cv2.IMREAD_UNCHANGED)

while True:
    ret, img = cap.read()
    print(ret)
    if not ret:
        break

    # 이미지 크기 조정
    img = cv2.resize(img, (int(img.shape[1]*scaler), int(img.shape[0]*scaler)))
    ori = img.copy()

    # 얼굴 인식
    faces = detector(img)
    # 여러 얼굴이 나오기 때문에 얼굴 한개만 지정
    face = faces[0]

    # 얼굴 특징점 추출
    dlib_shape = predictor(img, face)
    shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])

    # compute center and boundaries of face / 얼굴 중심을 구함
    top_left = np.min(shape_2d, axis=0)
    bottom_right = np.max(shape_2d, axis=0)

    face_size = int(max(bottom_right - top_left) * 1.8)

    center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

    result = overlay_processing.overlay_transparent(ori, overlay, center_x+25, center_y-25, overlay_size=(face_size, face_size))

    # visualize / 시각화
    img = cv2.rectangle(img, pt1=(face.left(), face.top()), pt2=(face.right(), face.bottom()), color=(255, 255, 255),
    thickness=2, lineType=cv2.LINE_AA)

    for s in shape_2d:
        cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow('result',result)
    cv2.waitKey(1)

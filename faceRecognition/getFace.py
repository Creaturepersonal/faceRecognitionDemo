import dlib
import cv2
import numpy as np

dlib_face_detector = dlib.get_frontal_face_detector()

FaceNum = 1
FaceLibPath = 'D:\Comprehensive3\FaceRecognition\FaceLib\\train\\'


def FaceDetecttion(img):
    dets = dlib_face_detector(img, 1)
    if len(dets) != 0:
        for detection in dets:
            img = cv2.rectangle(img,
                                (detection.left(), detection.top()),  # (x1,y1)
                                (detection.right(), detection.bottom()),  # (x2,y2)
                                (255, 255, 255),
                                2)
            img = img[detection.top():detection.bottom(), detection.left():detection.right()]

    else:
        print('未检测到人脸')
    return img

#检测到人脸图像之后进行裁剪
def FaceDetecttionCut(img):
    dets = dlib_face_detector(img, 1)
    if len(dets) != 0:
        for detection in dets:
            img = img[detection.top():detection.bottom(), detection.left():detection.right()]
            return img
    else:
        return np.array([0])

#将人脸图像进行保存
def GetFace():
    global FaceNum
    global FaceLibPath
    cap = cv2.VideoCapture(0)
    i = 0
    while True:
        ret, frame = cap.read()
        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        if ret == True and i == 5:
            i = 0
            image = frame
            dets = dlib_face_detector(image, 1)
            if len(dets) != 0:
                image = FaceDetecttionCut(image)
                image = cv2.resize(image, (128, 128))
                cv2.imwrite(FaceLibPath + '19_' + str(FaceNum) + '.bmp', image)
                if FaceNum == 50:#设置一次收集的图片数量
                    break
                FaceNum += 1
        i += 1


def main():
    GetFace()

if __name__ == '__main__':
    main()

import cv2

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(320))
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(240))

cv2.namedWindow("Take a photo")



img_counter = 417

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Take a photo", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "D:/Studia_PJATK/FruitsRecognition/FruitsRecogintion/MyTestImages/MyCreatedPhotos/tomato/tomato_{}.png".format(img_counter)
        frame = cv2.resize(frame, (480, 322),interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()
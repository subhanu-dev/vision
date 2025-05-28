import cv2
import numpy as np
from util import get_limits
from PIL import Image

cap = cv2.VideoCapture(0)  # 0 means the default camera of the system
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)  # setting the width of the frame
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # setting the height of the frame
# _, frame = cap.read()

# cv2.imshow("frame", frame)  # displaying  the captured frame , this only captures a single frame so in the bottom the frame is moved inside a loop
# cv2.waitKey(0)

# moving frame inside a loop to capture multiple frames

while True:
    _, frame = cap.read()
    hsv_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,  # convert color format to HSV from default RGB
    )

    height, width, _ = frame.shape  # getting the height and width of the frame
    cx = int(width / 2)
    cy = int(height / 2)

    # pick pixel value
    pixel_center = hsv_frame[cy, cx]
    print(pixel_center)

    hue_value = pixel_center[0]  # extracting the hue value from the pixel value
    saturation = pixel_center[1]  # extracting the saturation value from the pixel value

    # defining the range of hue value
    if hue_value > 200 and saturation < 50:  # High brightness and low saturation
        color = "white"
    elif hue_value < 5 or hue_value > 170:  # Red wraps around in HSV
        color = "red"
    elif hue_value < 15:
        color = "orange"
    elif hue_value < 33:
        color = "yellow"
    elif hue_value < 70:
        color = "green"
    elif hue_value < 131:
        color = "blue"
    elif hue_value < 170:
        color = "pink"
    else:
        color = "black"

    # extracting the hue value from the pixel value
    pixel_center_bgr = frame[cy, cx]  # extracting the bgr value from the pixel value
    b, g, r = (
        int(pixel_center_bgr[0]),
        int(pixel_center_bgr[1]),
        int(pixel_center_bgr[2]),
    )

    cv2.circle(frame, (cx, cy), 5, (b, g, r), 1)
    cv2.putText(
        frame,
        color,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (b, g, r),
        2,
    )  # putting the text on the frame

    yellow = [0, 255, 255]  # yellow in BGR colorspace

    hsv_image = cv2.cvtColor(
        frame, cv2.COLOR_BGR2HSV
    )  # converting the color format to HSV from default RGB

    lowerlimit, upperlimit = get_limits(color=yellow)

    mask = cv2.inRange(hsv_image, lowerlimit, upperlimit)

    mask_ = Image.fromarray(mask)
    bbox = mask_.getbbox()

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)  # waits for a key press for 1 millisecond
        if key == 27:  # if the key pressed is 'Esc' (ASCII value 27)
            break  # exit the loop


cap.release()  # releasing the camera
cv2.destroyAllWindows()  # closing all the opened windows


# ------------------images-----------------------------------
# img = cv2.imread("white.png")  # reading the locally stored image
# cv2.imshow("image", img)
# cv2.waitKey(
#     0
# )  # function waits for a key press from the user. The argument 0 means it will wait indefinitely until a key is pressed. If a non-zero value is passed, it specifies the delay in milliseconds to wait for a key press.


# print(img)  # printing the image array
# print(type(img))  # printing the type of the image array
# print(img.shape)  # printing the shape of the image array


# ----------------------color detecting with the use of HSV color space -------------------

# yellow = [0, 255, 255]  # yellow in BGR colorspace

# while True:
#     ret, frame = cap.read()

#     hsv_image = cv2.cvtColor(
#         frame, cv2.COLOR_BGR2HSV
#     )  # converting the color format to HSV from default RGB

#     lowerlimit, upperlimit = get_limits(color=yellow)

#     mask = cv2.inRange(hsv_image, lowerlimit, upperlimit)

#     mask_ = Image.fromarray(mask)
#     bbox = mask_.getbbox()

#     if bbox is not None:
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

#     cv2.imshow("frame", frame)

#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

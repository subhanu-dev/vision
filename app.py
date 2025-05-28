from ultralytics import YOLO
import cv2
import time

# Load a pre-trained YOLOv8 model (nano version for speed)
# model = YOLO("yolov8n.pt")
model = YOLO("yolo11s.pt")  # yolo v11
# model = YOLO("yolov8s-oiv7.pt")  # the model trained on openimages dataset v7

# Open webcam
cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run YOLO inference
#     results = model(frame)

#     # Visualize results on the frame
#     annotated_frame = results[0].plot()

#     cv2.imshow("YOLO Detection", annotated_frame)
#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()

############ same thing as above but with FPS and number of objects detected

prev_time = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Visualize results on the frame
    annotated_frame = results[0].plot()

    # Count number of objects detected
    num_objects = len(results[0].boxes)

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # Display number of objects and FPS on the frame
    cv2.putText(
        annotated_frame,
        f"Objects: {num_objects}  FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

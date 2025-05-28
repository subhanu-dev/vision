from ultralytics import YOLO
import cv2
from collections import defaultdict
import time

model = YOLO("yolov8n.pt")  # Pre-trained on COCO (includes 'person' class)
cap = cv2.VideoCapture(0)
line_x = 130  # Vertical counting line (adjust to your needs)

# Track crossing states: {track_id: "left"/"right"}
tracked_ids = defaultdict(str)
entry_count, exit_count = 0, 0

# Cooldown mechanism to prevent double-counting
cooldowns = defaultdict(int)  # {track_id: frame number when last counted}
cooldown_frames = (
    8  # Number of frames to wait before allowing another count for the same ID
)
frame_count = 0

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame, persist=True, tracker="bytetrack.yaml"
    )  # Enable tracking
    annotated_frame = results[0].plot()

    # Draw counting line
    cv2.line(annotated_frame, (line_x, 0), (line_x, frame.shape[0]), (255, 0, 0), 2)

    if results[0].boxes.id is not None:  # Check if tracking IDs exist
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy()

        for box, track_id, cls in zip(boxes, track_ids, clss):
            if results[0].names[int(cls)] == "person":
                x1, y1, x2, y2 = box
                cx = (x1 + x2) // 2

                # Cooldown check
                if frame_count - cooldowns[track_id] < cooldown_frames:
                    continue  # Skip if still in cooldown

                # Check if person crossed the line
                if track_id in tracked_ids:
                    if tracked_ids[track_id] == "left" and cx > line_x:
                        entry_count += 1
                        tracked_ids[track_id] = "right"
                        cooldowns[track_id] = frame_count
                    elif tracked_ids[track_id] == "right" and cx < line_x:
                        exit_count += 1
                        tracked_ids[track_id] = "left"
                        cooldowns[track_id] = frame_count
                else:
                    # Initialize tracking state
                    tracked_ids[track_id] = "left" if cx < line_x else "right"

    # Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if curr_time != prev_time else 0
    prev_time = curr_time

    # Display counts
    cv2.putText(
        annotated_frame,
        f"Entries: {entry_count} | Exits: {exit_count} | FPS: {fps:.2f}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("People Counter", annotated_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

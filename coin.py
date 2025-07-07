import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class CashDetector:
    def __init__(self):
        # Initialize YOLO model - you'll need to train this on cash dataset
        # For now, using a general model and we'll simulate cash detection
        self.model = YOLO("yolov8n.pt")

        # Define cash values (you'll need to customize based on your currency)
        self.coin_values = {
            "coin_1": 0.01,  # 1 cent
            "coin_5": 0.05,  # 5 cents
            "coin_10": 0.10,  # 10 cents
            "coin_25": 0.25,  # 25 cents
            "coin_50": 0.50,  # 50 cents
            "coin_100": 1.00,  # 1 dollar coin
        }

        self.note_values = {
            "note_1": 1.00,  # 1 dollar
            "note_5": 5.00,  # 5 dollars
            "note_10": 10.00,  # 10 dollars
            "note_20": 20.00,  # 20 dollars
            "note_50": 50.00,  # 50 dollars
            "note_100": 100.00,  # 100 dollars
        }

        # Combine all cash types
        self.all_cash_values = {**self.coin_values, **self.note_values}

        # Detection confidence threshold
        self.confidence_threshold = 0.5

    def detect_cash_by_color_size(self, frame):
        """
        Alternative method: Detect coins/notes using color and size analysis
        This is a backup method when YOLO model isn't available
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []

        # Define color ranges for different denominations (example for US currency)
        color_ranges = {
            "coin_penny": ([5, 50, 50], [15, 255, 255]),  # Copper color
            "coin_nickel": ([0, 0, 100], [180, 30, 200]),  # Silver color
            "coin_dime": ([0, 0, 150], [180, 30, 255]),  # Bright silver
            "coin_quarter": ([0, 0, 120], [180, 40, 220]),  # Silver-white
        }

        for cash_type, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            # Create mask
            mask = cv2.inRange(hsv, lower, upper)

            # Find contours
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)

                # Filter by size (coins should be roughly circular and within size range)
                if 500 < area < 5000:  # Adjust based on your camera distance
                    # Check if contour is roughly circular
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)

                    if circularity > 0.6:  # Reasonably circular
                        x, y, w, h = cv2.boundingRect(contour)
                        detections.append(
                            {
                                "type": cash_type,
                                "bbox": (x, y, x + w, y + h),
                                "confidence": circularity,
                                "area": area,
                            }
                        )

        return detections

    def process_video(self, video_path):
        """Process uploaded video and detect cash"""
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            return None, "Error: Could not open video file"

        total_cash = 0.0
        detected_items = defaultdict(int)
        frame_count = 0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing video: {total_frames} frames at {fps} FPS")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 10th frame to speed up processing
            if frame_count % 10 != 0:
                continue

            # Try YOLO detection first (will need custom trained model)
            # For now, using color/size detection as backup
            detections = self.detect_cash_by_color_size(frame)

            # Draw detections and calculate value
            annotated_frame = frame.copy()
            frame_total = 0.0

            for detection in detections:
                cash_type = detection["type"]
                bbox = detection["bbox"]
                confidence = detection["confidence"]

                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = bbox

                    # Map detected type to value
                    value = self.get_cash_value(cash_type)
                    detected_items[cash_type] += 1
                    frame_total += value

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with value
                    label = f"{cash_type}: ${value:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Update total (avoid double counting)
            if frame_total > total_cash:
                total_cash = frame_total

            # Display current frame info
            cv2.putText(
                annotated_frame,
                f"Frame Total: ${frame_total:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Max Total: ${total_cash:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_count}/{total_frames}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            # Show frame
            cv2.imshow("Cash Detection", annotated_frame)

            # Break on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        return total_cash, detected_items

    def get_cash_value(self, cash_type):
        """Get monetary value for detected cash type"""
        # Map color-based detection to actual values
        value_mapping = {
            "coin_penny": 0.01,
            "coin_nickel": 0.05,
            "coin_dime": 0.10,
            "coin_quarter": 0.25,
        }

        return value_mapping.get(cash_type, 0.0)

    def process_live_camera(self):
        """Process live camera feed for cash detection"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open camera")
            return

        print("Starting live cash detection. Press 'q' to quit, 'r' to reset count.")

        total_cash = 0.0
        detected_items = defaultdict(int)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect cash in current frame
            detections = self.detect_cash_by_color_size(frame)

            # Draw detections
            annotated_frame = frame.copy()
            frame_total = 0.0

            for detection in detections:
                cash_type = detection["type"]
                bbox = detection["bbox"]
                confidence = detection["confidence"]

                if confidence > self.confidence_threshold:
                    x1, y1, x2, y2 = bbox
                    value = self.get_cash_value(cash_type)
                    frame_total += value

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label = f"{cash_type}: ${value:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            # Update total
            if frame_total > total_cash:
                total_cash = frame_total

            # Display info
            cv2.putText(
                annotated_frame,
                f"Current: ${frame_total:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Max Total: ${total_cash:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                "Press 'q' to quit, 'r' to reset",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Live Cash Detection", annotated_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                total_cash = 0.0
                detected_items.clear()
                print("Count reset!")

        cap.release()
        cv2.destroyAllWindows()

        return total_cash, detected_items


def main():
    detector = CashDetector()

    print("Cash Detection System")
    print("1. Process uploaded video")
    print("2. Live camera detection")

    choice = input("Select option (1 or 2): ").strip()

    if choice == "1":
        video_path = input("Enter path to video file: ").strip()
        print(f"Processing video: {video_path}")

        total_cash, detected_items = detector.process_video(video_path)

        if total_cash is not None:
            print("\n=== DETECTION RESULTS ===")
            print(f"Total Cash Value: ${total_cash:.2f}")
            print(f"Detected items: {dict(detected_items)}")
        else:
            print("Error processing video")

    elif choice == "2":
        total_cash, detected_items = detector.process_live_camera()
        print("\n=== FINAL RESULTS ===")
        print(f"Total Cash Value: ${total_cash:.2f}")
        print(f"Detected items: {dict(detected_items)}")

    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

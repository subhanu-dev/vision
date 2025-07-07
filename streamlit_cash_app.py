import streamlit as st
import cv2
import numpy as np
from coin import CashDetector
import tempfile
import os
from PIL import Image
import time

# Configure Streamlit page
st.set_page_config(page_title="Cash Detection AI", page_icon="💰", layout="wide")

# Initialize session state
if "detector" not in st.session_state:
    st.session_state.detector = CashDetector()
if "total_cash" not in st.session_state:
    st.session_state.total_cash = 0.0
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False


def main():
    # Title and header
    st.title("💰 Cash Detection AI System")
    st.markdown("---")

    # Sidebar for options
    st.sidebar.title("Detection Options")
    detection_mode = st.sidebar.selectbox(
        "Choose Detection Mode:",
        ["📹 Video Upload", "📷 Live Camera", "🖼️ Image Upload"],
    )

    # Display current total
    st.sidebar.markdown("### 💵 Current Total")
    st.sidebar.metric("Cash Value", f"${st.session_state.total_cash:.2f}")

    if st.sidebar.button("🔄 Reset Count"):
        st.session_state.total_cash = 0.0
        st.success("Count reset to $0.00")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        if detection_mode == "📹 Video Upload":
            video_upload_interface()
        elif detection_mode == "📷 Live Camera":
            live_camera_interface()
        elif detection_mode == "🖼️ Image Upload":
            image_upload_interface()

    with col2:
        display_detection_info()


def video_upload_interface():
    st.header("📹 Video Upload Detection")

    uploaded_video = st.file_uploader(
        "Choose a video file", type=["mp4", "avi", "mov", "mkv", "wmv"]
    )

    if uploaded_video is not None:
        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        st.video(uploaded_video)

        col1, col2, col3 = st.columns(3)

        with col2:
            if st.button("🔍 Process Video", type="primary"):
                process_video(video_path)

        # Clean up temp file
        try:
            os.unlink(video_path)
        except Exception:
            pass


def live_camera_interface():
    st.header("📷 Live Camera Detection")

    # Camera controls
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("▶️ Start Camera", type="primary"):
            st.session_state.camera_running = True

    with col2:
        if st.button("⏹️ Stop Camera"):
            st.session_state.camera_running = False

    with col3:
        if st.button("📸 Take Screenshot"):
            take_screenshot()

    # Camera feed placeholder
    camera_placeholder = st.empty()

    if st.session_state.camera_running:
        run_live_camera(camera_placeholder)
    else:
        camera_placeholder.info("Click 'Start Camera' to begin live detection")


def image_upload_interface():
    st.header("🖼️ Image Upload Detection")

    uploaded_image = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        col1, col2, col3 = st.columns(3)

        with col2:
            if st.button("🔍 Detect Cash", type="primary"):
                process_image(image)


def process_video(video_path):
    """Process uploaded video and show progress"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    result_placeholder = st.empty()

    status_text.text("Processing video...")

    try:
        # Process video
        total_cash, detected_items = st.session_state.detector.process_video(video_path)

        if total_cash is not None:
            st.session_state.total_cash = total_cash
            progress_bar.progress(100)
            status_text.text("Processing complete!")

            # Display results
            with result_placeholder.container():
                st.success(f"💰 Total Cash Detected: ${total_cash:.2f}")

                if detected_items:
                    st.write("**Detected Items:**")
                    for item, count in detected_items.items():
                        st.write(f"- {item}: {count} pieces")
        else:
            st.error("❌ Error processing video")

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    finally:
        progress_bar.empty()
        status_text.empty()


def process_image(image):
    """Process uploaded image for cash detection"""
    # Convert PIL image to OpenCV format
    opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Detect cash
    detections = st.session_state.detector.detect_cash_by_color_size(opencv_image)

    # Calculate total value
    total_value = 0.0
    detected_items = {}

    # Draw detections on image
    annotated_image = opencv_image.copy()

    for detection in detections:
        cash_type = detection["type"]
        bbox = detection["bbox"]
        confidence = detection["confidence"]

        if confidence > st.session_state.detector.confidence_threshold:
            x1, y1, x2, y2 = bbox
            value = st.session_state.detector.get_cash_value(cash_type)
            total_value += value

            # Count items
            detected_items[cash_type] = detected_items.get(cash_type, 0) + 1

            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{cash_type}: ${value:.2f}"
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    # Update session state
    st.session_state.total_cash = total_value

    # Display results
    if total_value > 0:
        # Convert back to RGB for display
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        st.image(
            annotated_image_rgb, caption="Detection Results", use_column_width=True
        )

        st.success(f"💰 Total Cash Detected: ${total_value:.2f}")

        if detected_items:
            st.write("**Detected Items:**")
            for item, count in detected_items.items():
                st.write(f"- {item}: {count} pieces")
    else:
        st.warning("⚠️ No cash detected in the image")


def run_live_camera(placeholder):
    """Run live camera detection"""
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open camera. Please check your camera connection.")
        return

    # Create a container for the camera feed
    with placeholder.container():
        camera_col, info_col = st.columns([3, 1])

        with camera_col:
            frame_placeholder = st.empty()

        with info_col:
            info_placeholder = st.empty()

    frame_count = 0
    max_total = 0.0

    while st.session_state.camera_running:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Failed to capture frame from camera")
            break

        frame_count += 1

        # Process every 5th frame for performance
        if frame_count % 5 == 0:
            # Detect cash
            detections = st.session_state.detector.detect_cash_by_color_size(frame)

            # Draw detections
            annotated_frame = frame.copy()
            frame_total = 0.0
            detected_count = 0

            for detection in detections:
                cash_type = detection["type"]
                bbox = detection["bbox"]
                confidence = detection["confidence"]

                if confidence > st.session_state.detector.confidence_threshold:
                    x1, y1, x2, y2 = bbox
                    value = st.session_state.detector.get_cash_value(cash_type)
                    frame_total += value
                    detected_count += 1

                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    # Draw label
                    label = f"${value:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

            # Update max total
            if frame_total > max_total:
                max_total = frame_total
                st.session_state.total_cash = max_total

            # Add info overlay
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
                f"Max: ${max_total:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                f"Items: {detected_count}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

            # Display frame
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)

            # Display info
            with info_placeholder.container():
                st.metric("Current Frame", f"${frame_total:.2f}")
                st.metric("Max Detected", f"${max_total:.2f}")
                st.metric("Items Count", detected_count)

        # Small delay
        time.sleep(0.1)

    cap.release()


def take_screenshot():
    """Take a screenshot from camera"""
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Process the screenshot
            detections = st.session_state.detector.detect_cash_by_color_size(frame)

            # Calculate total
            total_value = 0.0
            for detection in detections:
                if (
                    detection["confidence"]
                    > st.session_state.detector.confidence_threshold
                ):
                    value = st.session_state.detector.get_cash_value(detection["type"])
                    total_value += value

            # Update total
            st.session_state.total_cash = total_value

            # Convert and display
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(rgb_frame, caption=f"Screenshot - Detected: ${total_value:.2f}")

            st.success(f"📸 Screenshot taken! Detected: ${total_value:.2f}")
        else:
            st.error("❌ Failed to capture screenshot")
    else:
        st.error("❌ Camera not available")

    cap.release()


def display_detection_info():
    """Display detection information and settings"""
    st.header("ℹ️ Detection Info")

    # Detection settings
    st.subheader("⚙️ Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.detector.confidence_threshold,
        step=0.1,
    )
    st.session_state.detector.confidence_threshold = confidence

    # Currency info
    st.subheader("💱 Supported Currency")
    st.write("**Coins:**")
    for coin, value in st.session_state.detector.coin_values.items():
        st.write(f"- {coin}: ${value:.2f}")

    st.write("**Notes:**")
    for note, value in st.session_state.detector.note_values.items():
        st.write(f"- {note}: ${value:.2f}")

    # Instructions
    st.subheader("📋 Instructions")
    st.write("""
    **Video Upload:**
    - Upload MP4, AVI, MOV, MKV, or WMV files
    - Click 'Process Video' to analyze
    
    **Live Camera:**
    - Click 'Start Camera' for real-time detection
    - Use 'Take Screenshot' to capture specific moments
    - Click 'Stop Camera' when done
    
    **Image Upload:**
    - Upload JPG, PNG, or BMP images
    - Click 'Detect Cash' to analyze
    
    **Tips:**
    - Ensure good lighting
    - Keep camera steady
    - Use contrasting backgrounds
    - Adjust confidence threshold if needed
    """)


if __name__ == "__main__":
    main()

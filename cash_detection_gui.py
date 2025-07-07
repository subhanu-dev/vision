import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import threading
from coin import CashDetector


class CashDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cash Detection System")
        self.root.geometry("800x600")

        # Initialize detector
        self.detector = CashDetector()

        # Variables
        self.video_path = tk.StringVar()
        self.total_cash = tk.DoubleVar()
        self.is_processing = False

        self.setup_ui()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Title
        title_label = ttk.Label(
            main_frame, text="Cash Detection System", font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Video upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Video Upload", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(upload_frame, text="Select video file:").grid(
            row=0, column=0, sticky=tk.W
        )

        ttk.Entry(upload_frame, textvariable=self.video_path, width=50).grid(
            row=0, column=1, padx=(10, 5), sticky=(tk.W, tk.E)
        )

        ttk.Button(upload_frame, text="Browse", command=self.browse_video).grid(
            row=0, column=2, padx=5
        )

        ttk.Button(
            upload_frame, text="Process Video", command=self.process_video_thread
        ).grid(row=1, column=1, pady=10)

        # Live camera section
        camera_frame = ttk.LabelFrame(main_frame, text="Live Camera", padding="10")
        camera_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Button(
            camera_frame, text="Start Live Detection", command=self.start_live_detection
        ).grid(row=0, column=0, padx=5)

        ttk.Button(camera_frame, text="Stop", command=self.stop_processing).grid(
            row=0, column=1, padx=5
        )

        # Results section
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(results_frame, text="Total Cash Value:").grid(
            row=0, column=0, sticky=tk.W
        )

        self.cash_label = ttk.Label(
            results_frame, text="$0.00", font=("Arial", 14, "bold"), foreground="green"
        )
        self.cash_label.grid(row=0, column=1, padx=10)

        ttk.Button(results_frame, text="Reset", command=self.reset_results).grid(
            row=0, column=2, padx=5
        )

        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # Status label
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=3)

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

    def browse_video(self):
        filename = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        if filename:
            self.video_path.set(filename)

    def process_video_thread(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first")
            return

        if self.is_processing:
            messagebox.showwarning("Warning", "Already processing. Please wait.")
            return

        # Start processing in separate thread
        thread = threading.Thread(target=self.process_video)
        thread.daemon = True
        thread.start()

    def process_video(self):
        self.is_processing = True
        self.progress.start()
        self.status_label.config(text="Processing video...")

        try:
            total_cash, detected_items = self.detector.process_video(
                self.video_path.get()
            )

            if total_cash is not None:
                self.total_cash.set(total_cash)
                self.cash_label.config(text=f"${total_cash:.2f}")
                self.status_label.config(
                    text=f"Processing complete. Detected: {dict(detected_items)}"
                )
            else:
                messagebox.showerror("Error", "Failed to process video")
                self.status_label.config(text="Error processing video")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_label.config(text="Error occurred")
        finally:
            self.progress.stop()
            self.is_processing = False

    def start_live_detection(self):
        if self.is_processing:
            messagebox.showwarning("Warning", "Already processing. Please stop first.")
            return

        # Start live detection in separate thread
        thread = threading.Thread(target=self.live_detection)
        thread.daemon = True
        thread.start()

    def live_detection(self):
        self.is_processing = True
        self.status_label.config(text="Starting live detection...")

        try:
            total_cash, detected_items = self.detector.process_live_camera()

            self.total_cash.set(total_cash)
            self.cash_label.config(text=f"${total_cash:.2f}")
            self.status_label.config(
                text=f"Live detection stopped. Final: {dict(detected_items)}"
            )

        except Exception as e:
            messagebox.showerror("Error", f"Camera error: {str(e)}")
            self.status_label.config(text="Camera error")
        finally:
            self.is_processing = False

    def stop_processing(self):
        self.is_processing = False
        self.status_label.config(text="Stopping...")
        cv2.destroyAllWindows()  # Close any open CV windows

    def reset_results(self):
        self.total_cash.set(0.0)
        self.cash_label.config(text="$0.00")
        self.status_label.config(text="Results reset")


def main():
    root = tk.Tk()
    CashDetectionGUI(root)  # No need to store in variable
    root.mainloop()


if __name__ == "__main__":
    main()

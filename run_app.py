#!/usr/bin/env python3
"""
Cash Detection Streamlit App Launcher
Run this script to start the Streamlit web interface
"""

import subprocess
import sys
import os


def check_requirements():
    """Check if required packages are installed"""
    required_packages = ["streamlit", "opencv-python", "ultralytics", "numpy", "Pillow"]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install them with:")
        print("   pip install -r requirements_cash.txt")
        return False

    return True


def main():
    print("💰 Cash Detection AI - Streamlit Launcher")
    print("=" * 50)

    # Check if we're in the right directory
    if not os.path.exists("coin.py"):
        print("❌ Error: coin.py not found!")
        print("Please run this script from the project directory.")
        return

    # Check requirements
    if not check_requirements():
        return

    print("✅ All requirements satisfied!")
    print("🚀 Starting Streamlit app...")
    print("\n" + "=" * 50)
    print("📱 The app will open in your default web browser")
    print("🌐 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the app")
    print("=" * 50 + "\n")

    try:
        # Launch Streamlit app
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_cash_app.py",
                "--server.address",
                "localhost",
                "--server.port",
                "8501",
            ]
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down the app...")
    except Exception as e:
        print(f"❌ Error starting app: {e}")


if __name__ == "__main__":
    main()

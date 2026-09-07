"""
Root Application Launcher for the AI-Powered Traffic Violation Dashboard.
Allows running directly with:
    python app.py
"""

import os
import sys
import threading

# Ensure local directories are on python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from web_dashboard.app import app, init_pipeline, video_worker

if __name__ == "__main__":
    print("=" * 70)
    print(" STARTING AI TRAFFIC MONITORING WEB DASHBOARD")
    print(" Server URL: http://127.0.0.1:5000")
    print(" Press Ctrl+C in terminal to stop.")
    print("=" * 70)

    force_cpu = "--cpu" in sys.argv or os.environ.get("FORCE_CPU") == "1"
    if force_cpu:
        print("[*] Running in pure CPU mode (GPU Disabled)")

    init_pipeline(force_cpu=force_cpu)
    t = threading.Thread(target=video_worker, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

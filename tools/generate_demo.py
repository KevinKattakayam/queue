"""Create a plainly synthetic video for software smoke testing only."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from queue_abandonment.config_loader import load_config

def generate(config_path: str = "config/config.yaml") -> None:
    """Write a synthetic scene with a moving geometric person-like figure."""
    config = load_config(config_path)
    destination = Path(config["video_source"]); fps = int(config["synthetic_fps"]); duration = int(config["synthetic_duration_seconds"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
    for frame_index in range(fps * duration):
        image = np.full((480, 640, 3), (85, 85, 85), dtype=np.uint8)
        cv2.rectangle(image, (30, 80), (400, 440), (55, 80, 55), -1)
        cv2.rectangle(image, (410, 140), (620, 400), (55, 55, 95), -1)
        cv2.putText(image, "SYNTHETIC DEMO - NOT EVALUATION DATA", (35, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        sec = frame_index / fps
        if sec < 2: x = int(500 - sec * 100); y = 300
        elif sec < 7: x = 300; y = 350
        elif sec < 13: x = int(300 + (sec - 7) * 35); y = 350
        else: x = 510; y = 300
        cv2.circle(image, (x, y - 115), 24, (40, 160, 230), -1)
        cv2.ellipse(image, (x, y - 65), (31, 50), 0, 0, 360, (40, 160, 230), -1)
        cv2.line(image, (x - 15, y - 30), (x - 25, y), (30, 30, 200), 9)
        cv2.line(image, (x + 15, y - 30), (x + 25, y), (30, 30, 200), 9)
        writer.write(image)
    writer.release()
    print(f"Synthetic smoke-test video written to {destination}; it is not a real-world result.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    generate(parser.parse_args().config)

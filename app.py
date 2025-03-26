import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Add this at the top of app.py
import cv2
import numpy as np
import tensorflow as tf
import tkinter as tk
from PIL import Image, ImageTk

class ExerciseTracker:
    def __init__(self):
        # Model setup
        self.interpreter = tf.lite.Interpreter(model_path="movenet_lightning.tflite")
        self.interpreter.allocate_tensors()
        
        # GUI setup
        self.root = tk.Tk()
        self.root.title("AI Fitness Coach")
        self.canvas = tk.Canvas(self.root, width=800, height=600)
        self.canvas.pack()
        
        # Exercise tracking
        self.reps = 0
        self.stage = "down"
        self.setup_ui()
        
        # Webcam
        self.cap = cv2.VideoCapture(0)
        self.process_frame()
        self.keypoint_buffer = []
    
    def smooth_keypoints(self, current_keypoints):
        self.keypoint_buffer.append(current_keypoints)
        if len(self.keypoint_buffer) > 5:
            self.keypoint_buffer.pop(0)
        return np.mean(self.keypoint_buffer, axis=0)

    def setup_ui(self):
        # Reset button
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)
        
        tk.Button(btn_frame, text="Reset Reps", command=self.reset_reps).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Exit", command=self.root.destroy).pack(side=tk.LEFT)

        # Rep counter display
        self.reps_label = tk.Label(self.root, text="Reps: 0", font=("Arial", 24))
        self.reps_label.pack()

    def reset_reps(self):
        self.reps = 0
        self.reps_label.config(text=f"Reps: {self.reps}")

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        return np.abs(radians * 180.0 / np.pi)

    def process_frame(self):
        ret, frame = self.cap.read()

        
        if ret:
            # Pose estimation
            input_frame = cv2.resize(frame, (192, 192))
            input_frame = np.expand_dims(input_frame, axis=0).astype(np.uint8)
            
            self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_frame)
            self.interpreter.invoke()
            keypoints = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0][0]

            # Keypoint indices for MoveNet
            shoulder = keypoints[5][:2] * [frame.shape[1], frame.shape[0]]
            elbow = keypoints[7][:2] * [frame.shape[1], frame.shape[0]]
            wrist = keypoints[9][:2] * [frame.shape[1], frame.shape[0]]

            # Rep counting logic
            angle = self.calculate_angle(shoulder, elbow, wrist)
            if angle < 90:
                self.stage = "up"
            elif angle > 160 and self.stage == "up":
                self.stage = "down"
                self.reps += 1
                self.reps_label.config(text=f"Reps: {self.reps}")

            # Visualization
            cv2.putText(frame, f"Angle: {int(angle)}°", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show feed
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.imgtk = imgtk
        prev_elbow = self.prev_elbow_position  # Store from previous frame
        current_elbow = elbow  # Current position
        velocity = np.linalg.norm(current_elbow - prev_elbow)
        self.root.after(10, self.process_frame)

        if velocity < 5:  # Pixels/frame threshold
            return

if __name__ == "__main__":
    app = ExerciseTracker()
    app.root.mainloop()
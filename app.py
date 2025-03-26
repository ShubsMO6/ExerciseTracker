import cv2
import mediapipe as mp
import numpy as np
import math

class ExerciseTracker:
    def __init__(self):
        integrate_model_trainer(self)
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize counters
        self.left_counter = 0
        self.right_counter = 0
        self.left_stage = "down"
        self.right_stage = "down"
        
        # Exercise mode tracking
        self.exercise_modes = ["Push-up", "Bicep Curl"]
        self.current_mode_index = 0
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        # Toggle cooldown to prevent rapid switching
        self.toggle_cooldown = 0
        
        # Setup resizable window
        self.setup_resizable_window()
        
        # Start processing
        self.run()

    def setup_resizable_window(self):
        # Create a named window that can be resized
        cv2.namedWindow('Exercise Tracker', cv2.WINDOW_NORMAL)
        
        # Optional: Set an initial window size
        cv2.resizeWindow('Exercise Tracker', 800, 600)

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        # Resize frame to fit window while maintaining aspect ratio
        frame = self.resize_frame(frame)

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = self.pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Toggle mode (on space key press with cooldown)
        keys = cv2.waitKey(10) & 0xFF
        if keys == 32 and self.toggle_cooldown == 0:  # Space key
            self.current_mode_index = (self.current_mode_index + 1) % len(self.exercise_modes)
            # Reset counters when switching mode
            self.left_counter = 0
            self.right_counter = 0
            self.left_stage = "down"
            self.right_stage = "down"
            # Add a cooldown to prevent rapid toggling
            self.toggle_cooldown = 20  # Frames of cooldown
        
        # Manage toggle cooldown
        if self.toggle_cooldown > 0:
            self.toggle_cooldown -= 1

        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates
            left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            left_elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]

            right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
            right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate angles
            left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Exercise-specific counting logic
            current_mode = self.exercise_modes[self.current_mode_index]

            if current_mode == "Push-up":
                # Push-up detection logic
                if left_angle > 160:
                    if self.left_stage == "up":
                        self.left_counter += 1
                    self.left_stage = "down"
                elif left_angle < 30:
                    self.left_stage = "up"

                if right_angle > 160:
                    if self.right_stage == "up":
                        self.right_counter += 1
                    self.right_stage = "down"
                elif right_angle < 30:
                    self.right_stage = "up"

            elif current_mode == "Bicep Curl":
                # Bicep curl detection logic
                if left_angle > 160:
                    if self.left_stage == "up":
                        self.left_counter += 1
                    self.left_stage = "down"
                elif left_angle < 30:
                    self.left_stage = "up"

                if right_angle > 160:
                    if self.right_stage == "up":
                        self.right_counter += 1
                    self.right_stage = "down"
                elif right_angle < 30:
                    self.right_stage = "up"

            # Draw landmarks
            self.mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )

            # Display mode and counter
            cv2.putText(image, f'Mode: {current_mode}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(image, f'Left: {self.left_counter}', 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, f'Right: {self.right_counter}', 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            
            # Instructions for toggling
            cv2.putText(image, 'Press SPACE to toggle mode', 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        except:
            pass

        return image

    def resize_frame(self, frame):
        """
        Resize frame to fit window while maintaining aspect ratio
        """
        # Get window size
        window_width = cv2.getWindowImageRect('Exercise Tracker')[2]
        window_height = cv2.getWindowImageRect('Exercise Tracker')[3]
        
        # If window size is 0, use default
        if window_width == 0 or window_height == 0:
            window_width, window_height = 800, 600
        
        # Calculate resize ratio
        h, w = frame.shape[:2]
        aspect_ratio = w / h
        
        # Resize based on window dimensions
        if w > h:
            new_width = window_width
            new_height = int(new_width / aspect_ratio)
            if new_height > window_height:
                new_height = window_height
                new_width = int(new_height * aspect_ratio)
        else:
            new_height = window_height
            new_width = int(new_height * aspect_ratio)
            if new_width > window_width:
                new_width = window_width
                new_height = int(new_width / aspect_ratio)
        
        # Resize frame
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def run(self):
        while self.cap.isOpened():
            image = self.process_frame()
            if image is None:
                break

            cv2.imshow('Exercise Tracker', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = ExerciseTracker()
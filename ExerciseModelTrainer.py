import cv2
import mediapipe as mp
import numpy as np
import os
import json
import datetime
import pickle

class ExerciseModelTrainer:
    def __init__(self, exercise_type='bicep_curl'):
        # MediaPipe setup
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Exercise and data collection parameters
        self.exercise_type = exercise_type
        self.training_data = {
            'good_reps': [],
            'bad_reps': []
        }
        
        # Setup data storage directories
        self.setup_directories()
        
        # Recording state
        self.is_recording = False
        self.current_rep_frames = []
        self.rep_type = None

    def setup_directories(self):
        """Create necessary directories for storing training data"""
        base_dir = f'exercise_training_data/{self.exercise_type}'
        self.good_reps_dir = os.path.join(base_dir, 'good_reps')
        self.bad_reps_dir = os.path.join(base_dir, 'bad_reps')
        
        os.makedirs(self.good_reps_dir, exist_ok=True)
        os.makedirs(self.bad_reps_dir, exist_ok=True)

    def extract_features(self, landmarks):
        """
        Extract key features from pose landmarks for machine learning
        """
        features = {
            'shoulder_angle': self.calculate_angle(
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
            ),
            'body_orientation': self.calculate_body_orientation(landmarks),
            'elbow_position': self.get_relative_position(
                landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            )
        }
        return features

    def calculate_angle(self, a, b, c):
        """Calculate angle between three landmarks"""
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - \
                  np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        return angle if angle <= 180 else 360 - angle

    def calculate_body_orientation(self, landmarks):
        """Calculate overall body orientation"""
        left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        return abs(left_shoulder.y - right_shoulder.y)

    def get_relative_position(self, point1, point2):
        """Get relative position of two landmarks"""
        return point1.x - point2.x, point1.y - point2.y

    def start_recording(self, rep_type):
        """Begin recording a repetition"""
        if rep_type not in ['good', 'bad']:
            raise ValueError("Rep type must be 'good' or 'bad'")
        
        self.is_recording = True
        self.rep_type = rep_type
        self.current_rep_frames = []

    def stop_recording(self):
        """Stop recording and save repetition data"""
        if not self.is_recording:
            return

        # Process and save rep data
        rep_features = self.process_rep_frames()
        
        if rep_features:
            # Save to corresponding list
            if self.rep_type == 'good':
                self.training_data['good_reps'].append(rep_features)
            else:
                self.training_data['bad_reps'].append(rep_features)
            
            # Optional: Save to file
            self.save_rep_data(rep_features, self.rep_type)

        # Reset recording state
        self.is_recording = False
        self.current_rep_frames = []
        self.rep_type = None

    def process_rep_frames(self):
        """
        Process frames of a single repetition
        Extract key features across frames
        """
        if not self.current_rep_frames:
            return None

        # Collect features from all frames
        frame_features = [self.extract_features(frame) for frame in self.current_rep_frames]
        
        # Aggregate features
        rep_summary = {
            'frames': len(frame_features),
            'avg_angle': np.mean([f['shoulder_angle'] for f in frame_features]),
            'angle_variation': np.std([f['shoulder_angle'] for f in frame_features]),
            'body_orientation_stability': np.std([f['body_orientation'] for f in frame_features])
        }

        return rep_summary

    def save_rep_data(self, rep_data, rep_type):
        """Save individual rep data to file"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{rep_type}_rep_{timestamp}.pkl"
        
        filepath = os.path.join(
            self.good_reps_dir if rep_type == 'good' else self.bad_reps_dir, 
            filename
        )
        
        with open(filepath, 'wb') as f:
            pickle.dump(rep_data, f)

    def train_model(self):
        """
        Train a simple machine learning model to classify good vs bad reps
        """
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report

        # Prepare training data
        X = []
        y = []

        for rep in self.training_data['good_reps']:
            X.append([rep['avg_angle'], rep['angle_variation'], rep['body_orientation_stability']])
            y.append(1)  # Good rep

        for rep in self.training_data['bad_reps']:
            X.append([rep['avg_angle'], rep['angle_variation'], rep['body_orientation_stability']])
            y.append(0)  # Bad rep

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train_scaled, y_train)

        # Evaluate
        y_pred = model.predict(X_test_scaled)
        print(classification_report(y_test, y_pred))

        # Save model
        import joblib
        joblib.dump(model, f'{self.exercise_type}_rep_classifier.joblib')
        joblib.dump(scaler, f'{self.exercise_type}_scaler.joblib')

    def add_frame_to_current_rep(self, landmarks):
        """Add a frame to the current rep being recorded"""
        if self.is_recording:
            self.current_rep_frames.append(landmarks)

# Integration with main ExerciseTracker
def integrate_model_trainer(exercise_tracker):
    """
    Add model training capabilities to the existing ExerciseTracker
    """
    # Add these to ExerciseTracker's __init__
    exercise_tracker.model_trainer = ExerciseModelTrainer()
    
    # Modify process_frame method to include:
    def extended_process_frame(self):
        # Original process_frame logic
        frame = original_process_frame(self)
        
        # Additional model training controls
        keys = cv2.waitKey(10) & 0xFF
        
        # 'G' key to start recording good rep
        if keys == ord('g'):
            self.model_trainer.start_recording('good')
            print("Recording GOOD rep...")
        
        # 'B' key to start recording bad rep
        elif keys == ord('b'):
            self.model_trainer.start_recording('bad')
            print("Recording BAD rep...")
        
        # 'S' key to stop recording
        elif keys == ord('s'):
            self.model_trainer.stop_recording()
            print("Stopped recording rep")
        
        # 'T' key to train model
        elif keys == ord('t'):
            self.model_trainer.train_model()
            print("Training model...")
        
        return frame

    # Replace process_frame method
    exercise_tracker.process_frame = extended_process_frame

# Usage instructions
"""
In your main ExerciseTracker class:

1. After __init__(), call:
   integrate_model_trainer(self)

2. During exercise tracking:
   - Press 'G' to start recording a GOOD rep
   - Press 'B' to start recording a BAD rep
   - Press 'S' to stop recording
   - Press 'T' to train the model

The model will create:
- Training data in exercise_training_data/
- Trained model: bicep_curl_rep_classifier.joblib
- Scaler: bicep_curl_scaler.joblib
"""
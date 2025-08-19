import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from collections import deque
import json
import time

class JujutsuHandSignTrainer:
    def __init__(self, images_folder_path):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        
        # High accuracy settings
        self.hands_static = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        self.hands_live = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.72,
            min_tracking_confidence=0.6,
            model_complexity=1
        )
        
        self.images_folder = images_folder_path
        self.hand_sign_library = {}  # Store all processed hand signs
        self.similarity_buffer = deque(maxlen=15)  # Smoothing buffer
        
        # Training state
        self.current_sign_index = 0
        self.sign_list = []
        self.detection_threshold = 0.70  # Lower threshold for easier progression
        self.stable_detections = 0
        self.required_stable_detections = 5 
        
        # Feature weights for accuracy
        self.weights = {
            'landmarks': 0.4,
            'finger_positions': 0.3,
            'hand_geometry': 0.2,
            'finger_angles': 0.1
        }
        
        print(f"🎯 Initializing Jujutsu Hand Sign Trainer from: {images_folder_path}")
    
    def process_image_library(self):
        """Process all images in the folder to create ordered training sequence"""
        print("📚 Processing jujutsu hand sign library...")
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(self.images_folder, extension)))
            image_files.extend(glob.glob(os.path.join(self.images_folder, extension.upper())))
        
        if not image_files:
            print(f"❌ No images found in {self.images_folder}")
            return False
        
        # Sort files for consistent order
        image_files.sort()
        print(f"🔍 Found {len(image_files)} images to process")
        
        processed_count = 0
        for i, image_path in enumerate(image_files):
            print(f"📋 Processing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            if self.process_single_image(image_path):
                processed_count += 1
                self.sign_list.append(os.path.splitext(os.path.basename(image_path))[0])
        
        print(f"✅ Training sequence created! Processed {processed_count}/{len(image_files)} images")
        print(f"📖 Training contains {len(self.hand_sign_library)} hand signs")
        
        # Display training sequence
        self.display_training_sequence()
        
        return len(self.hand_sign_library) > 0
    
    def process_single_image(self, image_path):
        """Process a single image and extract hand sign features"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"⚠️  Could not read image: {image_path}")
                return False
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.hands_static.process(image_rgb)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                # Get image name without extension for the sign name
                sign_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Process each hand in the image
                hands_data = []
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_data = {
                        'hand_type': handedness.classification[0].label,
                        'confidence': handedness.classification[0].score,
                        'landmarks': self.extract_normalized_landmarks(hand_landmarks),
                        'features': self.extract_comprehensive_features(hand_landmarks),
                        'finger_positions': self.extract_finger_positions(hand_landmarks),
                        'geometry': self.extract_hand_geometry(hand_landmarks),
                        'angles': self.extract_finger_angles(hand_landmarks)
                    }
                    hands_data.append(hand_data)
                
                # Store in library with processed display image
                # Resize image for display during processing
                height = 200
                aspect_ratio = image.shape[1] / image.shape[0]
                width = int(height * aspect_ratio)
                display_image = cv2.resize(image, (width, height))
                
                self.hand_sign_library[sign_name] = {
                    'image_path': image_path,
                    'hands': hands_data,
                    'processed_time': time.time(),
                    'completed': False,  # Track if user has successfully performed this sign
                    'display_image': display_image  # Store resized image for display
                }
                
                print(f"   ✅ Extracted {len(hands_data)} hand(s) from {sign_name}")
                return True
            
            else:
                print(f"   ❌ No hands detected in {os.path.basename(image_path)}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {str(e)}")
            return False
    
    def extract_normalized_landmarks(self, landmarks):
        """Extract and normalize landmark positions"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Normalize using wrist as origin
        wrist = points[0]
        normalized = points - wrist
        
        # Scale by hand span (wrist to middle finger tip)
        hand_span = np.linalg.norm(normalized[12][:2])  # Only x,y for span
        if hand_span > 0:
            normalized = normalized / hand_span
        
        return normalized.flatten()  # Flatten to 1D array
    
    def extract_comprehensive_features(self, landmarks):
        """Extract comprehensive hand features"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        features = []
        
        # Finger tip distances from wrist
        finger_tips = [4, 8, 12, 16, 20]
        wrist = points[0]
        
        for tip in finger_tips:
            distance = np.linalg.norm(points[tip] - wrist)
            features.append(distance)
        
        # Inter-finger distances
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                distance = np.linalg.norm(points[finger_tips[i]] - points[finger_tips[j]])
                features.append(distance)
        
        # Hand span and palm size
        hand_span = np.linalg.norm(points[20] - points[4])  # Pinky to thumb
        palm_size = np.linalg.norm(points[9] - points[0])   # Middle MCP to wrist
        features.extend([hand_span, palm_size])
        
        return np.array(features)
    
    def extract_finger_positions(self, landmarks):
        """Extract specific finger position features"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        finger_data = {}
        
        # Define finger landmark indices
        fingers = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        wrist = points[0]
        
        for finger_name, indices in fingers.items():
            # Calculate finger extension (tip distance from base)
            base_idx = indices[0]
            tip_idx = indices[3]
            
            extension = np.linalg.norm(points[tip_idx] - points[base_idx])
            tip_to_wrist = np.linalg.norm(points[tip_idx] - wrist)
            
            finger_data[finger_name] = {
                'extension': extension,
                'tip_distance': tip_to_wrist,
                'relative_position': (points[tip_idx] - wrist).tolist()
            }
        
        return finger_data
    
    def extract_hand_geometry(self, landmarks):
        """Extract hand geometric features"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Hand orientation
        wrist_to_middle = points[12] - points[0]
        orientation = np.arctan2(wrist_to_middle[1], wrist_to_middle[0])
        
        # Palm area (approximate using key points)
        palm_points = [0, 5, 9, 13, 17]  # Wrist and MCP joints
        palm_area = self.calculate_polygon_area(points[palm_points][:, :2])
        
        # Hand compactness
        finger_tips = [4, 8, 12, 16, 20]
        centroid = np.mean(points[finger_tips], axis=0)
        compactness = np.mean([np.linalg.norm(points[tip] - centroid) for tip in finger_tips])
        
        return {
            'orientation': orientation,
            'palm_area': palm_area,
            'compactness': compactness
        }
    
    def extract_finger_angles(self, landmarks):
        """Extract finger bend angles"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        angles = {}
        
        # Define finger segments for angle calculation
        finger_segments = {
            'thumb': [(1, 2, 3), (2, 3, 4)],
            'index': [(5, 6, 7), (6, 7, 8)],
            'middle': [(9, 10, 11), (10, 11, 12)],
            'ring': [(13, 14, 15), (14, 15, 16)],
            'pinky': [(17, 18, 19), (18, 19, 20)]
        }
        
        for finger_name, segments in finger_segments.items():
            finger_angles = []
            for p1_idx, p2_idx, p3_idx in segments:
                angle = self.calculate_angle(points[p1_idx], points[p2_idx], points[p3_idx])
                finger_angles.append(angle)
            angles[finger_name] = finger_angles
        
        return angles
    
    def calculate_angle(self, p1, p2, p3):
        """Calculate angle at p2 formed by p1-p2-p3"""
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return angle
    
    def calculate_polygon_area(self, points_2d):
        """Calculate polygon area using shoelace formula"""
        n = len(points_2d)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points_2d[i][0] * points_2d[j][1]
            area -= points_2d[j][0] * points_2d[i][1]
        return abs(area) / 2.0
    
    def display_training_sequence(self):
        """Display the training sequence"""
        print("\n🎓 JUJUTSU HAND SIGN TRAINING SEQUENCE")
        print("=" * 50)
        
        for i, sign_name in enumerate(self.sign_list, 1):
            status = "✅ Completed" if self.hand_sign_library[sign_name].get('completed', False) else "⏳ Pending"
            current_marker = "👉" if i-1 == self.current_sign_index else "  "
            print(f"{current_marker} {i:2d}. {sign_name} - {status}") 
        
        print("=" * 50)
        print(f"📊 Progress: {sum(1 for s in self.sign_list if self.hand_sign_library[s].get('completed', False))}/{len(self.sign_list)} signs completed")
    
    def get_current_sign(self):
        """Get the current sign to practice"""
        if self.current_sign_index < len(self.sign_list):
            return self.sign_list[self.current_sign_index]
        return None
    
    def check_sign_match(self, live_landmarks, live_handedness):
        """Check if live hand matches current target sign"""
        current_sign = self.get_current_sign()
        if not current_sign or current_sign not in self.hand_sign_library:
            return 0, None
        
        live_hand_type = live_handedness.classification[0].label
        
        # Extract features from live hand
        live_features = {
            'landmarks': self.extract_normalized_landmarks(live_landmarks),
            'features': self.extract_comprehensive_features(live_landmarks),
            'finger_positions': self.extract_finger_positions(live_landmarks),
            'geometry': self.extract_hand_geometry(live_landmarks),
            'angles': self.extract_finger_angles(live_landmarks)
        }
        
        best_similarity = 0
        best_hand_data = None
        
        # Compare with target sign
        sign_data = self.hand_sign_library[current_sign]
        for hand_data in sign_data['hands']:
            if hand_data['hand_type'] == live_hand_type:
                similarity = self.calculate_comprehensive_similarity(live_features, hand_data)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_hand_data = hand_data
        
        return best_similarity, best_hand_data
    
    def calculate_comprehensive_similarity(self, live_features, library_hand):
        """Calculate comprehensive similarity between live hand and library hand"""
        similarities = {}
        
        # 1. Landmark similarity
        live_landmarks = live_features['landmarks']
        lib_landmarks = library_hand['landmarks']
        landmark_distance = np.linalg.norm(live_landmarks - lib_landmarks)
        similarities['landmarks'] = max(0, 1 - landmark_distance / 3.0)
        
        # 2. Feature similarity
        live_feat = live_features['features']
        lib_feat = library_hand['features']
        feature_distance = np.linalg.norm(live_feat - lib_feat)
        max_feature_distance = np.linalg.norm(live_feat) + np.linalg.norm(lib_feat) + 1e-8
        similarities['features'] = max(0, 1 - feature_distance / max_feature_distance)
        
        # 3. Finger position similarity
        finger_sim = self.compare_finger_positions(
            live_features['finger_positions'], 
            library_hand['finger_positions']
        )
        similarities['finger_positions'] = finger_sim
        
        # 4. Geometry similarity
        geometry_sim = self.compare_geometry(
            live_features['geometry'], 
            library_hand['geometry']
        )
        similarities['geometry'] = geometry_sim
        
        # Weighted combination
        total_similarity = (
            self.weights['landmarks'] * similarities['landmarks'] +
            self.weights['finger_positions'] * similarities['finger_positions'] +
            self.weights['hand_geometry'] * similarities['geometry'] +
            self.weights['finger_angles'] * similarities.get('features', 0)
        )
        
        return total_similarity
    
    def compare_finger_positions(self, live_fingers, lib_fingers):
        """Compare finger position features"""
        similarities = []
        
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger_name in live_fingers and finger_name in lib_fingers:
                live_finger = live_fingers[finger_name]
                lib_finger = lib_fingers[finger_name]
                
                # Compare extension
                ext_diff = abs(live_finger['extension'] - lib_finger['extension'])
                ext_sim = max(0, 1 - ext_diff)
                
                # Compare tip distance
                tip_diff = abs(live_finger['tip_distance'] - lib_finger['tip_distance'])
                tip_sim = max(0, 1 - tip_diff)
                
                finger_similarity = (ext_sim + tip_sim) / 2
                similarities.append(finger_similarity)
        
        return np.mean(similarities) if similarities else 0
    
    def compare_geometry(self, live_geom, lib_geom):
        """Compare hand geometry features"""
        # Orientation similarity
        angle_diff = abs(live_geom['orientation'] - lib_geom['orientation'])
        angle_diff = min(angle_diff, 2*np.pi - angle_diff)
        orientation_sim = 1 - angle_diff / np.pi
        
        # Area similarity
        area_ratio = min(live_geom['palm_area'], lib_geom['palm_area']) / \
                    (max(live_geom['palm_area'], lib_geom['palm_area']) + 1e-8)
        
        # Compactness similarity
        comp_diff = abs(live_geom['compactness'] - lib_geom['compactness'])
        compactness_sim = max(0, 1 - comp_diff)
        
        return (orientation_sim + area_ratio + compactness_sim) / 3
    
    def advance_to_next_sign(self):
        """Advance to the next sign in sequence"""
        current_sign = self.get_current_sign()
        if current_sign:
            self.hand_sign_library[current_sign]['completed'] = True
            print(f"🎉 COMPLETED: {current_sign}")
        
        self.current_sign_index += 1
        self.stable_detections = 0
        self.similarity_buffer.clear()
        
        if self.current_sign_index >= len(self.sign_list):
            print("🏆 CONGRATULATIONS! All hand signs completed!")
            return True
        else:
            next_sign = self.get_current_sign()
            print(f"🎯 Next sign: {next_sign}")
            return False
    
    def start_training_session(self):
        """Start the guided training session"""
        if not self.hand_sign_library:
            print("❌ No hand signs in library! Process images first.")
            return
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Cannot open camera!")
            return
        
        print(f"🎯 Starting guided training with {len(self.hand_sign_library)} hand signs...")
        print("📋 NEW: Auto-advance when accuracy reaches 65% or higher!")
        print("📋 Controls: 'q'=quit, 'n'=next sign (skip), 'p'=previous sign, 'r'=reset progress")
        
        # Load reference images for display
        reference_images = {}
        for sign_name, sign_data in self.hand_sign_library.items():
            img = cv2.imread(sign_data['image_path'])
            if img is not None:
                # Resize for display
                height = 200
                aspect_ratio = img.shape[1] / img.shape[0]
                width = int(height * aspect_ratio)
                reference_images[sign_name] = cv2.resize(img, (width, height))
        
        training_complete = False
        
        while not training_complete:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hands
            results = self.hands_live.process(frame_rgb)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            
            current_sign = self.get_current_sign()
            
            # Display current target sign reference image
            if current_sign and current_sign in self.hand_sign_library:
                # Use pre-loaded display image
                ref_img = self.hand_sign_library[current_sign]['display_image']
                # Place reference image on the right side
                start_x = frame_bgr.shape[1] - ref_img.shape[1] - 10
                start_y = 10
                end_x = start_x + ref_img.shape[1]
                end_y = start_y + ref_img.shape[0]
                
                if end_y < frame_bgr.shape[0] and end_x < frame_bgr.shape[1]:
                    # Add semi-transparent background
                    overlay = frame_bgr.copy()
                    cv2.rectangle(overlay, (start_x-5, start_y-25), (end_x+5, end_y+5), (0, 0, 0), -1)
                    frame_bgr = cv2.addWeighted(frame_bgr, 0.7, overlay, 0.3, 0)
                    
                    # Place reference image
                    frame_bgr[start_y:end_y, start_x:end_x] = ref_img
                    
                    # Add border and label
                    cv2.rectangle(frame_bgr, (start_x-2, start_y-2), (end_x+2, end_y+2), (0, 255, 255), 2)
                    cv2.putText(frame_bgr, f"TARGET: {current_sign}", 
                              (start_x, start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw landmarks
                    self.mp_drawing.draw_landmarks(
                        frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                    )
                    
                    if current_sign:
                        # Check match with current target sign
                        similarity, _ = self.check_sign_match(hand_landmarks, handedness)
                        
                        # Store similarity for smoothing
                        self.similarity_buffer.append(similarity)
                        smoothed_similarity = np.mean(list(self.similarity_buffer))
                        
                        # Check if detection is stable and accurate
                        if smoothed_similarity >= self.detection_threshold:
                            self.stable_detections += 1
                        else:
                            self.stable_detections = max(0, self.stable_detections - 2)
                        
                        # Color and status based on similarity
                        if smoothed_similarity >= 0.65:
                            color = (0, 255, 0)  # Green
                            status = "GREAT! ✅ ADVANCING..."
                        elif smoothed_similarity >= 0.55:
                            color = (0, 255, 255)  # Yellow
                            status = "ALMOST THERE! ⭐"
                        elif smoothed_similarity >= 0.45:
                            color = (0, 165, 255)  # Orange
                            status = "GETTING CLOSER! 👍"
                        elif smoothed_similarity >= 0.35:
                            color = (0, 100, 255)  # Light Blue
                            status = "KEEP TRYING 📝"
                        else:
                            color = (0, 0, 255)  # Red
                            status = "TRY AGAIN ❌"
                        
                        # Display current progress
                        y_offset = 30
                        cv2.putText(frame_bgr, f'Training: {current_sign}', 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        y_offset += 35
                        
                        cv2.putText(frame_bgr, f'Accuracy: {smoothed_similarity*100:.1f}%', 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 30
                        
                        cv2.putText(frame_bgr, status, 
                                  (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        y_offset += 35
                        
                        # Progress bar
                        progress = min(self.stable_detections / self.required_stable_detections, 1.0)
                        bar_width = int(300 * progress)
                        cv2.rectangle(frame_bgr, (10, y_offset), (310, y_offset + 20), (100, 100, 100), 2)
                        if bar_width > 0:
                            cv2.rectangle(frame_bgr, (10, y_offset), (10 + bar_width, y_offset + 20), color, -1)
                        
                        cv2.putText(frame_bgr, f'Hold steady: {self.stable_detections}/{self.required_stable_detections}', 
                                  (10, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        
                        # Check if ready to advance (65% threshold)
                        if self.stable_detections >= self.required_stable_detections:
                            cv2.putText(frame_bgr, '🎉 SIGN COMPLETED! Moving to next...', 
                                      (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                            
                            # Auto-advance after a short delay
                            time.sleep(0.5)  # Shorter delay for faster progression
                            training_complete = self.advance_to_next_sign()
                            if not training_complete:
                                time.sleep(0.5)  # Brief pause before next sign
            
            else:
                cv2.putText(frame_bgr, f'Show hand sign: {current_sign if current_sign else "COMPLETE"}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                if current_sign:
                    cv2.putText(frame_bgr, 'Position your hand like the reference image →', 
                              (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Display overall progress
            completed_count = sum(1 for s in self.sign_list if self.hand_sign_library[s].get('completed', False))
            progress_text = f'Overall Progress: {completed_count}/{len(self.sign_list)} signs'
            cv2.putText(frame_bgr, progress_text, 
                      (10, frame_bgr.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Instructions
            cv2.putText(frame_bgr, 'Q:Quit | N:Skip | P:Previous | R:Reset | Auto-advance at 65%', 
                      (10, frame_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imshow('Jujutsu Hand Sign Trainer', frame_bgr)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('n'):
                # Skip to next sign
                training_complete = self.advance_to_next_sign()
            elif key == ord('p'):
                # Go to previous sign
                if self.current_sign_index > 0:
                    self.current_sign_index -= 1
                    self.stable_detections = 0
                    self.similarity_buffer.clear()
                    current_sign = self.get_current_sign()
                    if current_sign:
                        self.hand_sign_library[current_sign]['completed'] = False
                        print(f"🔙 Back to: {current_sign}")
            elif key == ord('r'):
                # Reset all progress
                self.current_sign_index = 0
                self.stable_detections = 0
                self.similarity_buffer.clear()
                for sign_name in self.sign_list:
                    self.hand_sign_library[sign_name]['completed'] = False
                print("🔄 Training progress reset!")
        
        if training_complete:
            # Show completion screen
            completion_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(completion_frame, '🏆 TRAINING COMPLETE! 🏆', 
                      (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(completion_frame, f'You mastered all {len(self.sign_list)} jujutsu signs!', 
                      (80, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(completion_frame, 'Press any key to exit', 
                      (200, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow('Jujutsu Hand Sign Trainer', completion_frame)
            cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()
        self.hands_live.close()
        self.hands_static.close()

def main():
    # Set your images folder path here
    images_folder = r"handsign"  
        
    # Create the trainer
    trainer = JujutsuHandSignTrainer(images_folder)
    
    # Process all images in the folder
    if trainer.process_image_library():        
        # Start guided training session
        trainer.start_training_session()
    else:
        print("❌ Failed to process image library!")
        print("📁 Make sure your folder contains jujutsu hand sign images!")
        print("📋 Supported formats: JPG, JPEG, PNG, BMP, TIFF")

if __name__ == "__main__":
    main()
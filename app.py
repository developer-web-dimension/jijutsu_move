import cv2
import mediapipe as mp
import numpy as np
import os
import glob
from collections import deque
import json
import time
import base64
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import threading

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
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            model_complexity=1
        )
        
        self.images_folder = images_folder_path
        self.hand_sign_library = {}  # Store all processed hand signs
        self.similarity_buffer = deque(maxlen=15)  # Smoothing buffer
        
        # Training state
        self.current_sign_index = 0
        self.sign_list = []
        self.detection_threshold = 0.65  # Lower threshold for easier progression
        self.stable_detections = 0
        self.required_stable_detections = 5 

        self.training_active = False
        self.current_accuracy = 0
        self.current_status = "Position your hand"

        # ADD THESE NEW LINES:
        self.sign_max_accuracies = {}  # Track max accuracy for each sign
        self.current_sign_start_time = None  # Track when current sign started
        
        # Feature weights for accuracy
        self.weights = {
            'landmarks': 0.2,
            'finger_positions': 0.25,
            'hand_geometry': 0.3,
            'finger_angles': 0.025,
        }
        
        # Flask-specific variables
        self.training_active = False
        self.current_accuracy = 0
        self.current_status = "Position your hand"
        
    
    def process_image_library(self):
        """Process all images in the folder to create ordered training sequence"""
        
        # Supported image format
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
            sign_name = os.path.splitext(os.path.basename(image_path))[0]
            
            if self.process_single_image(image_path):
                processed_count += 1
                # Only add to sign_list if not already present (prevents duplicates)
                if sign_name not in self.sign_list:
                    self.sign_list.append(sign_name)
        
        print(f"✅ Successfully processed {processed_count} images")
        print(f"📋 Unique signs found: {len(self.sign_list)}")
        print(f"🎯 Sign sequence: {self.sign_list}")
        
        return len(self.hand_sign_library) > 0
    
    def process_single_image(self, image_path):
        """Process a single image and extract hand sign features"""
        try:
            # Read image ONCE
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
                
                # Resize image for display ONCE and store both original and display versions
                height = 200
                aspect_ratio = image.shape[1] / image.shape[0]
                width = int(height * aspect_ratio)
                display_image_small = cv2.resize(image, (width, height))
                
                # Also create web display size (300px height for web)
                web_height = 300
                web_width = int(web_height * aspect_ratio)
                display_image_web = cv2.resize(image, (web_width, web_height))
                
                # Encode web image to base64 immediately to avoid repeated file reads
                _, buffer = cv2.imencode('.jpg', display_image_web)
                web_image_base64 = base64.b64encode(buffer).decode('utf-8')
                
                self.hand_sign_library[sign_name] = {
                    'image_path': image_path,
                    'hands': hands_data,
                    'processed_time': time.time(),
                    'completed': False,
                    'display_image': display_image_small,  # Small version for processing display
                    'web_image_base64': web_image_base64   # Pre-encoded base64 for web
                }
                
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
        """Advance to the next sign in sequence - auto-advance version"""
        current_sign = self.get_current_sign()
        if current_sign:
            self.hand_sign_library[current_sign]['completed'] = True
            
            # Store the max accuracy achieved for this sign
            if current_sign in self.sign_max_accuracies:
                max_accuracy = self.sign_max_accuracies[current_sign]
                print(f"🎉 COMPLETED: {current_sign} (Best: {max_accuracy:.1f}%)")
            else:
                print(f"🎉 COMPLETED: {current_sign}")
        
        self.current_sign_index += 1
        self.stable_detections = 0
        self.similarity_buffer.clear()
        
        # Reset tracking for next sign
        if self.current_sign_index < len(self.sign_list):
            next_sign = self.get_current_sign()
            if next_sign:
                # Initialize max accuracy tracking for new sign
                if next_sign not in self.sign_max_accuracies:
                    self.sign_max_accuracies[next_sign] = 0.0
                self.current_sign_start_time = time.time()
        
        if self.current_sign_index >= len(self.sign_list):
            print("🏆 CONGRATULATIONS! All hand signs completed!")
            return True
        else:
            next_sign = self.get_current_sign()
            print(f"🎯 Next sign: {next_sign}")
            return False


    # REPLACE the existing process_frame_from_websocket method with this simplified version:
    def process_frame_from_websocket(self, frame_data):
        """Process frame sent from WebSocket - simplified for auto-advance"""
        try:
            # Decode base64 image
            import io
            from PIL import Image
            
            # Remove data URL prefix if present
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            # Decode base64
            img_data = base64.b64decode(frame_data)
            img = Image.open(io.BytesIO(img_data))
            
            # Convert PIL to OpenCV format
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            
            # Process with MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands_live.process(frame_rgb)
            
            current_sign = self.get_current_sign()
            hand_detected = False
            
            # Count detected hands
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            
            if results.multi_hand_landmarks and results.multi_handedness:
                hand_detected = True
                
                if num_hands < 2:
                    self.current_status = f"Show BOTH hands ({num_hands}/2 detected)"
                    self.current_accuracy = 15  # Low accuracy for single hand
                else:
                    # Process hands normally when we have both hands
                    best_similarity = 0
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        if current_sign:
                            similarity, _ = self.check_sign_match(hand_landmarks, handedness)
                            best_similarity = max(best_similarity, similarity)
                    
                    if current_sign:
                        # Store similarity for display (no threshold checking)
                        self.similarity_buffer.append(best_similarity)
                        smoothed_similarity = np.mean(list(self.similarity_buffer))
                        self.current_accuracy = smoothed_similarity * 100
                        
                        # UPDATE MAX ACCURACY TRACKING FOR CURRENT SIGN
                        if current_sign not in self.sign_max_accuracies:
                            self.sign_max_accuracies[current_sign] = 0.0
                        
                        if self.current_accuracy > self.sign_max_accuracies[current_sign]:
                            self.sign_max_accuracies[current_sign] = self.current_accuracy
                        
                        # Status based on similarity (for feedback only)
                        if smoothed_similarity >= 0.65:
                            self.current_status = "EXCELLENT!"
                        elif smoothed_similarity >= 0.55:
                            self.current_status = "VERY GOOD!"
                        elif smoothed_similarity >= 0.45:
                            self.current_status = "GOOD!"
                        elif smoothed_similarity >= 0.35:
                            self.current_status = "KEEP PRACTICING"
                        else:
                            self.current_status = "TRY THE POSE"
            
            if not hand_detected:
                self.current_status = f'Show BOTH hands for sign: {current_sign if current_sign else "COMPLETE"}'
                self.current_accuracy = 0
            
            return {
                'training_complete': False,
                'accuracy': self.current_accuracy,
                'status': self.current_status,
                'hand_detected': hand_detected,
                'sign_completed': False
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'training_complete': False,
                'accuracy': 0,
                'status': 'Error processing frame',
                'hand_detected': False,
                'sign_completed': False
            }
    
    def get_current_sign_image_base64(self):
        """Get current sign reference image as base64 - using pre-stored version"""
        current_sign = self.get_current_sign()
        if current_sign and current_sign in self.hand_sign_library:
            # Return pre-encoded base64 image instead of reading from disk again
            return self.hand_sign_library[current_sign].get('web_image_base64')
        return None

# Initialize Flask app and SocketIO
app = Flask(__name__)
# socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',  # or 'eventlet' for production
    transport=['websocket', 'polling'],  # Allow fallback to polling
    ping_timeout=60,
    ping_interval=25,
    logger=False,
    engineio_logger=False
)

# Initialize trainer (you can change the path here)
trainer = JujutsuHandSignTrainer("handsign")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on_error_default
def default_error_handler(e):
    print(f"SocketIO Error: {e}")
    return {'error': str(e)}

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected - Resetting trainer state')
    
    # Reset trainer state on new connection (page refresh)
    trainer.current_sign_index = 0
    trainer.stable_detections = 0
    trainer.similarity_buffer.clear()
    trainer.training_active = False
    trainer.current_accuracy = 0
    trainer.current_status = "Position your hand"
    
    # Clear max accuracy tracking
    trainer.sign_max_accuracies.clear()
    trainer.current_sign_start_time = None
    
    # Reset all signs to not completed
    for sign_name in trainer.sign_list:
        if sign_name in trainer.hand_sign_library:
            trainer.hand_sign_library[sign_name]['completed'] = False
    
    emit('status', {'message': 'Connected to server - Training reset'})
    print('Trainer state reset for new client')


    # Add this new WebSocket event handler
@socketio.on('auto_advance_sign')
def handle_auto_advance_sign():
    """Auto-advance to next sign via WebSocket"""
    is_complete = trainer.advance_to_next_sign()
    next_sign = trainer.get_current_sign()
    
    emit('sign_auto_advanced', {
        'success': True, 
        'training_complete': is_complete,
        'next_sign': next_sign,
        'message': 'Auto-advanced to next sign!'
    })

# Add this new WebSocket event handler after the existing ones
@socketio.on('get_training_results')
def handle_get_training_results():
    """Get training results with max accuracies for each sign"""
    results = []
    total_accuracy = 0
    lowest_accuracy = float('inf')
    lowest_sign = None
    
    for sign_name in trainer.sign_list:
        if sign_name in trainer.sign_max_accuracies:
            accuracy = trainer.sign_max_accuracies[sign_name]
            results.append({
                'sign_name': sign_name,
                'max_accuracy': accuracy
            })
            total_accuracy += accuracy
            
            if accuracy < lowest_accuracy:
                lowest_accuracy = accuracy
                lowest_sign = sign_name
    
    # Calculate average
    average_accuracy = total_accuracy / len(results) if results else 0
    
    emit('training_results_update', {
        'results': results,
        'average_accuracy': average_accuracy,
        'lowest_sign': {
            'name': lowest_sign,
            'accuracy': lowest_accuracy if lowest_sign else 0
        },
        'total_signs': len(trainer.sign_list)
    })


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_training')
def handle_start_training():
    """Initialize training via WebSocket"""
    if not trainer.hand_sign_library:
        if not trainer.process_image_library():
            emit('training_response', {'success': False, 'message': 'Failed to process image library!'})
            return
    
    trainer.training_active = True
    emit('training_response', {'success': True, 'message': 'Training started!'})

@socketio.on('stop_training')
def handle_stop_training():
    """Stop training via WebSocket"""
    trainer.training_active = False
    emit('training_response', {'success': True, 'message': 'Training stopped!'})


@socketio.on('process_frame')
def handle_process_frame(data):
    """Process frame via WebSocket"""
    try:
        frame_data = data.get('frame')
        
        if frame_data and trainer.training_active:
            result = trainer.process_frame_from_websocket(frame_data)
            emit('frame_result', {
                'success': True,
                'training_complete': result.get('training_complete', False),
                'accuracy': result.get('accuracy', 0),
                'status': result.get('status', ''),
                'hand_detected': result.get('hand_detected', False),
                'sign_completed': result.get('sign_completed', False)
            })
        else:
            emit('frame_result', {'success': False, 'message': 'No frame data or training not active'})
    except Exception as e:
        emit('frame_result', {'success': False, 'message': str(e)})

@socketio.on('training_complete')
def handle_training_complete():
    """Handle training completion"""
    emit('training_complete_response', {
        'success': True,
        'message': 'Training completed successfully!',
        'total_signs': len(trainer.sign_list),
        'completion_time': time.time()
    })

@socketio.on('get_status')
def handle_get_status():
    """Get current training status via WebSocket"""
    current_sign = trainer.get_current_sign()
    completed_count = sum(1 for s in trainer.sign_list if trainer.hand_sign_library.get(s, {}).get('completed', False))
    
    emit('status_update', {
        'current_sign': current_sign,
        'accuracy': trainer.current_accuracy,
        'status': trainer.current_status,
        'completed_count': completed_count,
        'total_signs': len(trainer.sign_list),
        'stable_detections': trainer.stable_detections,
        'required_detections': trainer.required_stable_detections,
        'training_complete': trainer.current_sign_index >= len(trainer.sign_list)
    })

@socketio.on('get_current_sign_image')
def handle_get_current_sign_image():
    """Get current sign reference image via WebSocket"""
    img_base64 = trainer.get_current_sign_image_base64()
    current_sign = trainer.get_current_sign()
    
    emit('sign_image_update', {
        'image': img_base64,
        'sign_name': current_sign
    })

@socketio.on('next_sign')
def handle_next_sign():
    """Skip to next sign via WebSocket"""
    trainer.advance_to_next_sign()
    emit('sign_changed', {'success': True, 'message': 'Moved to next sign!'})

@socketio.on('previous_sign')
def handle_previous_sign():
    """Go to previous sign via WebSocket"""
    if trainer.current_sign_index > 0:
        trainer.current_sign_index -= 1
        trainer.stable_detections = 0
        trainer.similarity_buffer.clear()
        current_sign = trainer.get_current_sign()
        if current_sign:
            trainer.hand_sign_library[current_sign]['completed'] = False
    emit('sign_changed', {'success': True, 'message': 'Moved to previous sign!'})

@socketio.on('reset_progress')
def handle_reset_progress():
    """Reset all training progress via WebSocket"""
    trainer.current_sign_index = 0
    trainer.stable_detections = 0
    trainer.similarity_buffer.clear()
    
    # RESET MAX ACCURACY TRACKING
    trainer.sign_max_accuracies.clear()
    trainer.current_sign_start_time = None
    
    for sign_name in trainer.sign_list:
        if sign_name in trainer.hand_sign_library:
            trainer.hand_sign_library[sign_name]['completed'] = False
    emit('progress_reset', {'success': True, 'message': 'Training progress reset!'})

# Create templates directory and save HTML template
import os
if not os.path.exists('templates'):
    os.makedirs('templates')


html_template = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Jujutsu Trainer">
    <meta name="format-detection" content="telephone=no">
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Jujutsu Hand Sign Trainer</title>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            min-height: 100vh;
        }
        
        /* Top Progress Bar */
        .top-progress-bar {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 6px;
            background: rgba(255,255,255,0.2);
            z-index: 1000;
            display: none;
        }
        
        .top-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            transition: width 0.3s ease;
            width: 0%;
        }
        
        /* Top Left Sign Preview */
        .top-sign-preview {
            position: fixed;
            top: 20px;
            left: 20px;
            z-index: 999;
            background: rgba(0,0,0,0.7);
            border-radius: 10px;
            padding: 10px;
            display: none;
            max-width: 120px;
            backdrop-filter: blur(10px);
        }
        
        .top-sign-preview img {
            width: 100%;
            border-radius: 5px;
            margin-bottom: 8px;
        }
        
        .top-sign-preview .sign-info {
            text-align: center;
            font-size: 0.8em;
        }
        
        .top-sign-preview .sign-name {
            font-weight: bold;
            margin-bottom: 4px;
        }
        
        .top-sign-preview .sign-accuracy {
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            padding-top: 30px; /* Space for top elements */
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 400px;
            gap: 20px;
            align-items: start;
            justify-content: end;
            padding-right: 20px;
        }
        
        .video-section {
            display:none;
        }
        
        .side-panel {
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
        }
        
        #videoElement {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            object-fit: cover;
            z-index: -1;
            transform: scaleX(-1); /* Mirror the video for selfie mode */
            display: block; /* ADD THIS LINE */
        }
        
        #canvas {
            display: none;
        }
        
        .reference-image {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .reference-image img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }
        
        .status-panel {
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .accuracy-display {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            margin: 10px 0;
        }
        
        .progress-bar {
            width: 100%;
            height: 20px;
            background: rgba(255,255,255,0.3);
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            transition: width 0.3s ease;
            border-radius: 10px;
        }
        
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .btn {
            padding: 12px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .btn-primary {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #f093fb 0%, #f5576c 100%);
            color: white;
        }
        
        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4);
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.2);
            color: white;
            border: 1px solid rgba(255,255,255,0.3);
        }
        
        .btn-secondary:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none !important;
        }
        
        .startup-screen {
            text-align: center;
            padding: 50px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            margin:15px;
        }
        
        .startup-screen h2 {
            font-size: 2em;
            margin-bottom: 20px;
        }
        
        .startup-screen p {
            font-size: 1.2em;
            margin-bottom: 30px;
            opacity: 0.8;
        }
        
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            font-size: 0.9em;
        }
        
        .info-item {
            background: rgba(255,255,255,0.1);
            padding: 8px;
            border-radius: 5px;
        }
        
        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 20px;
            border-radius: 8px;
            color: white;
            font-weight: bold;
            z-index: 1000;
            opacity: 0;
            transform: translateX(300px);
            transition: all 0.3s ease;
        }
        
        .notification.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        .notification.success {
            background: linear-gradient(45deg, #4ecdc4, #44a08d);
        }
        
        .notification.error {
            background: linear-gradient(45deg, #ff6b6b, #ee5a52);
        }
        
        .connection-status {
            position: fixed;
            bottom: 20px;
            right: 20px;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            background: rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        }
        
        .connection-status.connected {
            background: rgba(76, 175, 80, 0.8);
        }
        
        .connection-status.disconnected {
            background: rgba(244, 67, 54, 0.8);
        }

        .completion-screen {
            text-align: center;
            padding: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            margin: 20px;
            animation: slideInFromBottom 0.8s ease-out;
        }

        .completion-screen h1 {
            font-size: 3em;
            margin-bottom: 20px;
            color: #fff;
        }

        .celebration-emoji {
            font-size: 4em;
            margin: 20px 0;
        }

        .btn-restart {
            background: linear-gradient(45deg, #4ecdc4 0%, #44a08d 100%);
            color: white;
            padding: 15px 30px;
            font-size: 1.2em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 20px;
        }
        
        /* Demo Video Styles */
        .demo-video-screen {
            text-align: center;
            padding: 30px;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            max-width: 800px;
            margin: 0 auto;
        }
        
        .demo-video-container h2 {
            font-size: 2em;
            margin-bottom: 15px;
            color: #4ecdc4;
        }
        
        .demo-video-container p {
            font-size: 1.1em;
            margin-bottom: 25px;
            opacity: 0.9;
        }
        
        .video-wrapper {
            position: relative;
            background: rgba(0,0,0,0.3);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 25px;
            height: 300px;
        }
        
        #demoVideo {
            width: 100%;
            max-width: 600px;
            height: 300px;
            border-radius: 8px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        
        .demo-controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            margin-bottom: 25px;
            flex-wrap: wrap;
        }
        
        .demo-info {
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        
        .demo-info p {
            margin-bottom: 10px;
            font-size: 1.1em;
        }
        
        .demo-info ul {
            text-align: left;
            margin: 10px auto;
            display: inline-block;
            font-size: 0.95em;
            line-height: 1.6;
        }
        
        .demo-info li {
            margin-bottom: 5px;
            opacity: 0.9;
        }

        .timer-display {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.9);
            color: white;
            font-size: 4em;
            font-weight: bold;
            padding: 30px 50px;
            border-radius: 20px;
            border: 3px solid #4ecdc4;
            z-index: 1001;
            backdrop-filter: blur(15px);
            text-align: center;
            min-width: 200px;
        }

        .timer-display.warning {
            border-color: #ffa726;
            color: #ffa726;
        }

        .timer-display.critical {
            border-color: #ff6b6b;
            color: #ff6b6b;
            animation: pulse 0.5s infinite alternate;
        }

        @keyframes pulse {
            from { transform: translate(-50%, -50%) scale(1); }
            to { transform: translate(-50%, -50%) scale(1.05); }
        }

        /* Results Summary Styles */
        .results-screen {
            text-align: center;
            padding: 40px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            margin: 20px;
            color: white;
        }

        .results-screen h2 {
            font-size: 2.5em;
            margin-bottom: 30px;
            color: #fff;
        }

        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 30px 0;
        }

        .score-item {
            background: rgba(255,255,255,0.2);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
        }

        .score-item.lowest {
            background: rgba(255, 107, 107, 0.3);
            border: 2px solid #ff6b6b;
        }

        .score-sign-name {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .score-percentage {
            font-size: 2em;
            font-weight: bold;
        }

        .summary-stats {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
        }

        .auto-mode-btn {
            background: linear-gradient(45deg, #4ecdc4 0%, #44a08d 100%);
            color: white;
            padding: 15px 30px;
            font-size: 1.1em;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            margin: 10px;
            font-weight: bold;
        }

        .auto-mode-btn:hover {
            transform: translateY(-2px);
        }

        

        /* HUD Overlay Styles */
        .hud-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            pointer-events: none;
            z-index: 10;
            display: none;
        }

        .hud-overlay.active {
            display: block;
        }

        /* Top HUD Bar */
        .hud-top-bar {
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(10px);
            border-radius: 25px;
            padding: 8px 20px;
            display: flex;
            align-items: center;
            gap: 20px;
            font-size: 0.9em;
        }

        .hud-progress-indicator {
            color: #4ecdc4;
            font-weight: bold;
        }

        .hud-sign-name {
            color: white;
            font-weight: bold;
        }

        /* Center Target Sign Display */
        .hud-target-sign {
            position: absolute;
            top: 30px;
            left: 30px;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 15px;
            max-width: 200px;
            border: 2px solid rgba(255,255,255,0.2);
        }

        .hud-target-sign img {
            width: 100%;
            border-radius: 8px;
            margin-bottom: 10px;
        }

        .hud-target-title {
            color: #4ecdc4;
            font-size: 0.9em;
            font-weight: bold;
            margin-bottom: 5px;
            text-align: center;
        }

        .hud-target-name {
            color: white;
            font-size: 1.1em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 8px;
        }

        /* Main Status HUD (Bottom Right) */
        .hud-status-panel {
            position: absolute;
            bottom: 30px;
            right: 30px;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(15px);
            border-radius: 20px;
            padding: 20px;
            min-width: 250px;
            border: 2px solid rgba(255,255,255,0.2);
        }

        .hud-accuracy-display {
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
            color: #ff6b6b;
        }

        .hud-status-text {
            text-align: center;
            font-size: 1.1em;
            font-weight: bold;
            color: white;
            margin-bottom: 15px;
            padding: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
        }

        .hud-progress-bar {
            width: 100%;
            height: 8px;
            background: rgba(255,255,255,0.2);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }

        .hud-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1);
            transition: width 0.3s ease;
            border-radius: 10px;
        }

        /* Stats Grid (Bottom Left) */
        .hud-stats-grid {
            position: absolute;
            bottom: 30px;
            left: 30px;
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(15px);
            border-radius: 15px;
            padding: 15px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            min-width: 280px;
            border: 2px solid rgba(255,255,255,0.2);
        }

        .hud-stat-item {
            background: rgba(255,255,255,0.1);
            padding: 8px 12px;
            border-radius: 8px;
            text-align: center;
        }

        .hud-stat-label {
            font-size: 0.8em;
            color: #4ecdc4;
            font-weight: bold;
            margin-bottom: 2px;
        }

        .hud-stat-value {
            font-size: 0.9em;
            color: white;
            font-weight: bold;
        }

        /* Controls HUD (Top Right) */
        .hud-controls {
            position: absolute;
            top: 30px;
            right: 30px;
            display: flex;
            gap: 10px;
            pointer-events: auto;
        }

        .hud-btn {
            background: rgba(0,0,0,0.8);
            backdrop-filter: blur(15px);
            border: 2px solid rgba(255,255,255,0.2);
            color: white;
            padding: 10px 15px;
            border-radius: 10px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 0.9em;
        }

        .hud-btn:hover {
            background: rgba(255,255,255,0.2);
            transform: translateY(-2px);
        }

        .hud-btn.danger {
            border-color: #ff6b6b;
            color: #ff6b6b;
        }

        .hud-btn.secondary {
            border-color: #4ecdc4;
            color: #4ecdc4;
        }


        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                justify-content: center;
                padding: 10px;
            }
            
            .side-panel {
                max-width: 400px;
                margin: 0 auto;
            }
            
            .controls {
                grid-template-columns: 1fr;
            }



            @media (max-width: 768px) {
                .hud-target-sign {
                    max-width: 150px;
                    left: 15px;
                    top: 15px;
                    padding: 10px;
                }
                
                .hud-status-panel {
                    bottom: 15px;
                    right: 15px;
                    min-width: 200px;
                    padding: 15px;
                }
                
                .hud-accuracy-display {
                    font-size: 2.5em;
                }
                
                .hud-stats-grid {
                    bottom: 15px;
                    left: 15px;
                    min-width: 240px;
                    padding: 10px;
                    gap: 8px;
                }
                
                .hud-controls {
                    top: 15px;
                    right: 15px;
                    flex-direction: column;
                }
                
                .hud-top-bar {
                    top: 10px;
                    left: 15px;
                    right: 15px;
                    transform: none;
                    width: calc(100% - 30px);
                    justify-content: center;
                }
            }

            
            .demo-video-screen {
                padding: 4px;
            }
            
            .demo-controls {
                flex-direction: column;
                align-items: center;
            }
            
            .demo-controls .btn {
                width: 200px;
            }
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            
            .controls {
                grid-template-columns: 1fr;
            }
        }
    </style>
    <!-- Include Socket.IO client -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
</head>
<body>
    <!-- Top Progress Bar -->
    <div id="topProgressBar" class="top-progress-bar">
        <div id="topProgressFill" class="top-progress-fill"></div>
    </div>
    
    <!-- Top Left Sign Preview -->
    <div id="topSignPreview" class="top-sign-preview">
        <img id="topSignImage" src="" alt="Current Sign">
        <div class="sign-info">
            <div id="topSignName" class="sign-name">Loading...</div>
            <div id="topSignAccuracy" class="sign-accuracy">0%</div>
        </div>
    </div>
    
    <!-- Connection Status -->
    <div id="connectionStatus" class="connection-status disconnected">
        Connecting...
    </div>
    
    <div class="container">
        <div class="header">
        </div>
        
        <div id="startupScreen" class="startup-screen">
            <h2>Welcome to Jujutsu Hand Sign Training!</h2>
            <p>Master the ancient art of hand signs with AI-powered training</p>
            <button class="btn btn-primary" onclick="showDemoVideo()" style="font-size: 1.2em; padding: 15px 30px;">
                🎯 Manual Training
            </button>
        </div>
        
        <!-- Demo Video Screen -->
        <div id="demoVideoScreen" class="demo-video-screen" style="display: none;">
            <div class="demo-video-container">
                <h2> Training Demo</h2>
                <p>Watch this quick demo to learn how to use the hand sign trainer</p>
                
                <div class="video-wrapper">
                    <video id="demoVideo" autoplay muted  playsinline webkit-playsinline>
                        <source src="/static/vikas.mp4" type="video/mp4">
                        <p>Your browser doesn't support video playback. <button class="btn btn-primary" onclick="skipDemo()">Skip to Training</button></p>
                    </video>
                </div>
                
                <div class="demo-controls">
                    <button class="btn btn-secondary" onclick="skipDemo()">⏭ Skip Demo</button>
                    <button class="btn btn-primary" onclick="proceedToTraining()" style="display: none;" id="proceedButton">
                        Start Training Now
                    </button>
                </div>
                
            </div>
        </div>
        
        <div id="trainingInterface" class="main-content" style="display: none;">
            <!-- Hide the old side panel -->
            <div class="side-panel" style="display: none;">
                <!-- Keep existing content for fallback -->
                <div class="reference-image">
                    <h3>Target Sign</h3>
                    <div id="signName" style="font-size: 1.2em; margin-bottom: 10px;">Loading...</div>
                    <img id="referenceImage" src="" alt="Reference Sign" style="display: none;">
                    <div id="noImageText" style="padding: 50px; background: rgba(255,255,255,0.1); border-radius: 10px;">
                        No reference image available
                    </div>
                </div>

                <div id="timerDisplay" class="timer-display" style="display: none;">
                    <div id="timerNumber">5</div>
                    <div style="font-size: 0.3em; margin-top: 10px;">seconds remaining</div>
                </div>
                
                <div class="status-panel">
                    <h3>Training Status</h3>
                    <div class="accuracy-display" id="accuracyDisplay">0%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" id="progressFill" style="width: 0%;"></div>
                    </div>
                    <div id="statusText" style="text-align: center; margin-top: 10px;">Position your hand</div>
                    
                    <div class="info-grid" style="margin-top: 15px;">
                        <div class="info-item">
                            <strong>Current:</strong> <span id="currentSign">-</span>
                        </div>
                        <div class="info-item">
                            <strong>Progress:</strong> <span id="progressText">0/0</span>
                        </div>
                        <div class="info-item">
                            <strong>Stability:</strong> <span id="stabilityText">0/5</span>
                        </div>
                        <div class="info-item">
                            <strong>Status:</strong> <span id="trainingStatus">Ready</span>
                        </div>
                    </div>
                </div>
                
                <div class="controls">
                    <button class="btn btn-danger" onclick="stopTraining()">🛑 Stop</button>
                    <button class="btn btn-secondary" onclick="resetProgress()">🔄 Reset</button>
                </div>
            </div>
            
            <!-- Video elements -->
            <video id="videoElement" autoplay muted playsinline webkit-playsinline x-webkit-airplay="deny"></video>
            <canvas id="canvas"></canvas>
            
            <!-- NEW HUD OVERLAY -->
            <div id="hudOverlay" class="hud-overlay">
                <!-- Top Progress Bar -->
                <div class="hud-top-bar">
                    <div class="hud-progress-indicator">
                        <span id="hudProgressText">0/0</span> Signs Completed
                    </div>
                    <div class="hud-sign-name">
                        Current: <span id="hudCurrentSign">-</span>
                    </div>
                </div>
                
                <!-- Target Sign Display -->
                <div class="hud-target-sign">
                    <div class="hud-target-title">Target Sign</div>
                    <img id="hudTargetImage" src="" alt="Target Sign" style="display: none;">
                    <div id="hudNoImage" style="padding: 20px; text-align: center; color: #666;">
                        No image available
                    </div>
                    <div class="hud-target-name" id="hudTargetName">Loading...</div>
                </div>
                
                <!-- Main Status Panel -->
                <div class="hud-status-panel">
                    <div class="hud-accuracy-display" id="hudAccuracyDisplay">0%</div>
                    <div class="hud-status-text" id="hudStatusText">Position your hands</div>
                    <div class="hud-progress-bar">
                        <div class="hud-progress-fill" id="hudProgressFill" style="width: 0%;"></div>
                    </div>
                </div>
                
                <!-- Stats Grid -->
                <div class="hud-stats-grid">
                    <div class="hud-stat-item">
                        <div class="hud-stat-label">CURRENT</div>
                        <div class="hud-stat-value" id="hudStatCurrent">sign (1)</div>
                    </div>
                    <div class="hud-stat-item">
                        <div class="hud-stat-label">PROGRESS</div>
                        <div class="hud-stat-value" id="hudStatProgress">0/8</div>
                    </div>
                    <div class="hud-stat-item">
                        <div class="hud-stat-label">STABILITY</div>
                        <div class="hud-stat-value" id="hudStatStability">0/5</div>
                    </div>
                    <div class="hud-stat-item">
                        <div class="hud-stat-label">STATUS</div>
                        <div class="hud-stat-value" id="hudStatStatus">Ready</div>
                    </div>
                </div>
                
                <!-- Control Buttons -->
                <div class="hud-controls">
                    <button class="hud-btn danger" onclick="stopTraining()">🛑 Stop</button>
                    <button class="hud-btn secondary" onclick="resetProgress()">🔄 Reset</button>
                </div>
            </div>
            
            <div id="cameraError" style="display: none; position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; padding: 50px; background: rgba(255,0,0,0.8); border-radius: 10px; z-index: 1000;">
                <h3>📷 Camera Error</h3>
                <p>Unable to access camera. Please check:</p>
                <ul style="text-align: left; margin: 20px 0;">
                    <li>Camera is connected and not used by other apps</li>
                    <li>Browser has camera permissions</li>
                    <li>Try refreshing the page</li>
                </ul>
                <button class="btn btn-secondary" onclick="initCamera()">Retry Camera</button>
            </div>
        </div>


        <!-- Completion Screen -->
        <div id="completionScreen" class="completion-screen" style="display: none;">
            <div class="celebration-emoji">🏆</div>
            <h2>Congratulations!</h2>
            <div class="completion-message">
                You have successfully mastered all the Jujutsu hand signs!
            </div>
        </div>
        <div id="resultsScreen" class="results-screen" style="display: none;">
            <h2>📊 Training Results</h2>
            
            <div class="summary-stats">
                <h3>Overall Performance</h3>
                <div id="averageScore" style="font-size: 2em; font-weight: bold; color: #4ecdc4;">0%</div>
                <p>Average Accuracy</p>
            </div>
            
            <div class="scores-grid" id="scoresGrid">
                <!-- Scores will be populated here -->
            </div>
            
            <div id="lowestSignInfo" class="summary-stats" style="background: rgba(255, 107, 107, 0.2);">
                <h3>🎯 Sign to Practice More</h3>
                <div id="lowestSignName" style="font-size: 1.5em; font-weight: bold;">-</div>
                <div id="lowestSignScore" style="font-size: 2em; font-weight: bold; color: #ff6b6b;">0%</div>
                <p>Lowest scoring sign</p>
            </div>
            
            <div style="margin-top: 30px;">
                <button class="btn-restart" onclick="goHome()" style="background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);">🏠 Home</button>
            </div>
        </div>

    </div>
    
    <div id="notification" class="notification"></div>
    
    <script>
        // Initialize Socket.IO connection
        // const socket = io();
        const socket = io({
            transports: ['websocket', 'polling'], // Allow fallback
            timeout: 30000, // Increased timeout for iOS
            forceNew: true,
            reconnection: true,
            reconnectionDelay: 2000, // Longer delay for iOS
            reconnectionAttempts: 10,
            maxReconnectionAttempts: 15,
            upgrade: true,
            rememberUpgrade: false
        });
        
        let videoElement = null;
        let canvas = null;
        let ctx = null;
        let frameProcessingInterval = null;
        let isTrainingActive = false;
        let timerInterval = null;
        let isAutoMode = false;
        let signTimer = null;
        let currentSignStartTime = null;
        let timeRemaining = 10;

        // Connection status handlers
        socket.on('connect', function() {
            console.log('Connected to server');
            document.getElementById('connectionStatus').textContent = 'Connected';
            document.getElementById('connectionStatus').className = 'connection-status connected';
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('connectionStatus').textContent = 'Disconnected';
            document.getElementById('connectionStatus').className = 'connection-status disconnected';
        });
        

        socket.on('sign_auto_advanced', function(data) {
            if (data.training_complete) {
                showNotification('Training Complete! All signs mastered!', 'success');
                showCompletionScreen();
            } else {
                showNotification(`Advanced to: ${data.next_sign}`, 'success');
                // Start timer for next sign
                startSignTimer();
            }
        });

        socket.on('training_response', function(data) {
            if (data.success) {
                if (data.message.includes('started')) {
                    document.getElementById('startupScreen').style.display = 'none';
                    document.getElementById('trainingInterface').style.display = 'grid';
                    
                    // Show top elements
                    document.getElementById('topProgressBar').style.display = 'block';
                    document.getElementById('topSignPreview').style.display = 'block';
                    
                    isTrainingActive = true;
                    
                    // Start the first sign timer
                    startSignTimer();
                    
                    // Start periodic status updates
                    setInterval(() => {
                        if (isTrainingActive) {
                            socket.emit('get_status');
                            socket.emit('get_current_sign_image');
                        }
                    }, 500);
                    
                    // Start frame processing
                    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
                    const frameRate = isIOS ? 200 : 100;
                    frameProcessingInterval = setInterval(captureAndSendFrame, frameRate);
                    
                    showNotification('Training started!');
                } else if (data.message.includes('stopped')) {
                    // Clear timer when stopping
                    if (signTimer) {
                        clearInterval(signTimer);
                        signTimer = null;
                    }
                    document.getElementById('timerDisplay').style.display = 'none';
                    
                    document.getElementById('startupScreen').style.display = 'block';
                    document.getElementById('trainingInterface').style.display = 'none';
                    document.getElementById('demoVideoScreen').style.display = 'none';
                    
                    // Hide top elements
                    document.getElementById('topProgressBar').style.display = 'none';
                    document.getElementById('topSignPreview').style.display = 'none';
                    
                    isTrainingActive = false;
                    
                    // Stop camera
                    if (videoElement && videoElement.srcObject) {
                        const tracks = videoElement.srcObject.getTracks();
                        tracks.forEach(track => track.stop());
                        videoElement.srcObject = null;
                    }
                    
                    // Clear intervals
                    if (frameProcessingInterval) {
                        clearInterval(frameProcessingInterval);
                        frameProcessingInterval = null;
                    }
                    
                    // Reset demo video
                    const demoVideo = document.getElementById('demoVideo');
                    demoVideo.pause();
                    demoVideo.currentTime = 0;
                    document.getElementById('proceedButton').style.display = 'none';
                    
                    showNotification('Training stopped!');
                }
            } else {
                showNotification(data.message, 'error');
            }
        });
        
        socket.on('frame_result', function(data) {
            if (data.success) {
                // Update real-time accuracy in UI
                updateInstantFeedback(data);
                
                if (data.training_complete) {
                    showNotification('Training Complete! All signs mastered!', 'success');
                    showCompletionScreen();

                } else if (data.sign_completed) {
                    showNotification('Sign completed! Moving to next...', 'success');
                }
            }
        });
        
        socket.on('status_update', function(data) {
            // Update main UI
            document.getElementById('accuracyDisplay').textContent = `${data.accuracy.toFixed(1)}%`;
            document.getElementById('statusText').textContent = data.status;
            document.getElementById('currentSign').textContent = data.current_sign || 'Complete';
            document.getElementById('progressText').textContent = `${data.completed_count}/${data.total_signs}`;
            document.getElementById('stabilityText').textContent = `${data.stable_detections}/${data.required_detections}`;
            
            // Update progress bars
            const progressPercent = (data.completed_count / data.total_signs) * 100;
            document.getElementById('progressFill').style.width = `${progressPercent}%`;
            document.getElementById('topProgressFill').style.width = `${progressPercent}%`;
            
            // Update top sign preview
            document.getElementById('topSignName').textContent = data.current_sign || 'Complete';
            document.getElementById('topSignAccuracy').textContent = `${data.accuracy.toFixed(1)}%`;
            
            // Update accuracy colors
            const accuracyElement = document.getElementById('accuracyDisplay');
            const topAccuracyElement = document.getElementById('topSignAccuracy');

            if (data.training_complete && isTrainingActive) {
                setTimeout(() => {
                    showCompletionScreen();
                }, 1000);
            }

            
            let color;
            if (data.accuracy >= 65) {
                color = '#4ecdc4';
            } else if (data.accuracy >= 45) {
                color = '#ffa726';
            } else {
                color = '#ff6b6b';
            }
            
            accuracyElement.style.color = color;
            topAccuracyElement.style.color = color;
        });
        
        socket.on('sign_image_update', function(data) {
            const imgElement = document.getElementById('referenceImage');
            const noImageElement = document.getElementById('noImageText');
            const signNameElement = document.getElementById('signName');
            const topSignImage = document.getElementById('topSignImage');
            
            if (data.image) {
                // Update main reference image
                imgElement.src = `data:image/jpeg;base64,${data.image}`;
                imgElement.style.display = 'block';
                noImageElement.style.display = 'none';
                
                // Update top sign preview image
                topSignImage.src = `data:image/jpeg;base64,${data.image}`;
            } else {
                imgElement.style.display = 'none';
                noImageElement.style.display = 'block';
                
                // Clear top sign preview image
                topSignImage.src = '';
            }
            
            signNameElement.textContent = data.sign_name || 'Training Complete';
        });
        
        socket.on('sign_changed', function(data) {
            showNotification(data.message);
            socket.emit('get_current_sign_image');
        });
        
        socket.on('progress_reset', function(data) {
            showNotification(data.message);
            socket.emit('get_current_sign_image');
        });

        socket.on('training_results_update', function(data) {
            displayTrainingResults(data);
        });

        // Add this new function
        function displayTrainingResults(data) {
            // Hide other screens and show results
            document.getElementById('completionScreen').style.display = 'none';
            document.getElementById('resultsScreen').style.display = 'block';
            
            // Update average score
            document.getElementById('averageScore').textContent = `${data.average_accuracy.toFixed(1)}%`;
            
            // Update lowest sign info
            if (data.lowest_sign.name) {
                document.getElementById('lowestSignName').textContent = data.lowest_sign.name;
                document.getElementById('lowestSignScore').textContent = `${data.lowest_sign.accuracy.toFixed(1)}%`;
            }
            
            // Populate scores grid
            const scoresGrid = document.getElementById('scoresGrid');
            scoresGrid.innerHTML = ''; // Clear existing content
            
            data.results.forEach((result, index) => {
                const scoreItem = document.createElement('div');
                scoreItem.className = 'score-item';
                
                // Highlight the lowest scoring sign
                if (result.sign_name === data.lowest_sign.name) {
                    scoreItem.classList.add('lowest');
                }
                
                scoreItem.innerHTML = `
                    <div class="score-sign-name">${result.sign_name}</div>
                    <div class="score-percentage" style="color: ${getAccuracyColor(result.max_accuracy)}">${result.max_accuracy.toFixed(1)}%</div>
                `;
                
                scoresGrid.appendChild(scoreItem);
            });
        }

        // Add helper function for color coding
        function getAccuracyColor(accuracy) {
            if (accuracy >= 65) return '#4ecdc4';
            if (accuracy >= 45) return '#ffa726';
            return '#ff6b6b';
        }

        function goHome() {
            window.location.reload();
        }

        
        function showNotification(message, type = 'success') {
            const notification = document.getElementById('notification');
            notification.textContent = message;
            notification.className = `notification ${type}`;
            notification.classList.add('show');
            
            setTimeout(() => {
                notification.classList.remove('show');
            }, 3000);
        }

        // Add this new function
        function updateTimerDisplay() {
            const timerDisplay = document.getElementById('timerDisplay');
            const timerNumber = document.getElementById('timerNumber');
            
            if (timeRemaining > 0) {
                timerDisplay.style.display = 'block';
                timerNumber.textContent = timeRemaining;
                
                // Style changes based on time remaining
                timerDisplay.className = 'timer-display';
                if (timeRemaining <= 2) {
                    timerDisplay.classList.add('critical');
                } else if (timeRemaining <= 3) {
                    timerDisplay.classList.add('warning');
                }
            } else {
                timerDisplay.style.display = 'none';
            }
        }

        // Add this new function
        function startSignTimer() {
            // Clear any existing timer
            if (signTimer) {
                clearInterval(signTimer);
            }
            
            timeRemaining = 10;
            currentSignStartTime = Date.now();
            
            // Update timer display immediately
            updateTimerDisplay();
            
            // Start countdown timer
            signTimer = setInterval(() => {
                timeRemaining--;
                updateTimerDisplay();
                
                if (timeRemaining <= 0) {
                    clearInterval(signTimer);
                    signTimer = null;
                    
                    // Auto-advance to next sign
                    socket.emit('auto_advance_sign');
                }
            }, 1000);
        }

        
        function showDemoVideo() {
            // Hide startup screen and show demo video
            document.getElementById('startupScreen').style.display = 'none';
            document.getElementById('demoVideoScreen').style.display = 'block';
            
            const demoVideo = document.getElementById('demoVideo');
            const proceedButton = document.getElementById('proceedButton');
            
            // Show proceed button when video ends
            demoVideo.addEventListener('ended', function() {
                proceedButton.style.display = 'inline-block';
                showNotification('Demo complete! Ready to start training? 🎬');
            });
            
            // Auto-play the demo video
            demoVideo.play().catch(error => {
                console.log('Auto-play failed:', error);
                showNotification('Click play to watch the demo video 📹', 'error');
            });
            
            showNotification('Watching training demo... ⏯️');
        }

        function showCompletionScreen() {
            // Stop training but don't go to home
            isTrainingActive = false;
            
            // Clear timer
            if (signTimer) {
                clearInterval(signTimer);
                signTimer = null;
            }
            document.getElementById('timerDisplay').style.display = 'none';
            
            // Stop camera and intervals
            if (videoElement && videoElement.srcObject) {
                const tracks = videoElement.srcObject.getTracks();
                tracks.forEach(track => track.stop());
                videoElement.srcObject = null;
            }
            
            if (frameProcessingInterval) {
                clearInterval(frameProcessingInterval);
                frameProcessingInterval = null;
            }
            
            // Hide training interface and show completion screen briefly
            document.getElementById('trainingInterface').style.display = 'none';
            document.getElementById('topProgressBar').style.display = 'none';
            document.getElementById('topSignPreview').style.display = 'none';
            document.getElementById('completionScreen').style.display = 'block';
            
            showNotification('🎉 All hand signs completed! 🥷', 'success');
            
            // Auto-show results after 2 seconds
            setTimeout(() => {
                socket.emit('get_training_results');
            }, 2000);
        }

        function restartTraining() {
            // Hide completion screen
            document.getElementById('completionScreen').style.display = 'none';
            document.getElementById('resultsScreen').style.display = 'none';
            
            // Reset progress on server
            socket.emit('reset_progress');
            
            // Start training again
            setTimeout(() => {
                startTraining();
            }, 1000);
        }

        
        function skipDemo() {
            showNotification('Demo skipped! Starting training... ⏭️');
            proceedToTraining();
        }
        
        function proceedToTraining() {
            // Hide demo screen
            document.getElementById('demoVideoScreen').style.display = 'none';
            
            // Pause and reset demo video
            const demoVideo = document.getElementById('demoVideo');
            demoVideo.pause();
            demoVideo.currentTime = 0;
            
            // Start the actual training
            startTraining();
        }
        
        async function initCamera() {
            try {
                videoElement = document.getElementById('videoElement');
                canvas = document.getElementById('canvas');
                
                if (!videoElement || !canvas) {
                    throw new Error('Video or canvas element not found');
                }
                
                ctx = canvas.getContext('2d');
                
                // iOS-compatible video constraints
                const constraints = {
                    video: {
                        width: { ideal: 640, max: 1280 },
                        height: { ideal: 480, max: 720 },
                        facingMode: 'user', // Front camera
                        frameRate: { ideal: 30, max: 30 }
                    },
                    audio: false
                };
                
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                
                videoElement.srcObject = stream;
                
                // iOS requires explicit play() call and user interaction
                videoElement.muted = true;
                videoElement.playsInline = true; // Critical for iOS
                videoElement.autoplay = true;
                
                // Wait for video to be ready
                await new Promise((resolve) => {
                    videoElement.onloadedmetadata = () => {
                        resolve();
                    };
                });
                
                // Explicitly play for iOS
                await videoElement.play();
                
                const cameraError = document.getElementById('cameraError');
                if (cameraError) {
                    cameraError.style.display = 'none';
                }
                
                // Set canvas size based on actual video dimensions
                canvas.width = videoElement.videoWidth || 640;
                canvas.height = videoElement.videoHeight || 480;
                
                showNotification('Camera connected successfully! 📷');
                return true;
            } catch (error) {
                console.error('Camera error:', error);
                const cameraError = document.getElementById('cameraError');
                if (cameraError) {
                    cameraError.style.display = 'block';
                }
                showNotification('Failed to access camera: ' + error.message, 'error');
                return false;
            }
        }
        
        function captureAndSendFrame() {
            if (!videoElement || !canvas || !ctx || !isTrainingActive) return;
            
            try {
                // Check if video is actually playing
                if (videoElement.readyState < 2) return;
                
                const videoWidth = videoElement.videoWidth;
                const videoHeight = videoElement.videoHeight;
                
                if (videoWidth === 0 || videoHeight === 0) return;
                
                // Update canvas size if needed
                if (canvas.width !== videoWidth || canvas.height !== videoHeight) {
                    canvas.width = videoWidth;
                    canvas.height = videoHeight;
                }
                
                // Clear canvas first (iOS optimization)
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw video frame to canvas
                ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                
                // Convert to base64 with lower quality for iOS performance
                const frameData = canvas.toDataURL('image/jpeg', 0.4);
                
                // Send to backend via WebSocket
                socket.emit('process_frame', { frame: frameData });
            } catch (error) {
                console.log('Frame capture error:', error);
            }
        }
        
        function updateInstantFeedback(data) {
            // Update accuracy display immediately from frame processing
            const accuracyElement = document.getElementById('accuracyDisplay');
            if (accuracyElement) {
                accuracyElement.textContent = `${data.accuracy.toFixed(1)}%`;
                
                // Update accuracy color
                if (data.accuracy >= 65) {
                    accuracyElement.style.color = '#4ecdc4';
                } else if (data.accuracy >= 45) {
                    accuracyElement.style.color = '#ffa726';
                } else {
                    accuracyElement.style.color = '#ff6b6b';
                }
            }
            
            // Update status
            const statusElement = document.getElementById('statusText');
            if (statusElement) {
                statusElement.textContent = data.status;
            }
            
            // Visual feedback for hand detection - FIXED
            const videoElement = document.getElementById('videoElement');
            if (videoElement) {
                if (data.hand_detected) {
                    videoElement.style.border = data.accuracy >= 65 ? '3px solid #4ecdc4' : '3px solid #ffa726';
                } else {
                    videoElement.style.border = '3px solid #ff6b6b';
                }
            }
        }
        
        async function startTraining() {
            showNotification('Initializing training...', 'success');
            
            // First initialize camera
            const cameraSuccess = await initCamera();
            if (!cameraSuccess) {
                return;
            }
            
            // Then start training via WebSocket
            socket.emit('start_training');
        }
        
        function stopTraining() {
            socket.emit('stop_training');
        }
        
        function resetProgress() {
            if (confirm('Are you sure you want to reset all progress?')) {
                socket.emit('reset_progress');
            }
        }
    </script>
</body>
</html>'''

# Save the HTML template
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(html_template)

if __name__ == "__main__":
    # Make sure the images folder exists
    if not os.path.exists("handsign"):
        os.makedirs("handsign")
        print("📁 Created 'handsign' folder. Please add your hand sign images here.")
    
    
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
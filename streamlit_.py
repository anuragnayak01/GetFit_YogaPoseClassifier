import streamlit as st
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import time
import mediapipe as mp
import numpy as np
import threading
import os

st.write(" ##### Code for Bharat Season - 2  ")
st.write("###### ~Anurag Nayak")
st.write("# GetFit - Yoga Pose Classifier")


# Add option selector
input_type = st.selectbox(
    "Choose input type:",
    ["Image", "Video", "Live Webcam"]
)

if input_type == "Image" :
    uploaded_file = st.file_uploader("Input (Image)", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        # Save to a temporary file
        image_path = os.path.join("", uploaded_file.name)

        # Make sure the temp directory exists
        os.makedirs("temp", exist_ok=True)

        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Now use the path with OpenCV
        save_path = cv2.imread(image_path)

        # Display using Streamlit (convert BGR to RGB)
        # st.image(cv2.cvtColor(save_path, cv2.COLOR_BGR2RGB), caption="Read via image path")
        #  # Resize the image to specific dimensions
        display_resized = cv2.resize(cv2.cvtColor(save_path, cv2.COLOR_BGR2RGB),(800, 600) )  # width, height
        st.image(display_resized, caption="Read via image path")

        # Optionally show the image path
        st.text(f"Image read from path: {image_path}")
        
            
        # STEP 1: Import the necessary modules
        # STEP 1: Import the necessary modules
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        import keras
        import pandas as pd
        import math
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

        # ==== CONFIGURATION ====
        model_path = "yoga_model.keras"
        class_names = ['Chair', 'Cobra', 'Dog', 'Tree', 'Warrior']
        threshold = 0.8
        torso_size_multiplier = 2.5

        # ==== LOAD YOGA CLASSIFICATION MODEL ====
        print("Loading yoga pose classification model...")
        try:
            yoga_model = keras.models.load_model(model_path)
            print(f"✓ Yoga model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading yoga model: {e}")
            exit()

        # ==== POSE LANDMARK CLASS ====
        class PoseLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y 
                self.z = z
                self.visibility = visibility

        # ==== COLUMN NAMES FOR MODEL ====
        col_names = []
        for i in range(33):
            landmark_name = f"LANDMARK_{i}"
            col_names.extend([f"{landmark_name}_X", f"{landmark_name}_Y", 
                            f"{landmark_name}_Z", f"{landmark_name}_V"])

        # ==== NORMALIZATION FUNCTION ====
        def normalize_landmarks(landmarks):
            """Normalize landmarks using the same method as training"""
            
            # Get key landmark indices
            left_hip_idx = 23
            right_hip_idx = 24  
            left_shoulder_idx = 11
            right_shoulder_idx = 12
            
            # Calculate center point (hip midpoint)
            center_x = (landmarks[right_hip_idx].x + landmarks[left_hip_idx].x) * 0.5
            center_y = (landmarks[right_hip_idx].y + landmarks[left_hip_idx].y) * 0.5
            
            # Calculate shoulder midpoint
            shoulders_x = (landmarks[right_shoulder_idx].x + landmarks[left_shoulder_idx].x) * 0.5
            shoulders_y = (landmarks[right_shoulder_idx].y + landmarks[left_shoulder_idx].y) * 0.5
            
            # Calculate max distance for normalization
            max_distance = max(
                max(math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in landmarks),
                math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2) * torso_size_multiplier
            )
            
            # Normalize all landmarks
            normalized_features = []
            for lm in landmarks:
                norm_x = (lm.x - center_x) / max_distance
                norm_y = (lm.y - center_y) / max_distance
                norm_z = lm.z / max_distance
                visibility = lm.visibility
                normalized_features.extend([norm_x, norm_y, norm_z, visibility])
            
            return normalized_features

        # ==== PREDICTION FUNCTION ====
        def predict_pose_from_landmarks(landmarks_data, model, class_names, threshold=0.8):
            """Predict yoga pose from 33 landmarks"""
            
            try:
                # Normalize landmarks
                normalized_data = normalize_landmarks(landmarks_data)
                
                # Create DataFrame
                input_df = pd.DataFrame([normalized_data], columns=col_names)
                
                # Make prediction
                predictions = model.predict(input_df, verbose=0)[0]
                max_confidence = max(predictions)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_class_idx]
                
                # Apply threshold
                if max_confidence < threshold:
                    predicted_class = "Unknown Pose"
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': max_confidence,
                    'all_probabilities': dict(zip(class_names, predictions)),
                    'accepted': max_confidence >= threshold
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                return {
                    'predicted_class': "Error",
                    'confidence': 0.0,
                    'accepted': False
                }

        # ==== ENHANCED DRAWING FUNCTION WITH COMPACT TEXT BOX ====
        def draw_landmarks_on_image(rgb_image, detection_result):
            """Enhanced function to draw pose landmarks and yoga pose prediction with compact text box"""
            
            # Convert to BGR for OpenCV operations
            annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            
            if not detection_result.pose_landmarks:
                # No pose detected
                cv2.putText(annotated_image, "No pose detected", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Get pose landmarks
            pose_landmarks = detection_result.pose_landmarks[0]
            height, width = annotated_image.shape[:2]
            
            # Define pose connections
            pose_connections = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10),
                # Upper body
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                (11, 23), (12, 24), (23, 24),
                # Lower body  
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for landmark in pose_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))
            
            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    start_point = points[start_idx]
                    end_point = points[end_idx]
                    cv2.line(annotated_image, start_point, end_point, (255, 255, 255), 3)
            
            # Draw landmarks
            for i, point in enumerate(points):
                cv2.circle(annotated_image, point, 6, (255, 0, 0), -1)  # Blue filled circle
                cv2.circle(annotated_image, point, 6, (255, 255, 255), 2)  # White border
            
            # ==== YOGA POSE CLASSIFICATION ====
            # Convert landmarks to PoseLandmark objects for prediction
            landmarks_objects = []
            for landmark in pose_landmarks:
                landmarks_objects.append(PoseLandmark(
                    landmark.x, landmark.y, landmark.z, landmark.visibility
                ))
            
            # Predict yoga pose
            result = predict_pose_from_landmarks(landmarks_objects, yoga_model, class_names, threshold)
            
            # Draw prediction results with COMPACT layout
            pose_name = result['predicted_class']
            confidence = result['confidence']
            
            # Calculate text size with smaller fonts for compact display
            text1 = f"Pose: {pose_name}"
            text2 = f"{confidence:.1%}"
            
            # Get text size for compact rectangle sizing
            (text_width1, text_height1), _ = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            (text_width2, text_height2), _ = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            
            # Make rectangle size based on text with MINIMAL padding
            rect_width = max(text_width1, text_width2) + 16  # Reduced from 30 to 16
            rect_height = text_height1 + text_height2 + 16   # Reduced from 30 to 16
            
            # Position rectangle in top-left corner with margin
            rect_x = 15  # Reduced from 20 to 15
            rect_y = 15  # Reduced from 20 to 15
            
            # Background rectangle for text (semi-transparent)
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            alpha = 0.7
            cv2.addWeighted(overlay, alpha, annotated_image, 1 - alpha, 0, annotated_image)
            
            # Draw main prediction text with compact positioning and smaller fonts
            cv2.putText(annotated_image, text1, (rect_x + 8, rect_y + 18), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Reduced font size and positioning
            cv2.putText(annotated_image, text2, (rect_x + 8, rect_y + 18 + text_height1 + 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Reduced spacing
            
            # Change border color based on confidence with thinner border
            border_color = (0, 255, 0) if result['accepted'] else (0, 0, 255)
            cv2.rectangle(annotated_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), border_color, 2)
            
            # Display all class probabilities in a smaller, more compact box
            if result['accepted']:
                prob_y_start = rect_y + rect_height + 10  # Reduced spacing
                prob_rect_height = len(class_names) * 16 + 20  # Reduced line height and padding
                prob_rect_width = 180  # Reduced from 250 to 180
                
                # Background for probabilities
                overlay2 = annotated_image.copy()
                cv2.rectangle(overlay2, (rect_x, prob_y_start), (rect_x + prob_rect_width, prob_y_start + prob_rect_height), (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.7, annotated_image, 0.3, 0, annotated_image)
                
                cv2.putText(annotated_image, "All Predictions:", (rect_x + 6, prob_y_start + 16), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)  # Smaller font
                
                for i, (class_name, prob) in enumerate(result['all_probabilities'].items()):
                    y_pos = prob_y_start + 28 + (i * 16)  # Reduced line spacing
                    color = (0, 255, 0) if class_name == pose_name else (255, 255, 255)
                    cv2.putText(annotated_image, f"{class_name}: {prob:.1%}", (rect_x + 6, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)  # Smaller font
                
                # Border for probabilities
                cv2.rectangle(annotated_image, (rect_x, prob_y_start), (rect_x + prob_rect_width, prob_y_start + prob_rect_height), (255, 255, 255), 1)
            
            # Convert back to RGB for matplotlib
            return cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        # STEP 2: Create a PoseLandmarker object
        base_options = python.BaseOptions(model_asset_path="pose_landmarker_full.task")
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path="pose_landmarker_full.task"),
            running_mode=RunningMode.IMAGE
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        # ==== SINGLE IMAGE PROCESSING ====

        # Load the image
        image_array = cv2.imread(image_path)  # Load as array
        if image_array is None:
            st.error(f"❌ Failed to read image from {image_path}")
            st.stop()

        # Convert to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Detect pose synchronously
        detection_result = detector.detect(mp_image)

        # Process results
        if detection_result.pose_landmarks:
            print("✓ Pose landmarks detected")

            # Draw results and get prediction
            result_image_rgb = draw_landmarks_on_image(rgb_image, detection_result)
            result_image_bgr = cv2.cvtColor(result_image_rgb, cv2.COLOR_RGB2BGR)

            # Convert to PoseLandmark objects for prediction
            pose_landmarks = detection_result.pose_landmarks[0]
            landmarks_objects = [PoseLandmark(lm.x, lm.y, lm.z, lm.visibility) for lm in pose_landmarks]

            result = predict_pose_from_landmarks(landmarks_objects, yoga_model, class_names, threshold)
            pose_name = result['predicted_class']
            confidence = result['confidence']
            print(f"✓ Detected: {pose_name} (Confidence: {confidence:.1%})")

        else:
            print("❌ No pose detected")
            result_image_bgr = image_array.copy()
            cv2.putText(result_image_bgr, "No pose detected", (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Save the result
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"output_{name}_processed{ext}"
        cv2.imwrite(output_path, result_image_bgr)
        print(f"✓ Result saved as: {output_path}")

        # Resize and display the result in Streamlit
        try:
            display_image = result_image_bgr.copy()
            height, width = display_image.shape[:2]

            # Resize for display
            max_display_width = 1600
            max_display_height = 1200
            min_display_width = 800
            min_display_height = 600

            scale_factor = 1.0
            if width < min_display_width or height < min_display_height:
                scale_factor = max(min_display_width / width, min_display_height / height)
            elif width > max_display_width or height > max_display_height:
                scale_factor = min(max_display_width / width, max_display_height / height)

            if scale_factor != 1.0:
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                display_image = cv2.resize(display_image, (new_width, new_height))
                print(f"Image resized: {width}x{height} -> {new_width}x{new_height}")

            # Display in Streamlit
            display_resized = cv2.resize(display_image, (800, 600))
            st.image(display_resized, caption="Your Output")
            st.balloons()

        except Exception as e:
            print(f"⚠️ Could not display image: {e}")
            print("Image saved successfully.")
elif input_type == "Video":
    uploaded_file = st.file_uploader("Input (Video)", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # Ensure the temp directory exists
        os.makedirs("temp", exist_ok=True)

        # Save to temp directory with correct path
        save_path = os.path.join("temp", uploaded_file.name)

        # Write the uploaded file to disk
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.video(save_path)
        print("Hello")
        print(save_path)

        import cv2
        import mediapipe as mp
        import numpy as np
        import time
        import keras
        import pandas as pd
        import math

        # ==== CONFIGURATION ====
        model_path = "yoga_model.keras"
        class_names = ['Chair', 'Cobra', 'Dog', 'Tree', 'Warrior']
        threshold = 0.8
        torso_size_multiplier = 2.5

        # ==== LOAD MODEL ====
        print("Loading yoga pose model...")
        try:
            model = keras.models.load_model(model_path)
            print(f"✓ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            exit()

        # ==== MEDIAPIPE SETUP ====
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Drawing utilities for visualization
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # ==== POSE LANDMARK CLASS ====
        class PoseLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y 
                self.z = z
                self.visibility = visibility

        # ==== COLUMN NAMES FOR MODEL ====
        col_names = []
        for i in range(33):
            landmark_name = f"LANDMARK_{i}"
            col_names.extend([f"{landmark_name}_X", f"{landmark_name}_Y", 
                            f"{landmark_name}_Z", f"{landmark_name}_V"])

        # ==== NORMALIZATION FUNCTION ====
        def normalize_landmarks(landmarks):
            """Normalize landmarks using the same method as training"""
            
            # Get key landmark indices
            left_hip_idx = 23
            right_hip_idx = 24  
            left_shoulder_idx = 11
            right_shoulder_idx = 12
            
            # Calculate center point (hip midpoint)
            center_x = (landmarks[right_hip_idx].x + landmarks[left_hip_idx].x) * 0.5
            center_y = (landmarks[right_hip_idx].y + landmarks[left_hip_idx].y) * 0.5
            
            # Calculate shoulder midpoint
            shoulders_x = (landmarks[right_shoulder_idx].x + landmarks[left_shoulder_idx].x) * 0.5
            shoulders_y = (landmarks[right_shoulder_idx].y + landmarks[left_shoulder_idx].y) * 0.5
            
            # Calculate max distance for normalization
            max_distance = max(
                max(math.sqrt((lm.x - center_x) ** 2 + (lm.y - center_y) ** 2) for lm in landmarks),
                math.sqrt((shoulders_x - center_x) ** 2 + (shoulders_y - center_y) ** 2) * torso_size_multiplier
            )
            
            # Normalize all landmarks
            normalized_features = []
            for lm in landmarks:
                norm_x = (lm.x - center_x) / max_distance
                norm_y = (lm.y - center_y) / max_distance
                norm_z = lm.z / max_distance
                visibility = lm.visibility
                normalized_features.extend([norm_x, norm_y, norm_z, visibility])
            
            return normalized_features

        # ==== PREDICTION FUNCTION ====
        def predict_pose_from_landmarks(landmarks_data, model, class_names, threshold=0.8):
            """Predict yoga pose from 33 landmarks"""
            
            try:
                # Normalize landmarks
                normalized_data = normalize_landmarks(landmarks_data)
                
                # Create DataFrame
                input_df = pd.DataFrame([normalized_data], columns=col_names)
                
                # Make prediction
                predictions = model.predict(input_df, verbose=0)[0]
                max_confidence = max(predictions)
                predicted_class_idx = np.argmax(predictions)
                predicted_class = class_names[predicted_class_idx]
                
                # Apply threshold
                if max_confidence < threshold:
                    predicted_class = "Unknown Pose"
                
                return {
                    'predicted_class': predicted_class,
                    'confidence': max_confidence,
                    'all_probabilities': dict(zip(class_names, predictions)),
                    'accepted': max_confidence >= threshold
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                return {
                    'predicted_class': "Error",
                    'confidence': 0.0,
                    'accepted': False
                }

        # ==== LANDMARK EXTRACTION ====
        def extract_landmarks_to_array(detection_result):
            """Extract 33 pose landmarks with x, y, z, visibility"""
            if detection_result.pose_landmarks:
                pose_landmarks = detection_result.pose_landmarks[0]
                landmarks_array = []
                for landmark in pose_landmarks:
                    landmarks_array.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return np.array(landmarks_array)
            else:
                return None

        # ==== CUSTOM DRAWING FUNCTION ====
        def draw_pose_landmarks_custom(image, pose_landmarks):
            """Custom function to draw pose landmarks compatible with new MediaPipe API"""
            if not pose_landmarks:
                return image
            
            height, width = image.shape[:2]
            
            # Define pose connections (same as MediaPipe POSE_CONNECTIONS)
            pose_connections = [
                # Face
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10),
                # Upper body
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                (11, 23), (12, 24), (23, 24),
                # Lower body  
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for landmark in pose_landmarks:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                points.append((x, y))
            
            # Draw connections
            for connection in pose_connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    start_point = points[start_idx]
                    end_point = points[end_idx]
                    cv2.line(image, start_point, end_point, (255, 255, 255), 2)  # White lines
            
            # Draw landmarks
            for i, point in enumerate(points):
                # Blue dots for all landmarks
                cv2.circle(image, point, 4, (255, 0, 0), -1)  # Blue filled circle
                cv2.circle(image, point, 4, (255, 255, 255), 1)  # White border
            
            return image

        # ==== GLOBAL VARIABLES ====
        latest_detection_result = None
        latest_landmarks = None

        # ==== CALLBACK FUNCTION ====
        def result_callback(detection_result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
            """Callback function for pose detection results"""
            global latest_detection_result, latest_landmarks
            latest_detection_result = detection_result
            latest_landmarks = extract_landmarks_to_array(detection_result)

        # ==== MEDIAPIPE SETUP ====
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback
        )

        try:
            detector = PoseLandmarker.create_from_options(options)
            print("✓ MediaPipe pose detector initialized")
        except Exception as e:
            print(f"❌ Error initializing MediaPipe: {e}")
            exit()

        # ==== INITIALIZE VIDEO CAPTURE FIRST ====
        print("Initializing video capture...")
        cap = cv2.VideoCapture(save_path)

        if not cap.isOpened():
            print("❌ Error: Could not open video file")
            exit()

        # ==== VIDEO WRITER SETUP (AFTER cap is initialized) ====
        # Get video properties from the opened video
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Validate video properties
        if frame_width == 0 or frame_height == 0 or fps == 0:
            print(f"❌ Error: Invalid video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}")
            cap.release()
            exit()
        
        
        import datetime
       

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"yoga_pose_output_{timestamp}.mp4v"
        out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))


        # Check if VideoWriter was initialized successfully
        if not out.isOpened():
            print("❌ Error: Could not initialize video writer")
            print("Trying alternative codec...")
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output_filename = "yoga_pose_output_{timestamp}.avi"
            out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
            
            if not out.isOpened():
                print("❌ Error: Video writer initialization failed with both codecs")
                cap.release()
                exit()

        print(f"✓ Video writer initialized - Output: {output_filename}")
        print(f"  Resolution: {frame_width}x{frame_height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # ==== MAIN PROCESSING LOOP ====
        print("Starting video processing... Press ESC to exit")
        frame_count = 0
        status_text = st.empty()  # Place this outside the loop

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video or failed to read frame")
                break
            
            frame_count += 1
            progress = 1
            # Show progress every 30 frames
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")
            status_text.text(f"Processing frame {frame_count}/{total_frames} ")

            image_height, image_width = image.shape[:2]
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Get timestamp
            timestamp_ms = int(time.time() * 1000)
            
            # Detect pose asynchronously
            detector.detect_async(mp_image, timestamp_ms)
            
            # Draw pose landmarks if available
            if latest_detection_result and latest_detection_result.pose_landmarks:
                # Draw pose landmarks using custom function
                pose_landmarks = latest_detection_result.pose_landmarks[0]
                image = draw_pose_landmarks_custom(image, pose_landmarks)
                
                # Make yoga pose prediction
                if latest_landmarks is not None:
                    # Convert to PoseLandmark objects
                    landmarks_objects = []
                    for i in range(0, len(latest_landmarks), 4):
                        landmarks_objects.append(PoseLandmark(
                            latest_landmarks[i], 
                            latest_landmarks[i+1], 
                            latest_landmarks[i+2], 
                            latest_landmarks[i+3]
                        ))
                    
                    # Predict pose
                    result = predict_pose_from_landmarks(landmarks_objects, model, class_names, threshold)
                    
                    # Draw prediction results
                    pose_name = result['predicted_class']
                    confidence = result['confidence']
                    
                    # Background rectangle for text
                    cv2.rectangle(image, (10, 10), (500, 100), (0, 0, 0), -1)

                    
                    # Draw text
                    cv2.putText(image, f"Pose: {pose_name}", (20, 45),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.putText(image, f"Confidence: {confidence:.1%}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                    # Change color based on confidence
                    color = (0, 255, 0) if result['accepted'] else (0, 0, 255)
                    cv2.rectangle(image, (10, 10), (500, 100), color, 4)  # Thickness = 4

            
            else:
                # No pose detected
                cv2.putText(image, "No pose detected", (15, 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write the frame to output video
            out.write(image)
            
            # Display the image (optional - comment out for faster processing)
            # cv2.imshow("Yoga Pose Detection", image)
            # st.image(image, caption="Read via image path")
            # # Break on ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        # Cleanup
        cap.release()
        out.release()  # Don't forget to release the video writer
        cv2.destroyAllWindows()

        # Verify output file
        import os
        if os.path.exists(output_filename):
            file_size = os.path.getsize(output_filename)
            print(f"✓ Video processing completed!")
            print(f"✓ Output saved as: {output_filename}")
            st.video(output_filename)
            print(f"✓ File size: {file_size / (1024*1024):.2f} MB")
        else:
            print("❌ Error: Output file was not created")

        print("✓ Application closed")
        st.balloons()

elif input_type == "Live Webcam":
        import streamlit as st
        import cv2
        import numpy as np
        import av
        import mediapipe as mp
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import keras
        import pandas as pd
        import math
        import time
        from typing import Optional

        # ==== STREAMLIT PAGE CONFIG ====
        st.set_page_config(page_title="Yoga Pose Detection", layout="wide")

        # ==== CONFIGURATION ====
        MODEL_PATH = "yoga_model.keras"
        CLASS_NAMES = ['Chair', 'Cobra', 'Dog', 'Tree', 'Warrior']
        THRESHOLD = 0.8
        TORSO_SIZE_MULTIPLIER = 2.5

        # ==== MEDIAPIPE SETUP ====
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        # ==== WEBRTC CONFIGURATION ====
        RTC_CONFIGURATION = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # ==== COLUMN NAMES ====
        col_names = []
        for i in range(33):
            col_names.extend([f"LANDMARK_{i}_X", f"LANDMARK_{i}_Y", f"LANDMARK_{i}_Z", f"LANDMARK_{i}_V"])

        class PoseLandmark:
            def __init__(self, x, y, z, visibility):
                self.x = x
                self.y = y
                self.z = z
                self.visibility = visibility

        # ==== LOAD MODEL ====
        @st.cache_resource
        def load_yoga_model():
            try:
                model = keras.models.load_model(MODEL_PATH)
                return model
            except Exception as e:
                st.error(f"Error loading model: {e}")
                return None

        # ==== NORMALIZATION ====
        def normalize_landmarks(landmarks):
            left_hip_idx, right_hip_idx = 23, 24
            left_shoulder_idx, right_shoulder_idx = 11, 12

            center_x = (landmarks[right_hip_idx].x + landmarks[left_hip_idx].x) * 0.5
            center_y = (landmarks[right_hip_idx].y + landmarks[left_hip_idx].y) * 0.5

            shoulders_x = (landmarks[right_shoulder_idx].x + landmarks[left_shoulder_idx].x) * 0.5
            shoulders_y = (landmarks[right_shoulder_idx].y + landmarks[left_shoulder_idx].y) * 0.5

            max_distance = max(
                max(math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2) for lm in landmarks),
                math.sqrt((shoulders_x - center_x)**2 + (shoulders_y - center_y)**2) * TORSO_SIZE_MULTIPLIER
            )

            normalized_features = []
            for lm in landmarks:
                norm_x = (lm.x - center_x) / max_distance
                norm_y = (lm.y - center_y) / max_distance
                norm_z = lm.z / max_distance
                normalized_features.extend([norm_x, norm_y, norm_z, lm.visibility])

            return normalized_features

        # ==== PREDICTION ====
        def predict_pose_from_landmarks(landmarks_data, model, class_names, threshold=0.8):
            try:
                normalized_data = normalize_landmarks(landmarks_data)
                input_df = pd.DataFrame([normalized_data], columns=col_names)
                predictions = model.predict(input_df, verbose=0)[0]
                max_conf = max(predictions)
                pred_class_idx = np.argmax(predictions)
                pred_class = class_names[pred_class_idx] if max_conf >= threshold else "Unknown Pose"
                return {
                    'predicted_class': pred_class,
                    'confidence': max_conf,
                    'all_probabilities': dict(zip(class_names, predictions)),
                    'accepted': max_conf >= threshold
                }
            except Exception as e:
                print(f"Prediction error: {e}")
                return {'predicted_class': "Error", 'confidence': 0.0, 'accepted': False}

        # ==== DRAW POSE ====
        def draw_pose_landmarks_custom(image, pose_landmarks):
            if not pose_landmarks:
                return image

            height, width = image.shape[:2]
            pose_connections = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
                (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                (11, 23), (12, 24), (23, 24),
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]

            points = [(int(lm.x * width), int(lm.y * height)) for lm in pose_landmarks]
            for start, end in pose_connections:
                cv2.line(image, points[start], points[end], (255, 255, 255), 3)
            for pt in points:
                cv2.circle(image, pt, 6, (255, 0, 0), -1)
                cv2.circle(image, pt, 6, (255, 255, 255), 1)
            return image

        # ==== VIDEO PROCESSOR CLASS ====
        class YogaPoseProcessor:
            def __init__(self):
                self.pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.model = load_yoga_model()
                self.frame_count = 0
                self.last_time = time.time()
                
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                
                # Flip the image horizontally for a later selfie-view display
                img = cv2.flip(img, 1)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process the image and detect poses
                results = self.pose.process(img_rgb)
                
                # Draw pose landmarks and make predictions
                if results.pose_landmarks:
                    # Draw pose landmarks
                    img = draw_pose_landmarks_custom(img, results.pose_landmarks.landmark)
                    
                    # Make pose prediction if model is loaded
                    if self.model is not None:
                        # Convert landmarks to our format
                        landmarks_objects = []
                        for lm in results.pose_landmarks.landmark:
                            landmarks_objects.append(PoseLandmark(lm.x, lm.y, lm.z, lm.visibility))
                        
                        # Predict pose
                        prediction_result = predict_pose_from_landmarks(
                            landmarks_objects, self.model, CLASS_NAMES, THRESHOLD
                        )
                        
                        pose_name = prediction_result['predicted_class']
                        confidence = prediction_result['confidence']
                        
                        # Draw prediction results
                        cv2.rectangle(img, (10, 10), (400, 80), (0, 0, 0), -1)
                        cv2.putText(img, f"Pose: {pose_name}", (15, 35), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(img, f"Confidence: {confidence:.1%}", (15, 65), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Color border based on confidence
                        color = (0, 255, 0) if prediction_result['accepted'] else (0, 0, 255)
                        cv2.rectangle(img, (10, 10), (400, 80), color, 2)
                else:
                    cv2.putText(img, "No pose detected", (15, 35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Calculate and display FPS
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_time >= 1.0:
                    fps = self.frame_count / (current_time - self.last_time)
                    self.frame_count = 0
                    self.last_time = current_time
                else:
                    fps = 0
                
                if fps > 0:
                    cv2.putText(img, f"FPS: {fps:.1f}", (10, 105), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                return av.VideoFrame.from_ndarray(img, format="bgr24")
        st.write("### Live Webcam Yoga Pose Detection")
        st.write("Click 'START' to begin real-time pose detection")
        
        # Model status
        model = load_yoga_model()
        if model is not None:
            st.success("✅ Yoga pose model loaded successfully!")
        else:
            st.error("❌ Failed to load yoga pose model. Please check if 'yoga_model.keras' exists.")
            st.stop()
        
        
        # Webcam streaming
        webrtc_ctx = webrtc_streamer(
            key="yoga-pose-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=YogaPoseProcessor,
            async_processing=True,
        )
        
        # Status information
        if webrtc_ctx.state.playing:
            st.success("🔴 Live webcam is running")
        else:
            st.info("📹 Click START to begin webcam capture")
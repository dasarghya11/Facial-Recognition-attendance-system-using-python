
# SECTION 1: IMPORT REQUIRED PACKAGES

import face_recognition     # Main face recognition library (showstopper)
import cv2                  # OpenCV for webcam input and video processing
import numpy as np          # For numerical operations (face distance calculations)
import csv                  # For creating and writing attendance CSV files
import os                   # For file operations (not heavily used here)
from datetime import datetime  # For getting current date/time stamps 【105.52, type: source】 【193.84, type: source】 


# SECTION 2: INITIALIZE WEBCAM

# Use default webcam (parameter 0)
video_capture = cv2.VideoCapture(0) 【471.44, type: source】 


# SECTION 3: LOAD KNOWN FACES FROM PHOTOS FOLDER

# Load images from 'photos' folder (must exist with 4 images)
jobs_image = face_recognition.load_image_file("photos/jobs.jpg")
tata_image = face_recognition.load_image_file("photos/tata.jpg")
mona_image = face_recognition.load_image_file("photos/mona.jpg")
tesla_image = face_recognition.load_image_file("photos/tesla.jpg") 【282.24, type: source】 

# SECTION 4: CREATE FACE ENCODINGS (RAW FACE DATA)

# Convert images to numerical face encodings (128-dimensional vectors)
jobs_encoding = face_recognition.face_encodings(jobs_image)[0]
tata_encoding = face_recognition.face_encodings(tata_image)[0]
mona_encoding = face_recognition.face_encodings(mona_image)[0]
tesla_encoding = face_recognition.face_encodings(tesla_image)[0] 【382.08, type: source】 


# SECTION 5: CREATE KNOWN FACES LISTS

# List of all known face encodings for comparison
known_face_encodings = [jobs_encoding, tata_encoding, mona_encoding, tesla_encoding]

# List of corresponding names
known_face_names = ["jobs", "tata", "mona", "tesla"]

# Copy for tracking daily attendance (prevents duplicates)
students = known_face_names.copy() 【382.08, type: source】 


# SECTION 6: FACE DETECTION VARIABLES

# Store face coordinates from webcam frame
face_locations = []
# Store face encodings from webcam frame  
face_encodings = []
# Store recognized names
face_names = []
# Flag for face detection (used in video)
s = True 【382.08, type: source】 


# SECTION 7: CREATE DAILY CSV FILE

# Get current date for filename (YYYY-MM-DD format)
current_date = datetime.now().strftime("%Y-%m-%d")

# Create CSV filename with today's date
csv_filename = f"{current_date}.csv"

# Open file in write+ mode, create writer object
f = open(csv_filename, "w+", newline='')
lnwriter = csv.writer(f) 【471.44, type: source】 

# SECTION 8: MAIN INFINITE LOOP (VIDEO PROCESSING)

while True:
    # Read frame from webcam (returns success flag + frame)
    _, frame = video_capture.read()
    
    # Resize frame to 1/4 size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small_frame = small_frame[:, :, ::-1] 【568.24, type: source】 
    
    # Detect faces in current frame (only if s=True)
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations) 【568.24, type: source】 

  
    # SECTION 9: FACE RECOGNITION & MATCHING
   
    face_names = []
    for face_encoding in face_encodings:
        # Compare webcam face with all known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        # Calculate distance to each known face, find closest match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        # If confidence is high enough, use matched name
        if matches[best_match_index]:
            name = known_face_names[best_match_index] 【666.88, type: source】 
        
        face_names.append(name)
        
     
        # SECTION 10: MARK ATTENDANCE IN CSV (DUPLICATE PREVENTION)
        
        # Only mark if face is known AND not already marked today
        if name in known_face_names and name in students:
            # Remove from students list (prevents multiple entries)
            students.remove(name)
            print(f"Attendance marked for {name}")
            
            # Get current time (HH:MM:SS format)
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Write row to CSV: [name, time]
            lnwriter.writerow([name, current_time])
            # Flush to ensure immediate write
            f.flush() 【766.8, type: source】 

  
    # SECTION 11: DRAW FACE BOXES AND NAMES ON VIDEO
 
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back to original frame size (x4)
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Green rectangle around detected face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Green background for name label
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
        
        # White text with name
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # SECTION 12: DISPLAY VIDEO & EXIT CONDITION
   
    # Show video feed with detections
    cv2.imshow('Attendance System', frame)
    
    # Exit loop when 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 【861.92, type: source】 


# SECTION 13: CLEANUP

# Release webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()
# Close CSV file
f.close() 

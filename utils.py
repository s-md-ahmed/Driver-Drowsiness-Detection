import numpy as np

def calculate_ear(eye_landmarks):
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 landmarks of an eye.
    """
    # Vertical eye landmarks
    v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    
    # Horizontal eye landmarks
    h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    # EAR
    ear = (v1 + v2) / (2.0 * h) if h != 0 else 0
    return ear

def calculate_mar(mouth_landmarks):
    """
    Calculates the Mouth Aspect Ratio (MAR) given inner mouth landmarks.
    Typically, 0=left corner, 4=right corner, 2=top lip, 6=bottom lip
    Wait, MediaPipe face mesh landmarks are different. We will pass exact subsets.
    """
    # Vertical distance
    v1 = np.linalg.norm(np.array(mouth_landmarks[1]) - np.array(mouth_landmarks[7]))
    v2 = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
    v3 = np.linalg.norm(np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[5]))

    # Horizontal distance
    h = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
    
    mar = (v1 + v2 + v3) / (3.0 * h) if h != 0 else 0
    return mar

def estimate_head_pitch(face_landmarks, w, h):
    """
    Estimates the head pitch using specific landmarks.
    Nose tip relative to the eyes and chin.
    """
    # Top of head (10), Chin (152), Nose (1)
    nose = face_landmarks[1]
    top = face_landmarks[10]
    chin = face_landmarks[152]
    
    # Calculate a simple ratio to approximate pitch.
    # When head nods down, nose comes closer to chin in 2D projection.
    dist_nose_chin = np.linalg.norm(np.array(nose) - np.array(chin))
    dist_nose_top = np.linalg.norm(np.array(nose) - np.array(top))
    
    pitch_ratio = dist_nose_chin / dist_nose_top if dist_nose_top != 0 else 1.0
    return pitch_ratio
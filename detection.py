import time

class DrowsinessDetector:
    def __init__(self, ear_threshold=0.25, mar_threshold=0.6, pitch_threshold=0.40, closed_time_threshold=2.0):
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.pitch_threshold = pitch_threshold
        self.closed_time_threshold = closed_time_threshold
        
        self.eyes_closed_start_time = None
        self.yawn_start_time = None
        self.nod_start_time = None
        
        self.is_drowsy = False
        self.fatigue_score = 0
        
    def evaluate(self, ear, mar, pitch, model_eye_state):
        alert_triggers = []
        
        # 1. EYE LOGIC (Hybrid: Math + DL Model)
        # model_eye_state == 0 means "closed" from your .pth model
        eyes_closed = (model_eye_state == 0) or (ear < self.ear_threshold)
        
        if eyes_closed:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = time.time()
            if (time.time() - self.eyes_closed_start_time) >= self.closed_time_threshold:
                self.is_drowsy = True
                alert_triggers.append("PROLONGED EYE CLOSURE")
        else:
            self.eyes_closed_start_time = None
            self.is_drowsy = False

        # 2. YAWN LOGIC
        if mar > self.mar_threshold:
            if self.yawn_start_time is None:
                self.yawn_start_time = time.time()
            elif (time.time() - self.yawn_start_time) > 1.5: 
                alert_triggers.append("YAWNING DETECTED")
        else:
            self.yawn_start_time = None

        # 3. HEAD NOD LOGIC
        if pitch < self.pitch_threshold:
            if self.nod_start_time is None:
                self.nod_start_time = time.time()
            elif (time.time() - self.nod_start_time) > 1.5: 
                alert_triggers.append("HEAD NODDING DETECTED")
        else:
            self.nod_start_time = None

        # 4. FATIGUE SCORE MANAGEMENT
        if alert_triggers:
            self.fatigue_score = min(100, self.fatigue_score + 0.5)
        else:
            self.fatigue_score = max(0, self.fatigue_score - 0.1)
            
        if self.fatigue_score > 50:
            alert_triggers.append("HIGH FATIGUE SCORE")
            
        return alert_triggers
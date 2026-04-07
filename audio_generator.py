import numpy as np
import scipy.io.wavfile as wavfile
import os

def generate_alarm_sound(filename="alarm.wav", duration=2.0, freq=800, sample_rate=44100):
    """
    Generates a generic alarm sound (square wave-ish or pulsed sine) 
    and saves to alarm.wav so we have a reliable local alarm for the browser.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Create a pulsating alarm sound by modulating the amplitude
    # 5 Hz modulation
    modulation = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    
    # 800 Hz carrier frequency
    tone = np.sin(2 * np.pi * freq * t)
    
    # Combine
    audio = modulation * tone
    
    # Normalize to 16-bit integer
    audio_int16 = np.int16(audio * 32767)
    
    filepath = os.path.join(os.path.dirname(__file__), filename)
    wavfile.write(filepath, sample_rate, audio_int16)
    print(f"Generated alarm sound at: {filepath}")

if __name__ == "__main__":
    generate_alarm_sound()

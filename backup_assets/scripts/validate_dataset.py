import os
import librosa
import numpy as np
from tqdm import tqdm

def calculate_snr(y):
    """Accurate Signal-to-Noise Ratio estimation for PCM audio."""
    abs_y = np.abs(y)
    signal_power = np.mean(y**2)
    # Estimate noise from the quietest 10% of the signal
    noise_threshold = np.percentile(abs_y, 10)
    noise_power = np.mean(y[abs_y <= noise_threshold]**2) + 1e-10
    return 10 * np.log10(signal_power / noise_power)

def validate_audio(file_path):
    """
    Validation logic:
    - SNR > 15dB
    - Not clipped
    - Minimum duration 1s
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        
        # 1. Check duration
        duration = len(y) / sr
        if duration < 1.0: return False, "Too short"
        
        # 2. Check clipping
        if np.abs(y).max() >= 0.99: return False, "Clipped"
        
        # 3. Check SNR
        snr = calculate_snr(y)
        if snr < 15: return False, f"Low SNR ({snr:.1f}dB)"
        
        return True, "OK"
    except Exception as e:
        return False, str(e)

def validate_dataset(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    valid_count = 0
    
    print(f"Validating {len(files)} files...")
    for f in tqdm(files):
        path = os.path.join(directory, f)
        is_valid, reason = validate_audio(path)
        if not is_valid:
            print(f"Rejecting {f}: {reason}")
        else:
            valid_count += 1
            
    print(f"Validation complete. Valid: {valid_count}/{len(files)}")

if __name__ == "__main__":
    clean_dir = "/Users/parth/Documents/marathi_tts_work/processed/wavs_cleaned"
    if os.path.exists(clean_dir):
        validate_dataset(clean_dir)
    else:
        print(f"Directory {clean_dir} not found.")

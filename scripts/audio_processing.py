import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import soundfile as sf
from pydub import AudioSegment
from PIL import Image
import traceback
from datetime import datetime
from pathlib import Path

# Define the project base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Set the directory for audio files
AUDIO_DIR = BASE_DIR / 'assets' / 'audios'
FEATURES_CSV = BASE_DIR / 'data' / 'audio_features.csv'

# Expected phrases based on the requirements
PHRASES = ['approve', 'confirm']  # For "Yes, approve" and "Confirm transaction"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def debug_directory_structure():
    """Debug function to check directory structure and files"""
    print("=== DEBUGGING DIRECTORY STRUCTURE ===")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Looking for audio directory: {AUDIO_DIR}")
    print(f"Audio directory exists: {os.path.exists(AUDIO_DIR)}")
    
    if os.path.exists(AUDIO_DIR):
        print(f"Files in {AUDIO_DIR}:")
        try:
            files = os.listdir(AUDIO_DIR)
            if files:
                for i, file in enumerate(files, 1):
                    file_path = os.path.join(AUDIO_DIR, file)
                    file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
                    print(f"  {i}. {file} ({file_size} bytes)")
            else:
                print("  No files found in audio directory")
        except Exception as e:
            print(f"  Error reading directory: {e}")
    else:
        print("Creating audio directory...")
        try:
            os.makedirs(AUDIO_DIR, exist_ok=True)
            print(f"Created directory: {AUDIO_DIR}")
        except Exception as e:
            print(f"Error creating directory: {e}")
    
    print()

def load_audio_robust(file_path):
    """
    Robust audio loading function that tries multiple methods
    Returns (audio_array, sample_rate) or (None, None) if all methods fail
    """
    print(f"    Attempting to load: {os.path.basename(file_path)}")
    
    # Check if file exists and has size
    if not os.path.exists(file_path):
        print(f"      File does not exist: {file_path}")
        return None, None
    
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        print(f"      File is empty (0 bytes): {file_path}")
        return None, None
    
    print(f"      File size: {file_size} bytes")
    
    # Method 1: Try librosa with default sample rate
    try:
        print("      Trying librosa (default)...")
        y, sr = librosa.load(file_path)
        if len(y) > 0:
            print(f"      SUCCESS with librosa: sr={sr}, duration={len(y)/sr:.2f}s, samples={len(y)}")
            return y, sr
        else:
            print("      librosa returned empty audio array")
    except Exception as e:
        print(f"      librosa failed: {str(e)[:100]}...")
    
    # Method 2: Try librosa with native sample rate
    try:
        print("      Trying librosa (native sr)...")
        y, sr = librosa.load(file_path, sr=None)
        if len(y) > 0:
            print(f"      SUCCESS with librosa (native): sr={sr}, duration={len(y)/sr:.2f}s, samples={len(y)}")
            return y, sr
        else:
            print("      librosa (native) returned empty audio array")
    except Exception as e:
        print(f"      librosa (native) failed: {str(e)[:100]}...")
    
    # Method 3: Try soundfile
    try:
        print("      Trying soundfile...")
        y, sr = sf.read(file_path)
        if len(y) > 0:
            print(f"      SUCCESS with soundfile: sr={sr}, duration={len(y)/sr:.2f}s, samples={len(y)}")
            return y, sr
        else:
            print("      soundfile returned empty audio array")
    except Exception as e:
        print(f"      soundfile failed: {str(e)[:100]}...")
    
    # Method 4: Try pydub conversion
    try:
        print("      Trying pydub conversion...")
        
        # Detect format and load with pydub
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.mp3':
            audio = AudioSegment.from_mp3(file_path)
        elif ext == '.m4a':
            audio = AudioSegment.from_file(file_path, format="m4a")
        elif ext == '.mp4':
            audio = AudioSegment.from_file(file_path, format="mp4")
        elif ext == '.wav':
            audio = AudioSegment.from_wav(file_path)
        else:
            raise ValueError(f"Unsupported format: {ext}")
        
        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # Normalize to [-1, 1] range
        samples = samples / (2**(audio.sample_width * 8 - 1))
        
        sr = audio.frame_rate
        
        if len(samples) > 0:
            print(f"      SUCCESS with pydub: sr={sr}, duration={len(samples)/sr:.2f}s, samples={len(samples)}")
            return samples, sr
        else:
            print("      pydub returned empty audio array")
        
    except Exception as e:
        print(f"      pydub failed: {str(e)[:100]}...")
    
    print("      ALL METHODS FAILED!")
    return None, None

def get_audio_paths():
    """
    Get organized audio paths by member and phrase
    Expected file naming: member_phrase.extension
    e.g., afsa_approve.m4a, amandine_confirm.mp3
    """
    audio_data = {}
    
    if not os.path.exists(AUDIO_DIR):
        print(f"Warning: Audio directory '{AUDIO_DIR}' does not exist!")
        return audio_data
    
    files = os.listdir(AUDIO_DIR)
    print(f"Scanning {len(files)} files in {AUDIO_DIR}...")
    
    for fname in files:
        print(f"  Checking file: {fname}")
        if fname.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4')):
            base = os.path.splitext(fname)[0]
            print(f"    Base name: {base}")
            
            # Handle different naming patterns
            if '_approve' in base.lower():
                member = base.lower().replace('_approve', '').replace(' ', '_')
                phrase = 'approve'
                print(f"    Identified: member='{member}', phrase='{phrase}'")
            elif '_confirm' in base.lower():
                member = base.lower().replace('_confirm', '').replace(' ', '_')
                phrase = 'confirm'
                print(f"    Identified: member='{member}', phrase='{phrase}'")
            else:
                print(f"    Skipping file {fname}: Could not identify phrase (approve/confirm)")
                continue
            
            audio_path = os.path.join(AUDIO_DIR, fname)
            
            # Initialize member dictionary if not exists
            if member not in audio_data:
                audio_data[member] = {}
            
            audio_data[member][phrase] = audio_path
            print(f"    Stored: {member} - {phrase} ({fname})")
        else:
            print(f"    Skipping non-audio file: {fname}")
    
    return audio_data

def display_audio_samples(audio_file_path, label, show_plots=True):
    """
    Display waveform and spectrogram for audio data using robust loading
    """
    try:
        # Use robust loading method
        y, sr = load_audio_robust(audio_file_path)
        
        if y is None or sr is None:
            print(f"    Error: Could not load audio file {audio_file_path}")
            return None, None
        
        if show_plots:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Display the waveform
            librosa.display.waveshow(y, sr=sr, ax=ax1)
            ax1.set_title(f'Waveform of {label}')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # Display the spectrogram
            D = librosa.stft(y)
            S_db = librosa.amplitude_to_db(abs(D), ref=np.max)
            img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr, ax=ax2)
            ax2.set_title(f'Spectrogram of {label}')
            fig.colorbar(img, ax=ax2, label='Amplitude (dB)')
            
            plt.tight_layout()
            plt.show()
        
        return y, sr
        
    except Exception as e:
        print(f"Error displaying audio for {label}:")
        print(f"    Error: {e}")
        print(f"    Traceback: {traceback.format_exc()}")
        return None, None

def apply_pitch_shift(y, sr, n_steps=4):
    """Apply pitch shifting (shift by n semitones)"""
    try:
        result = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)
        return result
    except Exception as e:
        print(f"Error applying pitch shift: {e}")
        return y

def apply_time_stretch(y, rate=1.2):
    """Apply time stretching (stretch by rate factor)"""
    try:
        result = librosa.effects.time_stretch(y=y, rate=rate)
        return result
    except Exception as e:
        print(f"Error applying time stretch: {e}")
        return y

def add_background_noise(y, noise_factor=0.005):
    """Add random background noise"""
    try:
        noise = np.random.randn(len(y))
        result = y + noise_factor * noise
        return result
    except Exception as e:
        print(f"Error adding background noise: {e}")
        return y

def extract_features(y, sr):
    """
    Extract audio features: MFCCs, Spectral Roll-off, Energy
    Returns summary statistics for each feature
    """
    try:
        print(f"      Extracting features from audio with {len(y)} samples at {sr}Hz...")

        # Extract MFCCs (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Extract Spectral Roll-off (85% roll-off)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
        # Extract Energy (Root Mean Square)
        energy = librosa.feature.rms(y=y)

        # Compute summary statistics for each feature
        features = {
            'mfccs_mean': np.mean(mfccs),
            'mfccs_std': np.std(mfccs),
            'mfccs_min': np.min(mfccs),
            'mfccs_max': np.max(mfccs),
            'rolloff_mean': np.mean(spectral_rolloff),
            'rolloff_std': np.std(spectral_rolloff),
            'rolloff_min': np.min(spectral_rolloff),
            'rolloff_max': np.max(spectral_rolloff),
            'energy_mean': np.mean(energy),
            'energy_std': np.std(energy),
            'energy_min': np.min(energy),
            'energy_max': np.max(energy),
        }
        return features

    except Exception as e:
        print(f"Error extracting features: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return {}

def process_audio_augmentations(y, sr, member, phrase, verbose=True, show_plots=False):
    """
    Process audio with augmentations and extract features
    Returns list of feature dictionaries
    """
    features_list = []

    # Process original audio
    print(f"    Processing original audio...")
    features = extract_features(y, sr)

    if features:  # Only add if features were successfully extracted
        features_list.append({
            'member': member,
            'phrase': phrase,
            'augmentation': 'original',
            **features
        })
        if verbose:
            print(f"      ✓ Original features extracted successfully")
    else:
        print(f"      ✗ Failed to extract original features")

    # Apply and process augmentations
    augmentations = [
        ('pitch_shift', lambda: apply_pitch_shift(y, sr, n_steps=4)),
        ('time_stretch', lambda: apply_time_stretch(y, rate=1.2)),
        ('background_noise', lambda: add_background_noise(y, noise_factor=0.005))
    ]

    for aug_name, aug_func in augmentations:
        if verbose:
            print(f"    Processing {aug_name} augmentation...")
        try:
            y_aug = aug_func()

            if y_aug is not None and len(y_aug) > 0:
                # Extract features for augmented audio
                features_aug = extract_features(y_aug, sr)

                if features_aug:  # Only add if features were successfully extracted
                    features_list.append({
                        'member': member,
                        'phrase': phrase,
                        'augmentation': aug_name,
                        **features_aug
                    })
                    if verbose:
                        print(f"      ✓ {aug_name} features extracted successfully")
                else:
                    print(f"      ✗ Failed to extract {aug_name} features")
            else:
                print(f"      ✗ {aug_name} augmentation returned empty/invalid audio")

        except Exception as e:
            print(f"      ✗ Error processing {aug_name}: {e}")
            print(f"      Traceback: {traceback.format_exc()}")

    if verbose:
        print(f"    Total features extracted: {len(features_list)}")
    return features_list

def save_features_to_csv(all_features, csv_path):
    """Save features to CSV with error handling and verification"""
    try:
        print(f"Attempting to save {len(all_features)} feature records to {csv_path}...")
        
        if not all_features:
            print("No features to save!")
            return False
        
        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        print(f"DataFrame created with shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        
        # Save to CSV
        abs_csv_path = os.path.abspath(csv_path)
        features_df.to_csv(abs_csv_path, index=False)
        
        # Verify the file was created
        if os.path.exists(abs_csv_path):
            file_size = os.path.getsize(abs_csv_path)
            print(f"✓ CSV file saved successfully!")
            print(f"  File path: {abs_csv_path}")
            print(f"  File size: {file_size} bytes")
            
            # Try to read it back to verify
            test_df = pd.read_csv(abs_csv_path)
            print(f"  Verification read: {test_df.shape[0]} rows, {test_df.shape[1]} columns")
            return True
        else:
            print("✗ CSV file was not created!")
            return False
            
    except Exception as e:
        print(f"Error saving CSV: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False

def create_test_audio_files():
    """Create sample audio files for testing if none exist"""
    import scipy.io.wavfile as wavfile
    
    print("=== CREATING TEST AUDIO FILES ===")
    os.makedirs(AUDIO_DIR, exist_ok=True)
    
    # Create simple test audio (sine waves with different frequencies)
    sample_rate = 22050
    duration = 2  # seconds
    t = np.linspace(0, duration, sample_rate * duration)
    
    test_members = ['alice', 'bob']
    frequencies = {'approve': 440, 'confirm': 660}  # Different tones for different phrases
    
    for member in test_members:
        for phrase, freq in frequencies.items():
            filename = f"{member}_{phrase}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            if not os.path.exists(filepath):
                # Create sine wave
                audio_data = np.sin(2 * np.pi * freq * t) * 0.3  # Lower amplitude
                # Add some variation to make it more realistic
                audio_data += np.random.normal(0, 0.01, len(audio_data))
                
                wavfile.write(filepath, sample_rate, audio_data.astype(np.float32))
                print(f"  Created: {filename}")
            else:
                print(f"  Exists: {filename}")
    print()

def main():
    """Main processing function with enhanced debugging"""
    print("=== Sound Data Collection and Processing ===")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Processing audio samples for phrases: 'Yes, approve' and 'Confirm transaction'")
    print()
    
    # Create test files if no audio files exist
    if not os.path.exists(AUDIO_DIR) or len([f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.m4a', '.mp4'))]) == 0:
        print("No audio files found. Creating test audio files...")
        create_test_audio_files()
    
    # Debug directory structure first
    debug_directory_structure()
    
    # Get organized audio paths
    print("=== SCANNING FOR AUDIO FILES ===")
    audio_data = get_audio_paths()
    
    if not audio_data:
        print("\n No audio files found!")
        print("Expected file structure:")
        print("  assets/audios/member1_approve.wav")
        print("  assets/audios/member1_confirm.wav") 
        print("  assets/audios/member2_approve.wav")
        print("  assets/audios/member2_confirm.wav")
        print("  ...")
        print("\nMake sure your audio files follow the naming convention: membername_phrase.extension")
        return
    
    print(f"\n✓ Found audio files for {len(audio_data)} members: {list(audio_data.keys())}")
    print()
    
    # Store all features
    all_features = []
    processing_stats = {
        'files_processed': 0,
        'files_failed': 0,
        'features_extracted': 0
    }
    
    # Process each member's audio files
    for member, audio_files in audio_data.items():
        print(f"=== PROCESSING MEMBER: {member} ===")
        print(f"Available phrases: {list(audio_files.keys())}")
        
        # Process each phrase
        for phrase in PHRASES:
            if phrase not in audio_files:
                print(f"  ⚠️  Warning: No '{phrase}' file found for {member}")
                processing_stats['files_failed'] += 1
                continue
                
            audio_file = audio_files[phrase]
            phrase_label = "Yes, approve" if phrase == "approve" else "Confirm transaction"
            print(f"\n  Processing phrase: '{phrase_label}' from file: {os.path.basename(audio_file)}")
            
            # Load and display original audio using robust method
            y, sr = display_audio_samples(audio_file, f'{member} - {phrase_label}', show_plots=False)
            
            if y is None or sr is None:
                print(f"    ❌ Skipping {member}-{phrase}: Could not load audio")
                processing_stats['files_failed'] += 1
                continue
            
            # Process augmentations and extract features
            phrase_features = process_audio_augmentations(y, sr, member, phrase)
            
            if phrase_features:
                all_features.extend(phrase_features)
                processing_stats['files_processed'] += 1
                processing_stats['features_extracted'] += len(phrase_features)
                print(f"    ✓ Successfully processed {len(phrase_features)} samples (original + augmentations)")
            else:
                print(f"    ❌ No features extracted for {member}-{phrase}")
                processing_stats['files_failed'] += 1
        
        print()
    
    print("=== PROCESSING SUMMARY ===")
    print(f"Files processed successfully: {processing_stats['files_processed']}")
    print(f"Files failed: {processing_stats['files_failed']}")
    print(f"Total feature records: {processing_stats['features_extracted']}")
    
    # Save features to CSV 
    if all_features:
        print(f"\n=== SAVING FEATURES TO CSV ===")
        success = save_features_to_csv(all_features, FEATURES_CSV)
        
        if success:
            # Create DataFrame for analysis
            features_df = pd.DataFrame(all_features)
            
            print("\n=== FINAL RESULTS ===")
            print(f"✓ Features saved to: {os.path.abspath(FEATURES_CSV)}")
            print(f"Total samples processed: {len(features_df)}")
            print(f"Members: {features_df['member'].nunique()}")
            print(f"Phrases: {features_df['phrase'].unique().tolist()}")
            print(f"Augmentations: {features_df['augmentation'].unique().tolist()}")
            print()
            print("DataFrame info:")
            print(f"Shape: {features_df.shape}")
            print(f"Columns: {list(features_df.columns)}")
            
            # Display sample of the data
            print("\nFirst few rows:")
            print(features_df[['member', 'phrase', 'augmentation']].head(10))
            
            # Show success rate
            total_expected = len(audio_data) * len(PHRASES)
            success_rate = processing_stats['files_processed'] / total_expected * 100 if total_expected > 0 else 0
            print(f"\nProcessing success rate: {processing_stats['files_processed']}/{total_expected} files ({success_rate:.1f}%)")
        else:
            print("Failed to save CSV file!")
        
    else:
        print("No features were extracted. Please check your audio files and try again.")
        print("\nTroubleshooting checklist:")
        print("1. Ensure audio files exist in the 'assets/audios' directory")
        print("2. Check file naming convention: membername_approve.ext or membername_confirm.ext")
        print("3. Verify audio files are not corrupted and have valid audio data")
        print("4. Check file permissions and disk space")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
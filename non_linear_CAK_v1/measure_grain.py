import librosa
import librosa.feature
import numpy as np
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_audio_file(filepath):
    """
    Analyze a single audio file and extract various features including grain (spectral flatness)
    """
    try:
        # Load audio
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Compute STFT
        S = np.abs(librosa.stft(y))
        
        # Compute spectral features
        spectral_flatness = librosa.feature.spectral_flatness(S=S).mean()
        spectral_centroid = librosa.feature.spectral_centroid(S=S).mean()
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y).mean()
        
        # Compute RMS energy (as a proxy for loudness)
        rms = librosa.feature.rms(S=S).mean()
        
        # Duration
        duration = len(y) / sr
        
        return {
            'grain_density': float(spectral_flatness),  # This is our main grain measure
            'spectral_centroid': float(spectral_centroid),
            'zero_crossing_rate': float(zero_crossing_rate),
            'rms_energy': float(rms),
            'duration': float(duration),
            'sample_rate': int(sr)
        }
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def create_metadata_for_directory(audio_dir, output_path='metadata.json'):
    """
    Process all audio files in a directory and create a metadata JSON file
    """
    metadata = {}
    audio_extensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a']
    
    # Get all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(Path(audio_dir).glob(f'**/*{ext}'))
    
    print(f"Found {len(audio_files)} audio files to process...")
    
    for i, audio_path in enumerate(audio_files):
        print(f"Processing {i+1}/{len(audio_files)}: {audio_path.name}")
        
        # Analyze the file
        features = analyze_audio_file(str(audio_path))
        
        if features:
            # Create entry in metadata
            relative_path = str(audio_path.relative_to(audio_dir))
            
            metadata[relative_path] = {
                'filename': audio_path.name,
                'path': relative_path,
                'features': features,
                'tags': extract_tags_from_filename(audio_path.name)
            }
            
            # Print grain value for this file
            print(f"  → Grain density: {features['grain_density']:.4f}")
    
    # Save metadata
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nMetadata saved to {output_path}")
    
    # Print summary statistics
    print_grain_statistics(metadata)
    
    return metadata

def extract_tags_from_filename(filename):
    """
    Extract semantic tags from filename
    Example: drone_evolving_lowfreq_stereomodulation_01.wav
    """
    parts = filename.replace('.wav', '').split('_')
    
    tags = {}
    
    # Try to identify common patterns in your filenames
    if 'drone' in parts:
        tags['sound_type'] = 'drone'
    if 'evolving' in parts:
        tags['quality'] = 'evolving'
    elif 'static' in parts:
        tags['quality'] = 'static'
    
    # Frequency range
    if 'lowfreq' in parts:
        tags['frequency_range'] = 'low'
    elif 'midfreq' in parts:
        tags['frequency_range'] = 'mid'
    elif 'highfreq' in parts:
        tags['frequency_range'] = 'high'
    elif 'wideband' in parts:
        tags['frequency_range'] = 'wide'
    
    # Stereo characteristics
    if 'mono' in parts:
        tags['stereo'] = 'mono'
    elif 'stereomodulation' in parts:
        tags['stereo'] = 'stereo_modulated'
    elif 'midwidth' in parts:
        tags['stereo'] = 'mid_width'
    
    return tags

def print_grain_statistics(metadata):
    """
    Print statistics about grain distribution in your dataset
    """
    grain_values = [entry['features']['grain_density'] 
                   for entry in metadata.values() 
                   if 'features' in entry]
    
    if grain_values:
        print("\n=== Grain Distribution Statistics ===")
        print(f"Number of files: {len(grain_values)}")
        print(f"Min grain: {min(grain_values):.4f}")
        print(f"Max grain: {max(grain_values):.4f}")
        print(f"Mean grain: {np.mean(grain_values):.4f}")
        print(f"Std grain: {np.std(grain_values):.4f}")
        
        # Show distribution
        print("\nGrain distribution:")
        hist, bins = np.histogram(grain_values, bins=5)
        for i in range(len(hist)):
            print(f"  {bins[i]:.3f} - {bins[i+1]:.3f}: {'█' * int(hist[i]*50/max(hist))} ({hist[i]} files)")

# Example usage:
if __name__ == "__main__":
    # Run from current directory - will search all subdirectories
    audio_directory = "."  # Current directory (where Drone and Harmonic Progression folders are)
    
    # Create metadata
    metadata = create_metadata_for_directory(audio_directory)
    
    # Optional: Create a simplified grain-only CSV for quick reference
    import csv
    with open('grain_values.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'grain_density'])
        for path, data in metadata.items():
            if 'features' in data:
                writer.writerow([data['filename'], data['features']['grain_density']])
    
    print("\nAlso saved simplified grain values to grain_values.csv")
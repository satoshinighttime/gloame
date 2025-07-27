import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def create_spectrogram_dataset(audio_dir, output_dir='spectrograms', 
                             spec_type='mel', n_mels=128, 
                             hop_length=512, n_fft=2048,
                             target_length=256):  # frames
    """
    Convert audio files to spectrograms for GAN training
    
    Parameters:
    - audio_dir: Directory containing audio files
    - output_dir: Where to save spectrogram images
    - spec_type: 'mel' or 'stft' 
    - n_mels: Number of mel bands (if using mel spectrograms)
    - hop_length: Hop length for STFT
    - n_fft: FFT window size
    - target_length: Target number of time frames (will pad/crop to this)
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories to maintain structure
    Path(f"{output_dir}/images").mkdir(exist_ok=True)
    Path(f"{output_dir}/arrays").mkdir(exist_ok=True)
    
    # Load metadata to get grain values
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Statistics tracking
    stats = {
        'processed': 0,
        'failed': 0,
        'spec_shape': None,
        'value_range': {'min': float('inf'), 'max': float('-inf')}
    }
    
    print(f"Converting audio to {spec_type} spectrograms...")
    print(f"Settings: n_mels={n_mels}, hop_length={hop_length}, n_fft={n_fft}")
    
    # Process each file
    for rel_path, file_info in metadata.items():
        try:
            audio_path = os.path.join(audio_dir, rel_path)
            
            # Load audio
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Create spectrogram
            if spec_type == 'mel':
                # Mel spectrogram
                S = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=n_mels, 
                    hop_length=hop_length, n_fft=n_fft
                )
                S_dB = librosa.power_to_db(S, ref=np.max)
            else:
                # Regular STFT spectrogram
                S_complex = librosa.stft(y, hop_length=hop_length, n_fft=n_fft)
                S = np.abs(S_complex)
                S_dB = librosa.amplitude_to_db(S, ref=np.max)
            
            # Pad or crop to target length
            if S_dB.shape[1] < target_length:
                # Pad with minimum values
                pad_width = target_length - S_dB.shape[1]
                S_dB = np.pad(S_dB, ((0, 0), (0, pad_width)), 
                            mode='constant', constant_values=S_dB.min())
            else:
                # Crop to target length
                S_dB = S_dB[:, :target_length]
            
            # Update statistics
            stats['spec_shape'] = S_dB.shape
            stats['value_range']['min'] = min(stats['value_range']['min'], S_dB.min())
            stats['value_range']['max'] = max(stats['value_range']['max'], S_dB.max())
            
            # Save as image (for visualization)
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, 
                                    x_axis='time', y_axis='mel' if spec_type == 'mel' else 'hz')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f"{file_info['filename']} (grain={file_info['features']['grain_density']:.6f})")
            plt.tight_layout()
            
            # Save image
            img_filename = rel_path.replace('/', '_').replace('.wav', '.png')
            plt.savefig(f"{output_dir}/images/{img_filename}")
            plt.close()
            
            # Save as numpy array (for training)
            array_filename = rel_path.replace('/', '_').replace('.wav', '.npy')
            np.save(f"{output_dir}/arrays/{array_filename}", S_dB)
            
            # Update metadata with spectrogram info
            metadata[rel_path]['spectrogram'] = {
                'array_path': f"arrays/{array_filename}",
                'image_path': f"images/{img_filename}",
                'shape': list(S_dB.shape),
                'type': spec_type,
                'normalized': False  # We'll normalize later in training
            }
            
            stats['processed'] += 1
            
            if stats['processed'] % 10 == 0:
                print(f"Processed {stats['processed']} files...")
                
        except Exception as e:
            print(f"Error processing {rel_path}: {e}")
            stats['failed'] += 1
    
    # Save updated metadata
    with open(f"{output_dir}/metadata_with_specs.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save preprocessing configuration
    config = {
        'spec_type': spec_type,
        'n_mels': n_mels,
        'hop_length': hop_length,
        'n_fft': n_fft,
        'target_length': target_length,
        'spec_shape': list(stats['spec_shape']) if stats['spec_shape'] else None,
        'value_range': {
            'min': float(stats['value_range']['min']),
            'max': float(stats['value_range']['max'])
        },
        'normalization': 'db_scale'
    }
    
    with open(f"{output_dir}/preprocessing_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n=== Conversion Complete ===")
    print(f"Processed: {stats['processed']} files")
    print(f"Failed: {stats['failed']} files")
    print(f"Spectrogram shape: {stats['spec_shape']}")
    print(f"Value range: [{stats['value_range']['min']:.2f}, {stats['value_range']['max']:.2f}] dB")
    print(f"\nSaved to: {output_dir}/")
    
    return metadata, stats

def create_training_pairs(spec_dir='spectrograms', output_file='training_pairs.json'):
    """
    Create training pairs of (spectrogram_path, grain_value) for the GAN
    """
    # Load metadata with spectrogram info
    with open(f"{spec_dir}/metadata_with_specs.json", 'r') as f:
        metadata = json.load(f)
    
    training_pairs = []
    
    for rel_path, info in metadata.items():
        if 'spectrogram' in info:
            pair = {
                'spec_path': os.path.join(spec_dir, info['spectrogram']['array_path']),
                'grain_value': info['features']['grain_density'],
                'filename': info['filename'],
                'tags': info.get('tags', {})
            }
            training_pairs.append(pair)
    
    # Sort by grain value for easier visualization
    training_pairs.sort(key=lambda x: x['grain_value'])
    
    # Save training pairs
    with open(output_file, 'w') as f:
        json.dump(training_pairs, f, indent=2)
    
    print(f"\nCreated {len(training_pairs)} training pairs")
    print(f"Saved to: {output_file}")
    
    # Print grain distribution
    grain_values = [p['grain_value'] for p in training_pairs]
    print("\nGrain value distribution in training set:")
    print(f"Min: {min(grain_values):.6f}")
    print(f"Max: {max(grain_values):.6f}")
    print(f"Mean: {np.mean(grain_values):.6f}")
    print(f"Median: {np.median(grain_values):.6f}")
    
    return training_pairs

def visualize_grain_examples(spec_dir='spectrograms', n_examples=6):
    """
    Visualize spectrograms across the grain spectrum
    """
    with open('training_pairs.json', 'r') as f:
        pairs = json.load(f)
    
    # Select examples across grain range
    indices = np.linspace(0, len(pairs)-1, n_examples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, i in enumerate(indices):
        spec = np.load(pairs[i]['spec_path'])
        grain = pairs[i]['grain_value']
        
        im = axes[idx].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
        axes[idx].set_title(f"Grain: {grain:.6f}")
        axes[idx].set_xlabel('Time')
        axes[idx].set_ylabel('Frequency')
    
    plt.suptitle('Spectrograms Across Grain Spectrum (Low to High)')
    plt.tight_layout()
    plt.savefig('grain_spectrum_examples.png', dpi=150)
    plt.show()
    
    print("Saved visualization to: grain_spectrum_examples.png")

# Example usage
if __name__ == "__main__":
    # STEP 1: Convert audio to spectrograms
    audio_directory = "."  # Current directory with Drone and Harmonic Progression folders
    
    # You can experiment with these parameters
    metadata, stats = create_spectrogram_dataset(
        audio_dir=audio_directory,
        output_dir='spectrograms',
        spec_type='mel',        # 'mel' or 'stft'
        n_mels=128,            # Height of mel spectrogram
        hop_length=512,        # Affects time resolution
        n_fft=2048,           # Affects frequency resolution
        target_length=256      # Width (time frames) - all specs will be this size
    )
    
    # STEP 2: Create training pairs
    training_pairs = create_training_pairs()
    
    # STEP 3: Visualize some examples
    visualize_grain_examples()
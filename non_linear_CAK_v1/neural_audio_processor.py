import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import queue
from pathlib import Path

# ============= TRUE CAK MODEL ARCHITECTURE =============
class PatternDetector(nn.Module):
    """Learns to detect patterns in spectrograms - the CORE of CAK"""
    
    def __init__(self):
        super(PatternDetector, self).__init__()
        # Multi-scale pattern detection
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5, 5), padding=(2, 2))
        self.conv3 = nn.Conv2d(16, 8, kernel_size=(3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(8, 1, kernel_size=(1, 1))
        
        # Initialize with small weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Detect patterns at multiple scales
        h = torch.nn.functional.leaky_relu(self.conv1(x), 0.2)
        h = torch.nn.functional.leaky_relu(self.conv2(h), 0.2)
        h = torch.nn.functional.leaky_relu(self.conv3(h), 0.2)
        patterns = self.conv4(h)  # No activation - can be positive or negative
        return patterns

class TrueCAKLayer(nn.Module):
    """
    TRUE CAK implementation - exactly Austin's formula:
    output = x + (detected_patterns × grain_value)
    """
    
    def __init__(self, temperature=10.0):
        super(TrueCAKLayer, self).__init__()
        self.pattern_detector = PatternDetector()
        self.temperature = temperature
        
        # Single scale parameter - that's it!
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def soft_gate(self, grain_value):
        """Soft gating for smooth transitions"""
        return torch.sigmoid(torch.abs(grain_value) * self.temperature)
    
    def forward(self, x, grain_value):
        """
        PURE AUSTIN FORMULA:
        1. Detect patterns in THIS input
        2. Scale by grain value
        3. Add to input
        """
        # Step 1: Detect patterns in the input
        patterns = self.pattern_detector(x)
        
        # Step 2: Scale patterns by grain value
        grain_v = grain_value.view(-1, 1, 1, 1)
        gate = self.soft_gate(grain_value).view(-1, 1, 1, 1)
        
        # Step 3: The Austin formula
        scaled_patterns = patterns * grain_v * gate * self.scale
        output = x + scaled_patterns
        
        return output, patterns

class PureNeuralGrainProcessor(nn.Module):
    """
    Simplified processor - NO residual connections!
    Just progressive CAK application
    """
    
    def __init__(self, n_freq=128):
        super(PureNeuralGrainProcessor, self).__init__()
        
        # Three CAK layers with increasing sensitivity
        self.cak1 = TrueCAKLayer(temperature=5.0)
        self.cak2 = TrueCAKLayer(temperature=10.0)
        self.cak3 = TrueCAKLayer(temperature=15.0)
        
        # Simple output normalization
        self.output_norm = nn.BatchNorm2d(1)
    
    def forward(self, x, grain_value):
        """
        Progressive CAK application - each layer refines the previous
        NO additions, NO residuals
        """
        # Progressive refinement
        x1, p1 = self.cak1(x, grain_value)
        x2, p2 = self.cak2(x1, grain_value * 0.7)  # Slightly reduced
        x3, p3 = self.cak3(x2, grain_value * 0.5)  # Further reduced
        
        # Normalize output
        output = self.output_norm(x3)
        
        # Return final output and all patterns for visualization
        all_patterns = torch.cat([p1, p2, p3], dim=1)
        
        return output, all_patterns

# ============= AUDIO PROCESSOR =============
class AudioProcessor:
    """Handles audio processing with the TRUE CAK neural grain model"""
    
    def __init__(self, model_path='true_cak_output/final_true_cak.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the TRUE CAK model
        self.model = PureNeuralGrainProcessor().to(self.device)
        
        # Try to load checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['processor'])
            print("Successfully loaded TRUE CAK model!")
        except Exception as e:
            print(f"Warning: Could not load model from {model_path}")
            print(f"Error: {e}")
            print("Using untrained model...")
        
        self.model.eval()
        
        # Audio processing parameters
        self.n_fft = 2048
        self.hop_length = 512
        self.n_mels = 128
        self.sr = None
        
    def process_audio(self, audio_path, grain_amount, progress_callback=None):
        """Process an audio file with the specified grain amount"""
        
        # Load audio
        if progress_callback:
            progress_callback(0.1, "Loading audio...")
        y, sr = librosa.load(audio_path, sr=None, mono=True)
        self.sr = sr
        
        # Split audio into chunks with overlap for smoother processing
        chunk_length = sr * 3  # 3-second chunks
        overlap = sr // 2  # 0.5 second overlap
        hop = chunk_length - overlap
        
        chunks = []
        positions = []
        
        for i in range(0, len(y) - overlap, hop):
            chunk = y[i:i + chunk_length]
            if len(chunk) < chunk_length:
                # Pad the last chunk
                chunk = np.pad(chunk, (0, chunk_length - len(chunk)), mode='constant')
            chunks.append(chunk)
            positions.append(i)
        
        processed_chunks = []
        total_chunks = len(chunks)
        
        for idx, chunk in enumerate(chunks):
            if progress_callback:
                progress = 0.1 + (0.7 * idx / total_chunks)
                progress_callback(progress, f"Processing chunk {idx+1}/{total_chunks}")
            
            # Convert to spectrogram
            S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=self.n_fft,
                                              hop_length=self.hop_length, n_mels=self.n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Normalize
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
            S_norm = S_norm * 2 - 1
            
            # Ensure correct shape for processing
            target_length = 256
            if S_norm.shape[1] < target_length:
                S_norm = np.pad(S_norm, ((0, 0), (0, target_length - S_norm.shape[1])), mode='constant')
            else:
                S_norm = S_norm[:, :target_length]
            
            # Process with TRUE CAK neural network
            with torch.no_grad():
                S_tensor = torch.FloatTensor(S_norm).unsqueeze(0).unsqueeze(0).to(self.device)
                grain_tensor = torch.FloatTensor([[grain_amount]]).to(self.device)
                
                processed_S, patterns = self.model(S_tensor, grain_tensor)
                processed_S = processed_S.cpu().numpy()[0, 0]
            
            # Denormalize
            processed_S = (processed_S + 1) / 2
            processed_S_db = processed_S * (S_db.max() - S_db.min()) + S_db.min()
            
            # Crop back to original spectrogram width if needed
            if processed_S_db.shape[1] > S_db.shape[1]:
                processed_S_db = processed_S_db[:, :S_db.shape[1]]
            
            # Convert back to linear scale
            processed_S_linear = librosa.db_to_power(processed_S_db, ref=1.0)
            
            # Inverse mel-spectrogram using Griffin-Lim
            processed_chunk = librosa.feature.inverse.mel_to_audio(
                processed_S_linear, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_iter=32  # More iterations for better quality
            )
            
            processed_chunks.append(processed_chunk[:chunk_length])
        
        # Combine chunks with crossfading
        if progress_callback:
            progress_callback(0.9, "Combining audio...")
        
        # Initialize output array
        output_length = positions[-1] + chunk_length if positions else len(y)
        processed_audio = np.zeros(output_length)
        
        # Crossfade overlapping chunks
        for i, (chunk, pos) in enumerate(zip(processed_chunks, positions)):
            if i == 0:
                # First chunk - no fade in
                processed_audio[pos:pos + len(chunk)] = chunk
            else:
                # Create crossfade for overlap region
                fade_length = overlap
                fade_in = np.linspace(0, 1, fade_length)
                fade_out = np.linspace(1, 0, fade_length)
                
                # Apply crossfade
                start_pos = pos
                end_pos = min(pos + len(chunk), len(processed_audio))
                chunk_to_add = chunk[:end_pos - start_pos]
                
                if start_pos + fade_length <= len(processed_audio):
                    # Crossfade the overlap
                    processed_audio[start_pos:start_pos + fade_length] *= fade_out
                    processed_audio[start_pos:start_pos + fade_length] += chunk_to_add[:fade_length] * fade_in
                    # Add the rest
                    if len(chunk_to_add) > fade_length:
                        processed_audio[start_pos + fade_length:end_pos] = chunk_to_add[fade_length:]
                else:
                    # Just add what fits
                    processed_audio[start_pos:end_pos] = chunk_to_add
        
        # Trim to original length
        processed_audio = processed_audio[:len(y)]
        
        # ============= LOUDNESS MATCHING =============
        # Match RMS energy to preserve perceived loudness
        original_rms = np.sqrt(np.mean(y**2))
        processed_rms = np.sqrt(np.mean(processed_audio**2))
        
        # Avoid division by zero
        if processed_rms > 0:
            rms_ratio = original_rms / processed_rms
            processed_audio = processed_audio * rms_ratio
        
        # Apply gentle limiting to prevent clipping
        max_val = np.max(np.abs(processed_audio))
        if max_val > 0.95:
            processed_audio = processed_audio * 0.95 / max_val
        
        if progress_callback:
            progress_callback(1.0, "Complete!")
        
        return processed_audio, sr

# ============= GUI APPLICATION =============
class NeuralAudioProcessorGUI:
    """GUI for the TRUE CAK Neural Audio Processor"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Grain Processor - TRUE CAK")
        self.root.geometry("900x700")
        
        # Initialize processor
        self.processor = AudioProcessor()
        
        # Audio data
        self.original_audio = None
        self.processed_audio = None
        self.sr = None
        self.current_file = None
        
        # Processing queue for threading
        self.process_queue = queue.Queue()
        
        # Create GUI
        self.create_widgets()
        
        # Start processing thread
        self.processing = False
        self.check_queue()
        
    def create_widgets(self):
        """Create the GUI widgets"""
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="TRUE CAK Neural Grain Processor", 
                               font=('Helvetica', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # File selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="10")
        file_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.file_label = ttk.Label(file_frame, text="No file selected")
        self.file_label.grid(row=0, column=0, padx=5)
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=5)
        
        # Grain control
        control_frame = ttk.LabelFrame(main_frame, text="Grain Control", padding="10")
        control_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(control_frame, text="Grain Amount:").grid(row=0, column=0, padx=5)
        
        self.grain_var = tk.DoubleVar(value=0.0)
        self.grain_slider = ttk.Scale(control_frame, from_=-3.0, to=3.0, 
                                     variable=self.grain_var, orient=tk.HORIZONTAL,
                                     length=400, command=self.update_grain_label)
        self.grain_slider.grid(row=0, column=1, padx=5)
        
        self.grain_label = ttk.Label(control_frame, text="0.0")
        self.grain_label.grid(row=0, column=2, padx=5)
        
        # Info label
        info_text = ("Negative values: Reduce texture (smooth)\n"
                    "Zero: No change (identity)\n"
                    "Positive values: Add texture (grain)")
        info_label = ttk.Label(control_frame, text=info_text, font=('Helvetica', 9))
        info_label.grid(row=1, column=0, columnspan=3, pady=5)
        
        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Audio", 
                                        command=self.process_audio, state=tk.DISABLED)
        self.process_button.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, 
                                           maximum=1.0, length=500)
        self.progress_bar.grid(row=4, column=0, columnspan=2, pady=5)
        
        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=5, column=0, columnspan=2, pady=5)
        
        # Visualization
        viz_frame = ttk.LabelFrame(main_frame, text="Spectrogram Visualization", padding="10")
        viz_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        # Create matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.fig.tight_layout()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Save button
        self.save_button = ttk.Button(main_frame, text="Save Processed Audio", 
                                     command=self.save_audio, state=tk.DISABLED)
        self.save_button.grid(row=7, column=0, columnspan=2, pady=10)
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
    def browse_file(self):
        """Browse for an audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")]
        )
        
        if filename:
            self.current_file = filename
            self.file_label.config(text=Path(filename).name)
            self.process_button.config(state=tk.NORMAL)
            
            # Load and display original spectrogram
            self.load_original_audio()
            
    def load_original_audio(self):
        """Load the original audio and display its spectrogram"""
        try:
            self.original_audio, self.sr = librosa.load(self.current_file, sr=None, mono=True)
            
            # Display spectrogram (first 5 seconds for clarity)
            display_length = min(len(self.original_audio), self.sr * 5)
            S = librosa.feature.melspectrogram(y=self.original_audio[:display_length], 
                                              sr=self.sr, n_mels=128, n_fft=2048, hop_length=512)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            self.ax1.clear()
            img1 = self.ax1.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
            self.ax1.set_title('Original')
            self.ax1.set_xlabel('Time')
            self.ax1.set_ylabel('Mel Frequency')
            
            self.ax2.clear()
            self.ax2.set_title('Processed (not yet processed)')
            self.ax2.set_xlabel('Time')
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio: {str(e)}")
            
    def update_grain_label(self, value):
        """Update the grain amount label"""
        val = float(value)
        self.grain_label.config(text=f"{val:.1f}")
        
        # Update label color based on value
        if abs(val) < 0.1:
            self.grain_label.config(foreground='green')  # Near identity
        elif val < 0:
            self.grain_label.config(foreground='blue')   # Smoothing
        else:
            self.grain_label.config(foreground='red')    # Adding grain
        
    def process_audio(self):
        """Process the audio in a separate thread"""
        if not self.processing:
            self.processing = True
            self.process_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.DISABLED)
            
            # Start processing in thread
            thread = threading.Thread(target=self._process_audio_thread)
            thread.start()
            
    def _process_audio_thread(self):
        """Thread function for audio processing"""
        try:
            grain_amount = self.grain_var.get()
            
            # Process audio with TRUE CAK
            self.processed_audio, self.sr = self.processor.process_audio(
                self.current_file, grain_amount, 
                progress_callback=lambda p, s: self.process_queue.put(('progress', p, s))
            )
            
            self.process_queue.put(('complete', None, None))
            
        except Exception as e:
            self.process_queue.put(('error', str(e), None))
            
    def check_queue(self):
        """Check the processing queue for updates"""
        try:
            while True:
                msg_type, data1, data2 = self.process_queue.get_nowait()
                
                if msg_type == 'progress':
                    self.progress_var.set(data1)
                    self.status_label.config(text=data2)
                    
                elif msg_type == 'complete':
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL)
                    self.save_button.config(state=tk.NORMAL)
                    self.progress_var.set(0)
                    self.status_label.config(text="Processing complete!")
                    
                    # Update visualization
                    self.update_visualization()
                    
                elif msg_type == 'error':
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL)
                    self.progress_var.set(0)
                    self.status_label.config(text="Error occurred")
                    messagebox.showerror("Processing Error", data1)
                    
        except queue.Empty:
            pass
            
        # Schedule next check
        self.root.after(100, self.check_queue)
        
    def update_visualization(self):
        """Update the spectrogram visualization"""
        if self.processed_audio is not None:
            # Display processed spectrogram (first 5 seconds)
            display_length = min(len(self.processed_audio), self.sr * 5)
            S = librosa.feature.melspectrogram(y=self.processed_audio[:display_length], 
                                              sr=self.sr, n_mels=128, n_fft=2048, hop_length=512)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            self.ax2.clear()
            img2 = self.ax2.imshow(S_db, aspect='auto', origin='lower', cmap='viridis')
            self.ax2.set_title(f'Processed (Grain: {self.grain_var.get():.1f})')
            self.ax2.set_xlabel('Time')
            
            self.canvas.draw()
            
    def save_audio(self):
        """Save the processed audio"""
        if self.processed_audio is not None:
            # Create default filename with grain value
            grain_val = self.grain_var.get()
            default_name = f"{Path(self.current_file).stem}_grain_{grain_val:.1f}.wav"
            
            filename = filedialog.asksaveasfilename(
                title="Save Processed Audio",
                defaultextension=".wav",
                initialfile=default_name,
                filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
            )
            
            if filename:
                try:
                    sf.write(filename, self.processed_audio, self.sr)
                    messagebox.showinfo("Success", f"Audio saved successfully!\n{Path(filename).name}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save audio: {str(e)}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = NeuralAudioProcessorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
import torch
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
import json
import sys

sys.path.append('.')

# cak model imports
from cak_main_sandbox import SharedTextureDetector, TextureGenerator


class AudioProcessor:
    """audio processor for CAK using iSTFT"""

    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else
                                   'mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # audio params stats from training
        self.sample_rate = 44100
        self.n_fft = 2048
        self.hop_size = 512
        self.win_size = 2048

        # global norm stats from training
        stats_path = 'cak_bigvgan_dataset/global_normalization_stats.json'
        with open(stats_path, 'r') as f:
            self.global_stats = json.load(f)

        print(f"Loaded global stats: low={self.global_stats['global_low']:.3f}, "
              f"high={self.global_stats['global_high']:.3f}")

        # create model architecture
        self.shared_detector = SharedTextureDetector()
        self.generator = TextureGenerator(self.shared_detector)

        # load trained weights
        checkpoint_path = 'wgan_grain_output/final_wgan_grain.pt'
        if Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.generator.load_state_dict(checkpoint['generator'])
            self.shared_detector.load_state_dict(checkpoint['shared_detector'])

            # set epoch for temperature (should be at final value, if last checkpoint is used)
            if 'epoch' in checkpoint:
                self.generator.update_epoch(checkpoint['epoch'])

            print(f"Model loaded from: {checkpoint_path}")
            print(f"Scale parameter: {self.generator.cak.scale.item():.3f}")
            print(f"Temperature: {self.generator.cak.temperature:.1f}")
            print(f"Threshold: {self.generator.cak.threshold.item():.2f}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        # move to device and eval
        self.shared_detector.to(self.device)
        self.generator.to(self.device)
        self.generator.eval()

        # sanity check to make sure iSTFT is calibrated
        print("Using iSTFT for phase-preserving reconstruction")

    def normalize_stft(self, stft_mag):
        """STFT norm that matches training"""
        # sqrt compression
        sqrt_mag = np.sqrt(stft_mag)

        # log compression
        log_mag = np.log1p(sqrt_mag * self.global_stats['alpha'])

        # norm with global stats
        normalized = np.clip(
            (log_mag - self.global_stats['global_low']) /
            (self.global_stats['global_high'] - self.global_stats['global_low'] + 1e-8),
            0, 1
        )

        # gamma correction
        gamma_corrected = np.power(normalized, self.global_stats['gamma'])

        return gamma_corrected

    def denormalize_stft(self, normalized):
        """reverse the STFT norm"""
        # ensure input is in valid range [0, 1]
        normalized = np.clip(normalized, 0, 1)

        # reverse gamma
        pre_gamma = np.power(normalized + 1e-8, 1.0 / self.global_stats['gamma'])

        # reverse norm
        log_mag = pre_gamma * (self.global_stats['global_high'] - self.global_stats['global_low']) + \
                  self.global_stats['global_low']

        # reverse log compression
        sqrt_mag = (np.exp(log_mag) - 1) / self.global_stats['alpha']
        sqrt_mag = np.maximum(sqrt_mag, 0)  # Ensure non-negative

        # reverse sqrt compression
        mag = sqrt_mag ** 2

        # safety check
        mag = np.maximum(mag, 0)

        return mag

    def process_audio(self, audio_path, grain_amount, progress_callback=None):
        """process audio file with learned neuron"""

        # load audio
        if progress_callback:
            progress_callback(0.1, "Loading audio...")

        y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # process in chunks (15 seconds, matching training, chunks will take longer if more than 15 seconds)
        chunk_duration = 15.0
        chunk_samples = int(chunk_duration * self.sample_rate)
        hop_samples = chunk_samples // 2  # 50% overlap

        chunks = []
        for start in range(0, len(y) - chunk_samples + hop_samples, hop_samples):
            chunk = y[start:start + chunk_samples]
            if len(chunk) < chunk_samples:
                # Pad last chunk
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            chunks.append((chunk, start))

        # if audio is shorter than chunk_samples, process as single chunk
        if len(chunks) == 0:
            chunk = np.pad(y, (0, max(0, chunk_samples - len(y))), mode='constant')
            chunks = [(chunk, 0)]

            print(f"\n=== Audio Info ===")
            print(f"Audio length: {len(y) / self.sample_rate:.2f} seconds")
            print(f"Chunk duration: {chunk_duration} seconds")
            print(f"Number of chunks: {len(chunks)}")
            print(f"Grain amount: {grain_amount}")

        processed_chunks = []
        total_chunks = len(chunks)

        for idx, (chunk, start_pos) in enumerate(chunks):
            if progress_callback:
                progress = 0.1 + (0.7 * idx / total_chunks)
                progress_callback(progress, f"Processing chunk {idx + 1}/{total_chunks}")

            # compute complex STFT
            stft_complex = librosa.stft(
                chunk,
                n_fft=self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size
            )

            # get magnitude and phase
            magnitude = np.abs(stft_complex)
            phase = np.angle(stft_complex)

            # norm magnitude
            normalized_mag = self.normalize_stft(magnitude)

            # process with trained neural network
            with torch.no_grad():
                # prepare 4D input [B, 1, F, T]
                mag_tensor = torch.FloatTensor(normalized_mag).unsqueeze(0).unsqueeze(0).to(self.device)
                grain_tensor = torch.FloatTensor([[grain_amount]]).to(self.device)

                # apply CAK processing
                processed_mag, patterns = self.generator(mag_tensor, grain_tensor)
                processed_mag = processed_mag.cpu().numpy()[0, 0]

                input_mean = normalized_mag.mean()
                output_mean = processed_mag.mean()
                print(f"\nChunk {idx + 1}:")
                print(f"  Input mean: {input_mean:.6f}")
                print(f"  Output mean: {output_mean:.6f}")
                print(f"  Difference: {abs(output_mean - input_mean):.6f}")
                print(f"  Gate value: {self.generator.cak.soft_gate(grain_tensor).item():.3f}")

            # denorm back to linear magnitude
            processed_mag_linear = self.denormalize_stft(processed_mag)

            # reconstruct complex STFT using original phase
            processed_stft = processed_mag_linear * np.exp(1j * phase)

            # convert back to audio using iSTFT
            processed_chunk = librosa.istft(
                processed_stft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                length=chunk_samples
            )

            # sanity debug for first chunk
            if idx == 0:
                print(f"\nProcessing stats:")
                print(f"  Original magnitude range: [{magnitude.min():.3f}, {magnitude.max():.3f}]")
                print(f"  Normalized range: [{normalized_mag.min():.3f}, {normalized_mag.max():.3f}]")
                print(
                    f"  Processed magnitude range: [{processed_mag_linear.min():.3f}, {processed_mag_linear.max():.3f}]")
                print(f"  Grain amount: {grain_amount:.2f}")
                if abs(grain_amount) < 1e-6:
                    # Check identity preservation
                    mag_diff = np.abs(processed_mag_linear - magnitude).mean()
                    print(f"  Identity check - mean magnitude difference: {mag_diff:.6f}")

            processed_chunks.append((processed_chunk, start_pos))

        # crossfade chunks
        if progress_callback:
            progress_callback(0.9, "Combining audio...")

        output_audio = self.combine_chunks(processed_chunks, len(y))

        # match RMS to original
        original_rms = np.sqrt(np.mean(y ** 2))
        output_rms = np.sqrt(np.mean(output_audio ** 2))

        if output_rms > 0:
            rms_ratio = original_rms / output_rms
            rms_ratio = np.clip(rms_ratio, 0.1, 10.0)
            output_audio *= rms_ratio

        # soft clipping to prevent any peaks
        output_audio = np.tanh(output_audio * 0.95) / 0.95

        if progress_callback:
            progress_callback(1.0, "Complete!")

        return output_audio, self.sample_rate

    def combine_chunks(self, chunks, target_length):
        """combine overlapping chunks with crossfade"""
        output = np.zeros(target_length)
        weights = np.zeros(target_length)

        for chunk, start_pos in chunks:
            # ensure 1D
            if chunk.ndim > 1:
                chunk = chunk.flatten()

            end_pos = min(start_pos + len(chunk), target_length)
            chunk_len = end_pos - start_pos

            # fade window
            window = np.ones(chunk_len)
            fade_len = min(self.hop_size // 2, chunk_len // 4)

            if start_pos > 0 and fade_len > 0:
                fade_in = np.linspace(0, 1, fade_len)
                window[:fade_len] = fade_in

            if end_pos < target_length and fade_len > 0:
                fade_out = np.linspace(1, 0, fade_len)
                window[-fade_len:] = fade_out

            output[start_pos:end_pos] += chunk[:chunk_len] * window
            weights[start_pos:end_pos] += window

        # norm by weights
        mask = weights > 0
        output[mask] /= weights[mask]

        return output

class NeuralAudioProcessorGUI:
    """GUI for WGAN-GP CAK Neural Audio Processor"""

    def __init__(self, root):
        self.root = root
        self.root.tk.call('tk', 'scaling', 0.3)
        self.root.title("WGAN-GP CAK Neural Processor - iSTFT")
        self.root.geometry("700x550")

        # initialize processor
        try:
            self.processor = AudioProcessor()
            detector_weights = self.processor.shared_detector.conv.weight.data.cpu().numpy()
            print("\n=== Detector Learned Pattern (3x3 kernel) ===")
            print(detector_weights.squeeze())
            print("=" * 40)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize processor: {e}")
            root.destroy()
            return

        # audio data
        self.original_audio = None
        self.processed_audio = None
        self.sr = None
        self.current_file = None

        # processing queue
        self.process_queue = queue.Queue()

        # create GUI
        self.create_widgets()

        # commence processing
        self.processing = False
        self.check_queue()

    def create_widgets(self):
        """Create the GUI widgets"""

        # main fram
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # title
        title_label = ttk.Label(main_frame, text="WGAN-GP CAK Neural Processor",
                                font=('Helvetica', 18, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)

        # model info
        info_text = (f"Soft-Gate CAK | τ={self.processor.generator.cak.threshold.item():.2f} | "
                     f"T={self.processor.generator.cak.temperature:.1f} | "
                     f"Scale={self.processor.generator.cak.scale.item():.3f} | "
                     f"Vocoder: iSTFT (phase-preserving)")
        info_label = ttk.Label(main_frame, text=info_text,
                               font=('Helvetica', 11, 'italic'), foreground='green')
        info_label.grid(row=1, column=0, columnspan=2, pady=5)

        # file selection
        file_frame = ttk.LabelFrame(main_frame, text="Audio File", padding="10")
        file_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.file_label = ttk.Label(file_frame, text="No file selected", width=40)
        self.file_label.grid(row=0, column=0, padx=5)

        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1, padx=5)

        # control and amount
        control_frame = ttk.LabelFrame(main_frame, text="Control", padding="10")
        control_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(control_frame, text="Amount:").grid(row=0, column=0, padx=5)

        self.grain_var = tk.DoubleVar(value=0.0)
        self.grain_slider = ttk.Scale(control_frame, from_=-2.0, to=2.0,
                                      variable=self.grain_var, orient=tk.HORIZONTAL,
                                      length=400, command=self.update_grain_label)
        self.grain_slider.grid(row=0, column=1, padx=5)

        self.grain_label = ttk.Label(control_frame, text="0.00", width=6)
        self.grain_label.grid(row=0, column=2, padx=5)

        # Gate indicator
        self.gate_label = ttk.Label(control_frame, text="Gate: 0.002", width=12)
        self.gate_label.grid(row=0, column=3, padx=5)

        # info about threshold
        threshold = self.processor.generator.cak.threshold.item()
        info_text = (f"Threshold τ = {threshold:.2f}\n"
                     f"Values < {threshold:.2f}: Minimal effect (gate closed)\n"
                     f"Values ≥ {threshold:.2f}: Grain modulation active\n"
                     f"Zero: Perfect identity preservation (phase-accurate!)")
        info_label = ttk.Label(control_frame, text=info_text, font=('Helvetica', 9))
        info_label.grid(row=1, column=0, columnspan=4, pady=5)

        # process button
        self.process_button = ttk.Button(main_frame, text="Process Audio",
                                         command=self.process_audio, state=tk.DISABLED)
        self.process_button.grid(row=4, column=0, columnspan=2, pady=10)

        # progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var,
                                            maximum=1.0, length=600)
        self.progress_bar.grid(row=5, column=0, columnspan=2, pady=5)

        self.status_label = ttk.Label(main_frame, text="Ready")
        self.status_label.grid(row=6, column=0, columnspan=2, pady=5)

        # visualize spec
        viz_frame = ttk.LabelFrame(main_frame, text="Spectrogram Visualization", padding="10")
        viz_frame.grid(row=7, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(4, 1.5))
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # playback
        playback_frame = ttk.LabelFrame(main_frame, text="Playback & Export", padding="10")
        playback_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        ttk.Button(playback_frame, text="Play Original",
                   command=self.play_original).grid(row=0, column=0, padx=5)
        ttk.Button(playback_frame, text="Play Processed",
                   command=self.play_processed).grid(row=0, column=1, padx=5)
        ttk.Button(playback_frame, text="Save Processed",
                   command=self.save_processed).grid(row=0, column=2, padx=5)

        # A/B test button
        ttk.Button(playback_frame, text="A/B Test",
                   command=self.ab_test).grid(row=0, column=3, padx=5)

    def update_grain_label(self, value):
        """update label and gate indicator"""
        grain_val = float(value)
        self.grain_label.config(text=f"{grain_val:.2f}")

        # calculate gate value
        with torch.no_grad():
            grain_tensor = torch.tensor([[grain_val]]).to(self.processor.device)
            gate_value = self.processor.generator.cak.soft_gate(grain_tensor).item()
            self.gate_label.config(text=f"Gate: {gate_value:.3f}")

    def browse_file(self):
        """browse for audio file"""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.m4a"), ("All Files", "*.*")]
        )
        if filename:
            self.current_file = filename
            self.file_label.config(text=Path(filename).name)
            self.process_button.config(state=tk.NORMAL)
            self.original_audio, self.sr = librosa.load(filename, sr=self.processor.sample_rate, mono=True)
            self.visualize_original()

    def visualize_original(self):
        """visualize original audio"""
        if self.original_audio is None:
            return

        # show first 3 seconds
        y_vis = self.original_audio[:self.sr * 3]

        # compute STFT for visualization
        stft = librosa.stft(y_vis, n_fft=self.processor.n_fft,
                            hop_length=self.processor.hop_size)
        mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        self.ax1.clear()
        librosa.display.specshow(mag_db, sr=self.sr, hop_length=self.processor.hop_size,
                                 x_axis='time', y_axis='hz', ax=self.ax1)
        self.ax1.set_title('Original Audio')
        self.ax1.set_ylim(0, 8000)  # focus on relevant frequency range

        self.canvas.draw()

    def process_audio(self):
        """process audio in background thread"""
        if self.processing:
            return

        self.processing = True
        self.process_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._process_thread)
        thread.start()

    def _process_thread(self):
        """background processing thread"""
        try:
            grain_amount = self.grain_var.get()

            self.processed_audio, self.sr = self.processor.process_audio(
                self.current_file,
                grain_amount,
                progress_callback=self._update_progress
            )

            self.process_queue.put(('complete', None))

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
            self.process_queue.put(('error', error_msg))

    def _update_progress(self, progress, message):
        self.process_queue.put(('progress', (progress, message)))

    def check_queue(self):
        """check processing queue"""
        try:
            while True:
                msg_type, data = self.process_queue.get_nowait()

                if msg_type == 'progress':
                    progress, message = data
                    self.progress_var.set(progress)
                    self.status_label.config(text=message)

                elif msg_type == 'complete':
                    self.visualize_processed()
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL)
                    self.status_label.config(text="Processing complete!")

                elif msg_type == 'error':
                    messagebox.showerror("Processing Error", data)
                    self.processing = False
                    self.process_button.config(state=tk.NORMAL)
                    self.status_label.config(text="Error occurred")

        except queue.Empty:
            pass

        self.root.after(100, self.check_queue)

    def visualize_processed(self):
        """visualize processed audio"""
        if self.processed_audio is None:
            return

        # show first 3 seconds
        y_vis = self.processed_audio[:self.sr * 3]

        # compute STFT
        stft = librosa.stft(y_vis, n_fft=self.processor.n_fft,
                            hop_length=self.processor.hop_size)
        mag_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        # plot
        self.ax2.clear()
        librosa.display.specshow(mag_db, sr=self.sr, hop_length=self.processor.hop_size,
                                 x_axis='time', y_axis='hz', ax=self.ax2)

        grain_val = self.grain_var.get()
        gate_val = float(self.gate_label.cget("text").split(": ")[1])

        if abs(grain_val) < 1e-6:
            title = 'Processed (IDENTITY - Phase Preserved!)'
        else:
            title = f'Processed (Grain: {grain_val:.2f}, Gate: {gate_val:.3f})'

        self.ax2.set_title(title)
        self.ax2.set_ylim(0, 8000)

        self.canvas.draw()

    def play_original(self):
        """play original audio"""
        if self.original_audio is None:
            return

        try:
            import sounddevice as sd
            sd.play(self.original_audio, self.sr)
        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")

    def play_processed(self):
        """play processed audio"""
        if self.processed_audio is None:
            messagebox.showinfo("Info", "Process audio first")
            return

        try:
            import sounddevice as sd
            sd.play(self.processed_audio, self.sr)
        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")

    def save_processed(self):
        """save processed audio"""
        if self.processed_audio is None:
            messagebox.showinfo("Info", "Process audio first")
            return

        filename = filedialog.asksaveasfilename(
            title="Save Processed Audio",
            defaultextension=".wav",
            filetypes=[("WAV Files", "*.wav"), ("All Files", "*.*")]
        )

        if filename:
            # ensure audio is in valid range
            audio_clipped = np.clip(self.processed_audio, -1.0, 1.0)
            sf.write(filename, audio_clipped, self.sr)
            messagebox.showinfo("Success", f"Saved to {filename}")

    def ab_test(self):
        """quick A/B test between original and processed"""
        if self.original_audio is None or self.processed_audio is None:
            messagebox.showinfo("Info", "Process audio first")
            return

        try:
            import sounddevice as sd
            import time

            # play 2 seconds of each, alternating
            messagebox.showinfo("A/B Test", "Playing: Original (2s) → Processed (2s) → Original (2s)")

            sd.play(self.original_audio[:self.sr * 2], self.sr)
            sd.wait()
            time.sleep(0.5)

            sd.play(self.processed_audio[:self.sr * 2], self.sr)
            sd.wait()
            time.sleep(0.5)

            sd.play(self.original_audio[:self.sr * 2], self.sr)
            sd.wait()

        except ImportError:
            messagebox.showinfo("Info", "Install sounddevice: pip install sounddevice")


def main():
    """run the application"""
    root = tk.Tk()
    app = NeuralAudioProcessorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
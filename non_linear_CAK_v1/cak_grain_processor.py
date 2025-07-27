import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from pathlib import Path

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# ============= TRUE CAK COMPONENTS - AUSTIN'S FORMULA =============
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
        h = F.leaky_relu(self.conv1(x), 0.2)
        h = F.leaky_relu(self.conv2(h), 0.2)
        h = F.leaky_relu(self.conv3(h), 0.2)
        patterns = self.conv4(h)  # No activation - can be positive or negative
        return patterns

class TrueCAKLayer(nn.Module):
    """
    TRUE CAK implementation - exactly Austin's formula:
    output = x + (detected_patterns × grain_value)
    
    NO spectral warping network
    NO complex transformations
    Just detect and scale!
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
        
        That's it. No warping, no complex networks.
        """
        # Step 1: Detect patterns in the input
        patterns = self.pattern_detector(x)
        
        # Step 2: Scale patterns by grain value
        grain_v = grain_value.view(-1, 1, 1, 1)
        gate = self.soft_gate(grain_value).view(-1, 1, 1, 1)
        
        # Step 3: The Austin formula
        # output = x + (patterns × grain_value × gate × scale)
        scaled_patterns = patterns * grain_v * gate * self.scale
        output = x + scaled_patterns
        
        return output, patterns

# ============= PURE NEURAL GRAIN PROCESSOR =============
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
        
        # NO refinement network - that was adding noise!
        # NO residual connections - that was bypassing CAK!
        
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

# ============= SIMPLE DISCRIMINATOR =============
class SimpleGrainAuditor(nn.Module):
    """
    Simplified auditor that ONLY checks grain level
    Mirrors generator's pattern detection
    """
    
    def __init__(self):
        super(SimpleGrainAuditor, self).__init__()
        
        # Mirror the generator's pattern detection
        self.pattern_net = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, kernel_size=(5, 5), stride=2, padding=(2, 2)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Predict grain level
        self.grain_predictor = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        features = self.pattern_net(x)
        features = features.view(features.size(0), -1)
        grain_pred = self.grain_predictor(features)
        return grain_pred

class SimpleDiscriminator(nn.Module):
    """
    Simplified discriminator - focus on grain compliance
    """
    
    def __init__(self):
        super(SimpleDiscriminator, self).__init__()
        
        # Basic realism check
        self.realism_net = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Grain auditor
        self.grain_auditor = SimpleGrainAuditor()
    
    def compute_grain_violation(self, claimed, detected):
        """Simple violation - how wrong was the grain prediction?"""
        return torch.abs(claimed - detected)
    
    def forward(self, x, grain_target):
        # Check realism
        realism_score = self.realism_net(x)
        
        # Check grain
        grain_pred = self.grain_auditor(x)
        grain_violation = self.compute_grain_violation(grain_target, grain_pred)
        
        return realism_score, grain_pred, grain_violation

# ============= FOCUSED DATASET =============
class FocusedGrainDataset(Dataset):
    """
    Simplified dataset - focus on grain transformation
    Always includes identity pairs for calibration
    """
    
    def __init__(self, metadata_path='metadata.json', n_fft=2048, hop_length=512, 
                 n_mels=128, target_length=256):
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Get audio files sorted by grain
        self.audio_pairs = []
        for path, info in self.metadata.items():
            if 'features' in info and 'grain_density' in info['features']:
                self.audio_pairs.append((path, info['features']['grain_density']))
        
        self.audio_pairs.sort(key=lambda x: x[1])
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.target_length = target_length
        
        print(f"Loaded {len(self.audio_pairs)} audio files")
        print(f"Grain range: [{self.audio_pairs[0][1]:.6f}, {self.audio_pairs[-1][1]:.6f}]")
    
    def __len__(self):
        return len(self.audio_pairs) * 2  # Each file used twice
    
    def __getitem__(self, idx):
        # 50% identity, 50% transformation
        if idx % 2 == 0:
            # IDENTITY: input = output, grain = 0
            file_idx = idx // 2
            audio_path, _ = self.audio_pairs[file_idx]
            
            # Load and process
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Random segment
            segment_length = sr * 3
            if len(y) > segment_length:
                start = np.random.randint(0, len(y) - segment_length)
                y = y[start:start + segment_length]
            else:
                y = np.pad(y, (0, segment_length - len(y)), mode='constant')
            
            # Create spectrogram
            S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft,
                                              hop_length=self.hop_length, n_mels=self.n_mels)
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Normalize
            if S_db.shape[1] < self.target_length:
                S_db = np.pad(S_db, ((0, 0), (0, self.target_length - S_db.shape[1])), mode='constant')
            else:
                S_db = S_db[:, :self.target_length]
            
            S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
            S_norm = S_norm * 2 - 1
            
            spec_tensor = torch.FloatTensor(S_norm).unsqueeze(0)
            
            # Identity transformation
            return spec_tensor, spec_tensor, torch.FloatTensor([0.0])
        
        else:
            # TRANSFORMATION: low → high grain
            mid = len(self.audio_pairs) // 2
            low_idx = (idx // 2) % mid
            high_idx = mid + ((idx // 2) % (len(self.audio_pairs) - mid))
            
            low_path, low_grain = self.audio_pairs[low_idx]
            high_path, high_grain = self.audio_pairs[high_idx]
            
            # Load both files
            y_low, sr = librosa.load(low_path, sr=None, mono=True)
            y_high, _ = librosa.load(high_path, sr=None, mono=True)
            
            # Process segments
            segment_length = sr * 3
            for y in [y_low, y_high]:
                if len(y) > segment_length:
                    start = np.random.randint(0, len(y) - segment_length)
                    y = y[start:start + segment_length]
            
            # Create spectrograms
            S_low = librosa.feature.melspectrogram(y=y_low[:segment_length], sr=sr,
                                                   n_fft=self.n_fft, hop_length=self.hop_length,
                                                   n_mels=self.n_mels)
            S_high = librosa.feature.melspectrogram(y=y_high[:segment_length], sr=sr,
                                                    n_fft=self.n_fft, hop_length=self.hop_length,
                                                    n_mels=self.n_mels)
            
            # Convert to dB and normalize
            S_low_db = librosa.power_to_db(S_low, ref=np.max)
            S_high_db = librosa.power_to_db(S_high, ref=np.max)
            
            # Pad/crop to target length
            if S_low_db.shape[1] < self.target_length:
                S_low_db = np.pad(S_low_db, ((0, 0), (0, self.target_length - S_low_db.shape[1])), mode='constant')
            else:
                S_low_db = S_low_db[:, :self.target_length]
                
            if S_high_db.shape[1] < self.target_length:
                S_high_db = np.pad(S_high_db, ((0, 0), (0, self.target_length - S_high_db.shape[1])), mode='constant')
            else:
                S_high_db = S_high_db[:, :self.target_length]
            
            # Normalize
            S_low_norm = (S_low_db - S_low_db.min()) / (S_low_db.max() - S_low_db.min() + 1e-8)
            S_high_norm = (S_high_db - S_high_db.min()) / (S_high_db.max() - S_high_db.min() + 1e-8)
            S_low_norm = S_low_norm * 2 - 1
            S_high_norm = S_high_norm * 2 - 1
            
            # Simplified grain scaling
            grain_diff = (high_grain - low_grain) * 100  # Stronger signal
            
            return (torch.FloatTensor(S_low_norm).unsqueeze(0),
                   torch.FloatTensor(S_high_norm).unsqueeze(0),
                   torch.FloatTensor([grain_diff]))

# ============= FOCUSED TRAINING =============
def train_true_cak(processor, discriminator, dataloader, num_epochs=100):
    """Training focused on learning Austin's formula"""
    
    # Optimizers - higher learning rate for faster convergence
    g_optimizer = optim.Adam(processor.parameters(), lr=0.0004, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Losses
    bce_loss = nn.BCELoss()
    l1_loss = nn.L1Loss()
    
    # History
    history = {
        'g_loss': [], 'd_loss': [],
        'grain_error': [], 'identity_loss': [],
        'pattern_strength': []
    }
    
    # Output dirs
    Path('true_cak_output').mkdir(exist_ok=True)
    Path('true_cak_output/checkpoints').mkdir(exist_ok=True)
    Path('true_cak_output/samples').mkdir(exist_ok=True)
    
    print("\n=== TRUE CAK TRAINING ===")
    print("Learning Austin's formula: output = x + (patterns × grain)")
    print("NO warping networks, NO residuals, just detect and scale!\n")
    
    for epoch in range(num_epochs):
        g_losses, d_losses, grain_errors, id_losses = [], [], [], []
        pattern_strengths = []
        
        progress = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for input_specs, target_specs, grain_values in progress:
            batch_size = input_specs.size(0)
            input_specs = input_specs.to(device)
            target_specs = target_specs.to(device)
            grain_values = grain_values.to(device)
            
            # Track identity
            is_identity = (grain_values == 0).float()
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            discriminator.zero_grad()
            
            # Process with generator
            processed_specs, patterns = processor(input_specs, grain_values)
            
            # Track pattern strength
            pattern_strengths.append(patterns.abs().mean().item())
            
            # Discriminate real
            real_score, real_grain, real_violation = discriminator(target_specs, grain_values)
            d_real_loss = bce_loss(real_score, real_labels)
            
            # Discriminate fake
            fake_score, fake_grain, fake_violation = discriminator(
                processed_specs.detach(), grain_values
            )
            d_fake_loss = bce_loss(fake_score, fake_labels)
            
            # Total D loss - HEAVY emphasis on grain accuracy
            d_loss = d_real_loss + d_fake_loss + real_violation.mean() * 5.0
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            processor.zero_grad()
            
            # Generate
            processed_specs, patterns = processor(input_specs, grain_values)
            
            # Fool discriminator
            fake_score, fake_grain, fake_violation = discriminator(processed_specs, grain_values)
            
            # Losses
            g_adv_loss = bce_loss(fake_score, real_labels)
            g_recon_loss = l1_loss(processed_specs, target_specs)
            
            # Extra weight for identity
            identity_weight = 5.0  # VERY important to learn grain=0 means no change
            weighted_recon = (g_recon_loss * is_identity.view(-1, 1, 1, 1) * identity_weight +
                             g_recon_loss * (1 - is_identity.view(-1, 1, 1, 1))).mean()
            
            # Track identity loss
            if is_identity.sum() > 0:
                id_loss = l1_loss(
                    processed_specs[is_identity.bool()],
                    target_specs[is_identity.bool()]
                ).item()
                id_losses.append(id_loss)
            
            # Total G loss - MASSIVE emphasis on grain compliance
            g_loss = g_adv_loss + weighted_recon * 10 + fake_violation.mean() * 20
            g_loss.backward()
            g_optimizer.step()
            
            # Record
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            grain_errors.append(fake_violation.mean().item())
            
            # Progress
            progress.set_postfix({
                'G': f'{g_loss.item():.3f}',
                'D': f'{d_loss.item():.3f}',
                'GrainErr': f'{fake_violation.mean().item():.3f}',
                'Pattern': f'{patterns.abs().mean().item():.3f}'
            })
        
        # Epoch stats
        history['g_loss'].append(np.mean(g_losses))
        history['d_loss'].append(np.mean(d_losses))
        history['grain_error'].append(np.mean(grain_errors))
        history['identity_loss'].append(np.mean(id_losses) if id_losses else 0)
        history['pattern_strength'].append(np.mean(pattern_strengths))
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"G: {history['g_loss'][-1]:.4f}, "
              f"D: {history['d_loss'][-1]:.4f}, "
              f"Identity: {history['identity_loss'][-1]:.4f}, "
              f"Pattern: {history['pattern_strength'][-1]:.4f}")
        
        # Visualize
        if (epoch + 1) % 10 == 0:
            visualize_true_cak(processor, input_specs[0:1], epoch + 1, device)
        
        # Checkpoint
        if (epoch + 1) % 25 == 0:
            torch.save({
                'processor': processor.state_dict(),
                'discriminator': discriminator.state_dict(),
                'history': history
            }, f'true_cak_output/checkpoints/epoch_{epoch+1}.pt')
    
    # Save final
    torch.save({
        'processor': processor.state_dict(),
        'discriminator': discriminator.state_dict(),
        'history': history
    }, 'true_cak_output/final_true_cak.pt')
    
    plot_true_cak_history(history)
    return history

def visualize_true_cak(processor, test_spec, epoch, device):
    """Visualize grain control"""
    processor.eval()
    
    grain_levels = [-3, -1.5, 0, 1.5, 3]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    with torch.no_grad():
        for i, grain in enumerate(grain_levels):
            grain_tensor = torch.FloatTensor([[grain]]).to(device)
            processed, patterns = processor(test_spec, grain_tensor)
            
            # Show processed
            spec = ((processed[0, 0].cpu().numpy() + 1) / 2)
            axes[0, i].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
            axes[0, i].set_title(f'Grain: {grain}')
            
            # Show detected patterns
            pattern = patterns[0, 0].cpu().numpy()
            axes[1, i].imshow(pattern, aspect='auto', origin='lower', cmap='RdBu')
            axes[1, i].set_title(f'Detected Patterns')
    
    plt.suptitle(f'TRUE CAK at Epoch {epoch} - Austin\'s Formula')
    plt.tight_layout()
    plt.savefig(f'true_cak_output/samples/epoch_{epoch}_grain.png')
    plt.close()
    
    processor.train()

def plot_true_cak_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Losses
    axes[0, 0].plot(history['g_loss'], label='Generator')
    axes[0, 0].plot(history['d_loss'], label='Discriminator')
    axes[0, 0].set_title('Training Losses')
    axes[0, 0].legend()
    
    # Grain error
    axes[0, 1].plot(history['grain_error'], color='red')
    axes[0, 1].set_title('Grain Violation')
    
    # Identity loss
    axes[1, 0].plot(history['identity_loss'], color='purple')
    axes[1, 0].set_title('Identity Loss (grain=0)')
    
    # Pattern strength
    axes[1, 1].plot(history['pattern_strength'], color='green')
    axes[1, 1].set_title('Pattern Detection Strength')
    
    plt.tight_layout()
    plt.savefig('true_cak_output/training_history.png')
    plt.close()

# ============= MAIN =============
def main():
    # Dataset
    dataset = FocusedGrainDataset('metadata.json')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    print(f"Dataset: {len(dataset)} pairs (50% identity)")
    
    # Models
    processor = PureNeuralGrainProcessor().to(device)
    discriminator = SimpleDiscriminator().to(device)
    
    print(f"\nProcessor parameters: {sum(p.numel() for p in processor.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Train
    history = train_true_cak(processor, discriminator, dataloader, num_epochs=100)
    
    print("\nTraining complete! Check 'true_cak_output' folder.")
    
    # Final test
    test_final_true_cak(processor)

def test_final_true_cak(processor):
    """Test the learned formula"""
    processor.eval()
    
    # Load test file
    with open('metadata.json', 'r') as f:
        metadata = json.load(f)
    
    test_files = list(metadata.keys())[:5]
    
    fig, axes = plt.subplots(len(test_files), 6, figsize=(24, 4*len(test_files)))
    
    for row, test_file in enumerate(test_files):
        y, sr = librosa.load(test_file, sr=None, mono=True)
        y = y[:sr*3]
        
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min())
        S_norm = S_norm * 2 - 1
        S_tensor = torch.FloatTensor(S_norm[:, :256]).unsqueeze(0).unsqueeze(0).to(device)
        
        grain_values = [-2, -1, 0, 1, 2]
        
        with torch.no_grad():
            # Original
            orig = ((S_tensor[0, 0].cpu().numpy() + 1) / 2)
            axes[row, 0].imshow(orig, aspect='auto', origin='lower', cmap='viridis')
            axes[row, 0].set_title('Original' if row == 0 else '')
            axes[row, 0].set_ylabel(f'File {row+1}')
            
            # Grain variations
            for i, grain in enumerate(grain_values):
                grain_tensor = torch.FloatTensor([[grain]]).to(device)
                processed, _ = processor(S_tensor, grain_tensor)
                
                spec = ((processed[0, 0].cpu().numpy() + 1) / 2)
                axes[row, i+1].imshow(spec, aspect='auto', origin='lower', cmap='viridis')
                if row == 0:
                    axes[row, i+1].set_title(f'Grain: {grain}')
    
    plt.suptitle('TRUE CAK - Austin\'s Formula: output = x + (patterns × grain)', fontsize=16)
    plt.tight_layout()
    plt.savefig('true_cak_output/final_test_multiple_files.png', dpi=150)
    print("Saved final test to true_cak_output/final_test_multiple_files.png")

if __name__ == "__main__":
    main()
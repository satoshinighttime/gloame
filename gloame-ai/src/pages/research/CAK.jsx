import { useNavigate } from 'react-router-dom'
import { useEffect } from 'react'

function CAK() {
  const navigate = useNavigate()
  
  useEffect(() => {
    // Render LaTeX equations with KaTeX
    if (window.renderMathInElement) {
      window.renderMathInElement(document.body, {
        delimiters: [
          {left: '$$', right: '$$', display: true},
          {left: '$', right: '$', display: false},
        ],
        throwOnError: false
      })
    }
  }, [])
  
  return (
    <section className="paper-detail-section">
      <button 
        className="back-button"
        onClick={() => navigate('/research')}
      >
        ← back to research
      </button>
      
      <article className="paper-full">
        <h1 className="paper-detail-title">
          CAK: Conditioning-Aware Kernels for Personalized Audio Effects
        </h1>
        
        <div className="paper-detail-meta">
          <span className="authors">Austin Rockman, Roopam Garg</span>
          <span className="year">2025</span>
          <span className="venue">GLOAME</span>
        </div>
        
        <div className="paper-actions">
          <a href="#" className="action-button">PDF</a>
          <a href="#" className="action-button">ArXiv</a>
          <a href="#" className="action-button">GitHub</a>
        </div>
        
        <div className="paper-detail-abstract">
          <h3>Abstract</h3>
          <p>
            We demonstrate that a single 3×3 convolutional kernel can produce multiple audio effects when trained on just 200 samples 
            from a personalized corpus. This result challenges fundamental assumptions about neural audio processing complexity. 
            We achieve this through two key innovations: (1) Conditioning Aware Kernels (CAK), where output = input + (learned_pattern × control), 
            with a soft-gate mechanism guaranteeing identity preservation at zero control; and (2) AuGAN (Audit GAN), which reframes 
            adversarial training from "is this real?" to "did you apply the requested value?"
          </p>
          <p>
            Rather than learning to generate or detect forgeries, our networks cooperate to verify control application, discovering 
            interpretable features. The learned kernel exhibits a diagonal structure creating frequency-dependent temporal shifts that 
            produce perceptually distinct, musically meaningful effects based on input characteristics. This emergent behavior arises 
            not from explicit multi-task training but from the interaction between a single learned CAK and diverse input signals. 
            Our results show that adversarial dynamics can uncover novel signal transformations, enabling personalized effect design 
            across composition and sound-design workflows.
          </p>
        </div>
        
        <div className="paper-sections">
          <section className="paper-section">
            <h3>1. Introduction</h3>
            <p>
              Generative AI has captured the world's imagination by transforming deep learning from an analytical tool into a creative medium. 
              Although GANs and diffusion models enable stunning visual art and music generation, the manipulation of existing audio remains 
              firmly rooted in traditional signal processing. While mathematical DSP has given us powerful audio effects, from convolution 
              reverbs to analog circuit models, these rely on human insight to translate acoustic phenomena into equations. What if we could 
              learn audio effects directly from sound itself? We bridge this gap with Conditioning Aware Kernels (CAK), a novel modulation 
              technique that discovers transformations directly from data through adversarial training.
            </p>
            <p>
              This approach represents a radical departure from the scale-driven paradigm dominating modern AI. While contemporary models 
              pursue ever-larger architectures, demanding massive computational infrastructure and thousands of training examples, we embrace 
              an opposite philosophy: neural networks excel as simplification approximators. We show that complex behaviors can emerge without 
              architectural depth. By constraining our model to find minimal viable solutions, we demonstrate that sophisticated audio 
              transformations can emerge from just 200 training samples and 11 learnable parameters (a single 3×3 kernel with bias and scale).
            </p>
            <p>
              Human perception mirrors this efficiency. We learn to recognize the phenomenology of rain from a handful of experiences, 
              the pattern of droplets on windows, the rhythm on rooftops, the scent of petrichor, extracting rich, multi-sensory understanding 
              from minimal exposure. Similarly, experienced audio engineers develop intuition for effects through limited but focused interaction. 
              CAK captures this principle computationally: given a small corpus of audio with varying features, our system learns not just to 
              reproduce but to understand and apply discovered qualities across any input.
            </p>
            <p>
              Our method reformulates the adversarial game. Traditional GANs (Goodfellow et al., 2014) pit generator against discriminator 
              in a forgery detection contest. We propose an 'audit game' (AuGAN) where the discriminator must verify that the generator 
              applied the user's control value to its learned features, whatever those features turn out to be. This structural shift, 
              from deception to verification, enables the discriminator to guide the discovery of meaningful audio transformations rather 
              than policing authenticity.
            </p>
            <p>
              Our learned transformation consists of a single 3×3 convolutional kernel that exhibits remarkable adaptive behavior: the same 
              kernel produces a spectrum of perceptually distinct effects, as demonstrated in our listening examples and interactive GUI. 
              Analysis reveals an interpretable diagonal structure in the learned weights. This emergence of multiple effects from a single 
              pattern challenges the assumption that behavioral diversity requires architectural complexity.
            </p>
            <p>
              The implications extend beyond technical efficiency. CAK opens new creative possibilities: artists can devise sound effects 
              from individual collections. Our work demonstrates that the future of audio effects may lie not in more complex mathematics, 
              but in learning from the rich complexity already present in sound itself.
            </p>
            <p>
              To our knowledge, this is the first learned audio effect processor that generalizes musical modulation behavior from a minimal 
              convolutional kernel and a small training set. We hypothesize that emergent multi-behavior capabilities in neural networks do 
              not require depth or architectural complexity, but can arise from the interaction between a minimal learned pattern and diverse 
              input characteristics.
            </p>
          </section>
          
          <section className="paper-section">
            <h3>2. Related Work</h3>
            <p>
              <strong>Neural Audio Synthesis and Effects</strong>: WaveGAN (Donahue et al., 2018) and GANSynth (Engel et al., 2019) first showed 
              that adversarial training can generate raw waveform audio, but both rely on large datasets and models. DDSP (Engel et al., 2020) 
              and RAVE (Caillon & Esling, 2021) achieve high quality synthesis through compact architectures and strong inductive biases, 
              enabling efficient training even with limited data. Unlike those works, CAK is not a generator; it learns effects from a small, 
              personalized corpus and runs in real time with just 11 parameters.
            </p>
            <p>
              <strong>Conditioning Mechanisms</strong>: Feature-wise Linear Modulation (FiLM; Perez et al., 2018) conditions deep networks via 
              channel wise affine transforms. In a 2019 retrospective, the authors note that FiLM often needs additional task-specific inductive 
              biases to remain data-efficient (Perez et al., 2019 retrospective), a limitation we also observed when applying FiLM to complex 
              conditioning vectors. Dynamic kernel methods such as CondConv (Yang et al., 2019) synthesize weights by mixing basis filters. 
              CondConv effectively asks which kernel a layer should use; CAK instead detects a salient pattern once and modulates its residual 
              contribution by the user's control, staying within a two-digit parameter budget.
            </p>
            <p>
              <strong>Few-Shot Learning</strong>: Audio few shot work typically relies on meta-learning (MAML, Finn et al., 2017; Prototypical 
              Networks, Snell et al., 2017), which require access to large and diverse meta-training sets composed of many tasks. CAK operates 
              in a more extreme regime: it learns directly from a single, 50-minute corpus without episodic sampling or meta-tasks.
            </p>
            <p>
              <strong>Emergent Complexity from Simple Rules</strong>: Growing Isotropic Neural Cellular Automata (Mordvintsev, Randazzo, and Fouts, 2022) 
              demonstrates similar principles in the visual domain, where simple local update rules produce complex emergent patterns. Like CAK, 
              this work shows that behavioral diversity can arise from the interaction between minimal fixed rules and varying initial conditions, 
              rather than from architectural complexity. This parallel from a different domain validates our hypothesis that emergence, not 
              engineering, can drive adaptable behavior.
            </p>
            <p>
              <strong>Biological Inspiration</strong>: Our emphasis on minimal, data-efficient representations echoes the efficient-coding hypothesis 
              (Barlow, 1961) and sparse coding results in V1 (Olshausen & Field, 1996) which suggest biological systems seek minimal representations. 
              Feature Integration Theory (Treisman & Gelade, 1980) likewise suggests that selective modulation of simple detectors can explain 
              complex percepts, paralleling CAK's single-kernel modulation.
            </p>
          </section>
          
          <section className="paper-section">
            <h3>3. Method</h3>
            
            <h4>3.1 Empirical Motivation</h4>
            <p>
              Our initial approach followed established conditioning methods, using FiLM (Perez et al., 2018) with 24-dimensional control 
              vectors encoding categorical tags, continuous DSP parameters, and perceptual attributes. This failed catastrophically, not 
              merely poor results, but complete training collapse. FiLM's affine transformations expect simple categorical inputs and rely 
              on deep networks to achieve complex modulation, making them unsuitable for rich musical descriptors and small datasets.
            </p>
            <p>
              This failure revealed a crucial insight: complex control doesn't require complex modulation. Through ablation, we discovered 
              that our initially complex network consistently relied on a small subset of detected patterns, regardless of the control 
              vector's dimensionality. This suggested a radically different approach: instead of learning how to modulate based on complex 
              inputs, learn what patterns to detect, then simply scale them.
            </p>
            <p>
              The audit game naturally enforces this sparsity, the discriminator must verify control values, incentivizing the generator 
              to find clear, interpretable features. This led to CAK: a single learned detector whose output is scaled by a scalar control 
              value. The shift from 24-dimensional FiLM modulation to 1-dimensional scaling didn't reduce expressiveness, it revealed that 
              one well-learned pattern could create diverse effects through context-aware application.
            </p>
            
            <div className="figure-container">
              <img 
                src="/images/cak-training-dynamics.png" 
                alt="CAK vs FiLM Training Dynamics Comparison"
                className="paper-figure"
              />
              <p className="figure-caption">
                <strong>Figure 1:</strong> Training dynamics comparison between FiLM and CAK architectures. Both networks were trained under 
                identical conditions with no divergence mitigation strategies. FiLM-based conditioning with 24-dimensional control vectors 
                exhibits catastrophic instability, with discriminator loss exceeding 10^7 and complete training failure. In contrast, CAK 
                achieves stable training on the same dataset, enabling the discovery of meaningful patterns despite its ultimate convergence 
                to a single dominant feature. This demonstrates that architectural simplicity, not complexity, enables stable learning from 
                minimal personalized audio data.
              </p>
            </div>
            
            <h4>3.2 CAK Architecture</h4>
            <p>
              The core CAK operation implements a simple principle:
            </p>
            <div className="equation">
              {"$$\\text{output} = \\text{input} + (\\text{learned\\_pattern} \\times \\text{control})$$"}
            </div>
            <p>
              Formally, this becomes:
            </p>
            <div className="equation">
              {"$$y = x + (D(x) \\times c \\times \\sigma(c) \\times s)$$"}
            </div>
            <p>
              where:
            </p>
            <ul className="equation-list">
              <li>{"$x \\in \\mathbb{R}^{F \\times T}$"} is the input magnitude spectrogram</li>
              <li>{"$D: \\mathbb{R}^{F \\times T} \\rightarrow \\mathbb{R}^{F \\times T}$"} is a learned 3×3 convolutional detector (same padding, with bias)</li>
              <li>{"$c \\in \\mathbb{R}$"} is a per-example control scalar (randomly sampled during training) that broadcasts over the {"$F \\times T$"} dimensions</li>
              <li>{"$\\sigma(c) = \\text{sigmoid}((c - \\tau) \\times \\text{temp})$"} is a soft-gate function with:
                <ul>
                  <li>{"$\\tau \\in \\mathbb{R}$"}: threshold (0.3 in our experiments, ensuring control values below 0.3 produce minimal activation)</li>
                  <li>{"$\\text{temp} \\in \\mathbb{R}$"}: temperature parameter (2 → 20 linear ramp during training); lower values create gradual transitions, higher values approach a hard cutoff at {"$\\tau$"}</li>
                </ul>
              </li>
              <li>{"$s \\in \\mathbb{R}$"} is a learned scale parameter that correlates with effect intensity</li>
            </ul>
            <p>
              During training, each spectrogram is paired with a randomly sampled control value {"$c$"}. This scalar multiplies the detected 
              patterns {"$D(x)$"} element-wise across the entire spectrogram, creating a simple yet effective modulation mechanism. The audit 
              game ensures the network learns to detect patterns whose intensity scales meaningfully with {"$c$"}. We think of {"$c$"} as a 
              continuous modulation knob where higher values produce proportionally stronger effects, a direct, intuitive mapping rare in 
              neural audio systems. While we use random sampling in this work, the framework supports experiments using {"$c$"} to encode 
              semantic attributes.
            </p>
            <p>
              The soft-gate {"$\\sigma(c)$"} provides smooth onset based on the control value, which also directly scales the effect through 
              multiplication. We choose multiplication as it preserves sonic character while scaling intensity, aligning with human auditory 
              perception, which responds to amplitude ratios rather than absolute differences. This is arguably the simplest scaling solution 
              possible. Our dual use of {"$c$"} ensures both proportional intensity and gated activation, while the residual path guarantees 
              transparency at zero control.
            </p>
            
            <p><strong>Key Properties:</strong></p>
            <ol className="equation-list">
              <li><strong>Identity Preservation</strong>: At {"$c = 0$"}, the residual term {"$c \\cdot \\sigma(c)$"} is zero, so {"$y = x$"} exactly; {"$\\tau$"} and {"$\\text{temp}$"} shape only the onset for small {"$c$"}.</li>
              <li><strong>Additive Modulation</strong>: Unlike multiplicative attention, additive residual modulation preserves the original signal pathway, reducing risk of information loss compared to multiplicative gating.</li>
              <li><strong>Shared Detection</strong>: The shared detector {"$D$"} is updated jointly by generator and discriminator (critic) gradients; we did not freeze it in either branch.</li>
            </ol>
            
            <p>
              The soft-gate mechanism enables precise control. With threshold {"$\\tau = 0.3$"} and temperature annealing:
            </p>
            <ul className="equation-list">
              <li>Control values below {"$\\tau$"} produce minimal effect</li>
              <li>Smooth transition around {"$\\tau$"} prevents discontinuities</li>
              <li>High final temperature creates sharp but differentiable gating</li>
            </ul>
            <p>
              Unlike FiLM (per-channel affine {"$\\gamma, \\beta$"} conditioned on the input) or CondConv/dynamic conv (per-input kernels via routing), 
              CAK uses a single shared detector {"$D$"} and a user-supplied scalar {"$c$"} (with a soft gate) to scale a fixed residual.
            </p>
            
            <h4>3.3 The AuGAN Framework</h4>
            <p>
              Traditional GANs optimize {"$\\min_G \\max_D V(D,G)$"} where {"$G$"} tries to fool {"$D$"}. We reformulate this as AuGAN (Audit GAN), 
              where both networks cooperate to verify control application:
            </p>
            <p>
              <strong>Generator Objective</strong>: Apply transformations proportional to control value
            </p>
            <p>
              <strong>Discriminator Objective</strong>: Verify if the correct control amount was applied
            </p>
            <p>
              Crucially, both networks share the same detector {"$D$"}. This prevents the generator from learning arbitrary transformations, 
              any pattern it uses must help the discriminator verify control values. AuGAN's cooperative dynamics naturally promote:
            </p>
            <ol className="equation-list">
              <li><strong>Interpretable Features</strong>: Random patterns won't help verification</li>
              <li><strong>Proportional Application</strong>: The transformation strength must scale consistently with the control value</li>
              <li><strong>Smooth Control</strong>: The discriminator needs to distinguish nearby control values</li>
            </ol>
            <p>
              Unlike traditional adversarial training where deception drives learning, AuGAN's verification objective ensures that discovered 
              features correspond to meaningful transformations.
            </p>
            <p>
              We implement AuGAN using WGAN-GP with additional terms to enforce control compliance. Following WGAN convention, we refer to 
              the discriminator as the Critic (C):
            </p>
            <p><strong>Discriminator Loss:</strong></p>
            <div className="equation">
              {"$$L_C = -\\mathbb{E}[C(x_{\\text{real}}, c)] + \\mathbb{E}[C(x_{\\text{fake}}, c)] + \\lambda_{\\text{gp}} \\cdot \\text{GP} + \\lambda_{\\text{comp}} \\cdot \\mathbb{E}[V(x_{\\text{fake}}, c)]$$"}
            </div>
            <p><strong>Generator Loss:</strong></p>
            <div className="equation">
              {"$$L_G = -\\mathbb{E}[C(x_{\\text{fake}}, c)] + \\lambda_{\\text{comp}} \\cdot \\mathbb{E}[V(x_{\\text{fake}}, c)] + \\lambda_{\\text{recon}} \\cdot ||x_{\\text{fake}} - x_{\\text{real}}||_1 - \\lambda_{\\text{reg}} \\cdot \\mathbb{E}[\\log(\\varepsilon + \\text{mean}_{F,T}|D(x_{\\text{in}})|)]$$"}
            </div>
            <p>
              where:
            </p>
            <ul className="equation-list">
              <li>{"$x_{\\text{fake}} = G(x_{\\text{in}}, c)$"} is the generator output</li>
              <li>{"$C(\\cdot)$"} outputs realness score and violation {"$V(\\cdot, c)$"} for control verification</li>
              <li>{"$V(x, c) = |\\text{measured\\_texture}(x) - c|$"} where {"$\\text{measured\\_texture}(x) = \\text{mean}(D(x))$"} using the shared detector</li>
              <li>{"$\\text{GP}$"} is the gradient penalty for Lipschitz constraint computed on interpolations between {"$x_{\\text{real}}$"} and {"$x_{\\text{fake}}$"} with the same control value {"$c$"}</li>
              <li>{"$D(x_{\\text{in}})$"} in the regularization term refers to the shared detector patterns on input</li>
              <li>{"$\\varepsilon = 10^{-8}$"} for numerical stability</li>
              <li>{"$\\lambda_{\\text{gp}} = 10.0$"}, {"$\\lambda_{\\text{comp}} = 2.0$"}, {"$\\lambda_{\\text{recon}} = 5.0$"}, {"$\\lambda_{\\text{reg}} = 0.01$"}</li>
            </ul>
            <p>
              The violation term {"$V(x, c)$"} measures whether the applied effect matches the control value, enabling the audit verification. 
              Pattern regularization prevents detector collapse while reconstruction loss maintains spectral fidelity.
            </p>
          </section>
          
          <section className="paper-section">
            <h3>4. Experiments</h3>
            
            <h4>4.1 Experimental Setup</h4>
            <p>
              We designed our experiments to reflect realistic artistic workflows. Musicians and sound designers typically work with curated 
              personal collections rather than massive datasets. Our setup mirrors this reality:
            </p>
            <p>
              <strong>Dataset</strong>: 200 fifteen-second audio segments derived from the author's musical corpus, representing the scale of 
              material an artist might realistically collect and curate for a specific project. This includes varied timbral content mainly 
              centered around electronic and electroacoustic music composition: synthesized textures, field recordings, and acoustic instrumentation.
            </p>
            <p>
              <strong>Preprocessing</strong>: STFT with 2048-point FFT, 512 sample hop, 44.1 kHz sample rate, standard parameters for musical applications.
            </p>
            <p>
              <strong>Training</strong>: 100 epochs on Apple M4 (48GB unified memory), completing in approximately 2 hours. This wall-clock time 
              makes iterative experimentation feasible within a single studio session.
            </p>
            <p>
              All audio evaluations and demos use held out clips not seen during training, and the publicly released GUI applies a learned 
              effect to any user-provided audio.
            </p>
            
            <h4>4.2 Identity Preservation</h4>
            <p>
              The identity constraint at control value zero is a calibration mechanism, ensuring that our learned transformation maintains ideal 
              magnitude reconstruction when no modulation is desired. We enforce this through explicit identity pairs where input equals target 
              and control equals zero, penalizing any deviation. This prevents any neural spectral coloration in bypass mode and forces the 
              network to learn truly residual transformations.
            </p>
            <p>
              Identity preservation was verified on held out, diverse audio sources, with gate activation of 0.0025 on average at zero control 
              and perfect magnitude preservation (difference {"$< 10^{-9}$"}), confirming the soft-gate mechanism successfully guarantees 
              transparent pass-through. These results can be tested audibly in our GUI by simply processing a sample at a control value of 0.
            </p>
            
            <h4>4.3 Emergent Behavior and Kernel Analysis</h4>
            <p>
              The learned 3×3 detector kernel reveals how CAK achieves multiple effects through a single pattern. Figure 2 shows the learned 
              weights and their interpretable structure.
            </p>
            
            <div className="figure-container">
              <img 
                src="/images/kernel-analysis.png" 
                alt="Learned detector kernel analysis showing diagonal pattern and frequency response"
                className="paper-figure"
              />
              <p className="figure-caption">
                <strong>Figure 2:</strong> The learned detector kernel shows the diagonal pattern that underlies CAK's multi-effect behaviour. 
                During convolution with STFT magnitude, each weight indicates how strongly that time-frequency relationship contributes to 
                the output. The diagonal pattern (high weights at positions [0,2] and [2,2]) creates frequency-dependent temporal shifts, 
                higher frequencies are delayed relative to lower frequencies. This single kernel produces different perceptual effects 
                depending on input characteristics: signals with harmonic content experience phase shifts between frequency components, 
                while signals with temporal evolution undergo time-domain smearing. The frequency band response (right) shows emergent 
                selectivity, with stronger low-frequency weighting (0.115) despite no explicit frequency conditioning during training. 
                This demonstrates how the CAK framework discovers both spectral and temporal patterns directly from data.
              </p>
            </div>
            
            <p>
              Audio demonstrations showcasing these emergent behaviors are available at [URL]. Users can also experiment with their own audio 
              using our interactive GUI.
            </p>
            
            <h4>4.4 Training Efficiency and Computational Requirements</h4>
            <p>
              CAK's minimal architecture translates to exceptional computational efficiency:
            </p>
            <p>
              <strong>Training Metrics:</strong>
            </p>
            <ul className="equation-list">
              <li>Training time: ~2 hours for 100 epochs on Apple M4 (48GB)</li>
              <li>Dataset size: 200 samples (15 seconds each, {"$200 \\times 15\\text{s} = 50\\text{ min}$"} of audio)</li>
              <li>Batch size: 16</li>
              <li>Convergence: Stable training without divergence (Figure 3)</li>
            </ul>
            <p>
              <strong>Model Complexity:</strong>
            </p>
            <ul className="equation-list">
              <li>Learned effect: 11 parameters (9 kernel weights + 1 bias + 1 scale)</li>
              <li>Inference memory: ~44 bytes (critic only needed during training)</li>
              <li>No specialized hardware requirements</li>
            </ul>
            
            <div className="figure-container">
              <img 
                src="/images/cak-augan-training-history.png" 
                alt="CAK training dynamics over 100 epochs"
                className="paper-figure"
              />
              <p className="figure-caption">
                <strong>Figure 3:</strong> Training dynamics of CAK over 100 epochs using 200 15-second samples. (a) Generator and 
                discriminator losses show stable convergence. (b) Increasing Wasserstein distance indicates healthy adversarial learning. 
                (c) Decreasing audit violations demonstrate successful effect control learning. (d) Temperature annealing (orange) sharpens 
                the soft-gate while the scale parameter (brown) adapts to optimal effect strength.
              </p>
            </div>
          </section>
          
          <section className="paper-section">
            <h3>5. Future Work</h3>
            <p>
              While CAK demonstrates the power of minimal architectures, several directions merit exploration:
            </p>
            <p>
              <strong>Alternative Training Frameworks</strong>: Although we trained CAK using adversarial dynamics, the architecture itself 
              is training-agnostic. Investigating CAK within VAE frameworks or through direct supervised learning could reveal different 
              emergent behaviors and potentially simpler training procedures.
            </p>
            <p>
              <strong>Semantic Control</strong>: Our current approach learns effects from data without semantic labels. Incorporating dilated 
              convolutions or attention mechanisms could enable targeting specific perceptual qualities (e.g., "brightness," "warmth") while 
              maintaining our minimal parameter philosophy.
            </p>
            <p>
              <strong>Architectural Extensions</strong>: Stacking multiple CAK layers with varying kernel sizes could capture multi-scale 
              patterns. More intriguingly, frequency-band-specific CAK modules could enable surgical audio manipulation, applying different 
              learned transformations to isolated spectral regions and recombining them for complex, structured effects.
            </p>
            <p>
              <strong>Hybrid Approaches</strong>: Combining CAK with traditional DSP could leverage both learned and designed transformations. 
              A CAK module could learn residual transformations that complement existing effects, creating novel hybrid processors.
            </p>
            <p>
              <strong>Cross-Domain Applications</strong>: The principle of learning minimal patterns that interact with input characteristics 
              may extend beyond audio. Investigating CAK on image or video data could validate whether this emergence phenomenon generalizes 
              across modalities.
            </p>
            <p>
              These directions maintain our core insight: complex behaviors need not always require complex architectures.
            </p>
          </section>
          
          <section className="paper-section">
            <h3>6. Conclusion</h3>
            <p>
              The history of audio effects is a chronicle of human ingenuity, from the discovery of tape delay to the mathematical elegance 
              of convolution reverb. Each breakthrough required an engineer to first imagine, then formalize, a new way of transforming sound. 
              CAK inverts this paradigm: rather than encoding our understanding into algorithms, we let algorithms discover understanding from 
              sound itself.
            </p>
            <p>
              Our work demonstrates that embracing neural networks as simplification approximators can yield surprising power. With just 200 
              samples and 11 parameters, a single 3×3 kernel learns rich, adaptive audio transformations. By asking our model to find one 
              simple solution, we enabled it to discover complex effects that adapt to any input. The control mechanism itself embraces this 
              simplicity: a single scalar that could represent anything from effect intensity to semantic categories, limited only by the 
              user's imagination. This suggests a fundamentally different perspective on how deep learning achieves generalization, not through 
              scale, but through the interaction between network minimalism and input diversity.
            </p>
            <p>
              Beyond artistic applications, CAK suggests new research directions. The framework's flexibility invites exploration: what effects 
              emerge from different corpora? Can control values encode perceptual dimensions like 'warmth' or 'aggression'? Could multiple CAK 
              modules be combined for complex, multi-dimensional control? The simplicity of our approach makes such extensions straightforward 
              to explore.
            </p>
            <p>
              We hypothesize that neural networks, when presented with minimally expressive structures and appropriate training dynamics, 
              approximate complex behaviors by discovering simplicity rather than accumulating complexity. CAK validates this hypothesis, 
              opening new directions for both audio processing and neural architecture design.
            </p>
          </section>
          
        </div>
      </article>
    </section>
  )
}

export default CAK
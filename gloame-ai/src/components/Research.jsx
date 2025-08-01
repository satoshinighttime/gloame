import { Link } from 'react-router-dom'

function Research() {
  const papers = [
    {
      id: 1,
      title: "CAK: Emergent Audio Effects from Minimal Deep Learning",
      authors: "Austin Rockman",
      year: 2025,
      abstract: "We demonstrate that a single 3×3 convolutional kernel can produce emergent audio effects when trained on 200 samples from a personalized corpus. We achieve this through two key techniques: (1) Conditioning Aware Kernels (CAK), where output = input + (learned_pattern × control), with a soft-gate mechanism supporting identity preservation at zero control; and (2) AuGAN (Audit GAN), which reframes adversarial training from \"is this real?\" to \"did you apply the requested value?\"",
      slug: "cak"
    }
  ]

  return (
    <section className="research-section">
      <h2>research</h2>
      <div className="papers-list">
        {papers.map(paper => (
          <article key={paper.id} className="paper-item">
            <Link to={`/research/${paper.slug}`} className="paper-title-link">
              <h3 className="paper-title">{paper.title}</h3>
            </Link>
            <p className="paper-meta">{paper.authors} • {paper.year}</p>
            <p className="paper-abstract">{paper.abstract}</p>
            <Link to={`/research/${paper.slug}`} className="paper-link">
              read more →
            </Link>
          </article>
        ))}
      </div>
    </section>
  )
}

export default Research
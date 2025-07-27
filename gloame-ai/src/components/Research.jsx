import { Link } from 'react-router-dom'

function Research() {
  const papers = [
    {
      id: 1,
      title: "CAK: Conditioning-Aware Kernels for Personalized Audio Effects",
      authors: "Rockman & Garg",
      year: 2025,
      abstract: "We demonstrate that a single 3×3 convolutional kernel can produce multiple audio effects when trained on just 200 samples from a personalized corpus. This result challenges fundamental assumptions about neural audio processing complexity.",
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
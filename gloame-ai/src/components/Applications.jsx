function Applications() {
  const applications = [
    {
      id: 1,
      name: "CAK Audio Processor",
      description: "Interactive GUI for real-time audio processing using Conditioning-Aware Kernels",
      status: "coming soon"
    }
  ]

  return (
    <section className="applications-section">
      <h2>applications</h2>
      <div className="applications-grid">
        {applications.map(app => (
          <div key={app.id} className="application-card">
            <h3 className="app-name">{app.name}</h3>
            <p className="app-description">{app.description}</p>
            <div className="app-status">{app.status}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Applications
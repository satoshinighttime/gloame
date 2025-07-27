import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Home from './components/Home'
import About from './components/About'
import Research from './components/Research'
import CAK from './pages/research/CAK'
import Applications from './components/Applications'
import Audio from './components/Audio'
import './App.css'

function App() {
  const location = useLocation()
  
  const getActiveSection = () => {
    const path = location.pathname
    if (path === '/research' || path.startsWith('/research/')) return 'research'
    if (path === '/applications') return 'applications'
    if (path === '/audio') return 'audio'
    if (path === '/about') return 'about'
    return 'home'
  }

  return (
    <div className="app">
      <header className="header">
        <div className="container">
          <div className="header-content">
            <div className="logo">
              <Link 
                to="/"
                className="logo-text"
              >
                gloame.ai
              </Link>
            </div>
            <nav className="nav">
              <Link 
                to="/research"
                className={`nav-link ${getActiveSection() === 'research' ? 'active' : ''}`}
              >
                research
              </Link>
              <Link 
                to="/applications"
                className={`nav-link ${getActiveSection() === 'applications' ? 'active' : ''}`}
              >
                applications
              </Link>
              <Link 
                to="/about"
                className={`nav-link ${getActiveSection() === 'about' ? 'active' : ''}`}
              >
                about
              </Link>
            </nav>
          </div>
        </div>
      </header>

      <main className="main">
        <div className="container">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/research" element={<Research />} />
            <Route path="/research/cak" element={<CAK />} />
            <Route path="/applications" element={<Applications />} />
            <Route path="/audio" element={<Audio />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </div>
      </main>

      <footer className="footer">
        <div className="container">
          <p className="footer-text">© 2025 gloame.ai</p>
        </div>
      </footer>
    </div>
  )
}

export default App
const express = require('express');
const path = require('path');
const crypto = require('crypto');

const app = express();
const PORT = process.env.PORT || 8080;

// Secret for sessions
const SESSION_SECRET = process.env.SESSION_SECRET || crypto.randomBytes(32).toString('hex');
const PASSWORD = 'perfectdark';

// Simple in-memory session store
const sessions = new Map();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Session middleware
app.use((req, res, next) => {
  const sessionId = req.headers.cookie?.split(';')
    .find(c => c.trim().startsWith('session='))
    ?.split('=')[1];
  
  req.session = sessions.get(sessionId) || null;
  next();
});

// Authentication endpoint
app.post('/auth', (req, res) => {
  const { password } = req.body;
  
  if (password === PASSWORD) {
    const sessionId = crypto.randomBytes(32).toString('hex');
    sessions.set(sessionId, { authenticated: true, timestamp: Date.now() });
    
    res.setHeader('Set-Cookie', `session=${sessionId}; HttpOnly; SameSite=Strict; Max-Age=${24 * 60 * 60}; Path=/`);
    res.json({ success: true });
  } else {
    res.status(401).json({ success: false });
  }
});

// Logout endpoint
app.post('/logout', (req, res) => {
  const sessionId = req.headers.cookie?.split(';')
    .find(c => c.trim().startsWith('session='))
    ?.split('=')[1];
  
  if (sessionId) {
    sessions.delete(sessionId);
  }
  
  res.clearCookie('session');
  res.json({ success: true });
});

// Authentication check middleware
const requireAuth = (req, res, next) => {
  if (!req.session?.authenticated) {
    res.sendFile(path.join(__dirname, 'password.html'));
  } else {
    next();
  }
};

// Serve static files only to authenticated users
app.use('/assets', requireAuth, express.static(path.join(__dirname, 'dist/assets')));
app.use('/audio', requireAuth, express.static(path.join(__dirname, 'dist/audio')));
app.use('/images', requireAuth, express.static(path.join(__dirname, 'dist/images')));

// Serve vite.svg without auth for favicon
app.get('/vite.svg', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'vite.svg'));
});

// Serve index.html for authenticated users
app.get('*', requireAuth, (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

// Clean up old sessions periodically
setInterval(() => {
  const now = Date.now();
  for (const [id, session] of sessions.entries()) {
    if (now - session.timestamp > 24 * 60 * 60 * 1000) {
      sessions.delete(id);
    }
  }
}, 60 * 60 * 1000); // Every hour

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
import { useState, useRef, useEffect } from 'react'

function Audio() {
  const [selectedCategory, setSelectedCategory] = useState('all')
  const [currentlyPlaying, setCurrentlyPlaying] = useState(null)
  const [playbackProgress, setPlaybackProgress] = useState({})
  const [durations, setDurations] = useState({})
  const audioRefs = useRef({})

  // Define the audio corpus structure
  const audioCorpus = {
    'ex machina': [
      { name: 'ex_machina_1', file: 'EX_MACHINA_1.wav' },
      { name: 'ex_machina_2', file: 'EX_MACHINA_2.wav' },
      { name: 'ex_machina_3', file: 'EX_MACHINA_3.wav' },
      { name: 'ex_machina_4', file: 'EX_MACHINA_4.wav' },
      { name: 'ex_machina_5', file: 'EX_MACHINA_5.wav' },
      { name: 'ex_machina_6', file: 'EX_MACHINA_6.wav' },
      { name: 'ex_machina_7', file: 'EX_MACHINA_7.wav' },
      { name: 'ex_machina_8', file: 'EX_MACHINA_8.wav' },
    ],
    'facehugger': [
      { name: 'facehugger_1', file: 'FACEHUGGER_1.wav' },
      { name: 'facehugger_2', file: 'FACEHUGGER_2.wav' },
      { name: 'facehugger_3', file: 'FACEHUGGER_3.wav' },
    ],
    'fields': [
      { name: 'field_1_[compel]', file: 'FIELD_1 [COMPEL].wav' },
      { name: 'field_2_[forest]', file: 'FIELD_2 [FOREST].wav' },
    ],
    'forerunner': [
      { name: 'forerunner_1', file: 'FORERUNNER_1.wav' },
      { name: 'forerunner_2', file: 'FORERUNNER_2.wav' },
      { name: 'forerunner_3', file: 'FORERUNNER_3.wav' },
      { name: 'forerunner_4', file: 'FORERUNNER_4.wav' },
      { name: 'forerunner_5', file: 'FORERUNNER_5.wav' },
      { name: 'forerunner_6_[persona_1]', file: 'FORERUNNER_6_[PERSONA_1].wav' },
      { name: 'forerunner_7_[persona_2]', file: 'FORERUNNER_7_[PERSONA_2].wav' },
      { name: 'forerunner_8_[persona_3]', file: 'FORERUNNER_8_[PERSONA_3].wav' },
      { name: 'forerunner_9_[persona_4]', file: 'FORERUNNER_9_[PERSONA_4].wav' },
      { name: 'forerunner_10', file: 'FORERUNNER_10.wav' },
      { name: 'forerunner_17', file: 'FORERUNNER_17.wav' },
      { name: 'forerunner_18', file: 'FORERUNNER_18.wav' },
      { name: 'forerunner_19', file: 'FORERUNNER_19.wav' },
      { name: 'forerunner_20', file: 'FORERUNNER_20.wav' },
    ],
    'latent space': [
      { name: 'latent_space_1', file: 'LATENT_SPACE_1.wav' },
      { name: 'latent_space_2', file: 'LATENT_SPACE_2.wav' },
      { name: 'latent_space_3', file: 'LATENT_SPACE_3.wav' },
      { name: 'latent_space_4', file: 'LATENT_SPACE_4.wav' },
      { name: 'latent_space_5', file: 'LATENT_SPACE_5.wav' },
      { name: 'latent_space_6', file: 'LATENT_SPACE_6.wav' },
      { name: 'latent_space_7', file: 'LATENT_SPACE_7.wav' },
      { name: 'latent_space_8', file: 'LATENT_SPACE_8.wav' },
    ],
    'lux': [
      { name: 'lux_1', file: 'LUX_1.wav' },
      { name: 'lux_2', file: 'LUX_2.wav' },
      { name: 'lux_3', file: 'LUX_3.wav' },
      { name: 'lux_4', file: 'LUX_4.wav' },
      { name: 'lux_5', file: 'LUX_5.wav' },
      { name: 'lux_6', file: 'LUX_6.wav' },
      { name: 'lux_7', file: 'LUX_7.wav' },
      { name: 'lux_8', file: 'LUX_8.wav' },
      { name: 'lux_9', file: 'LUX_9.wav' },
      { name: 'lux_10', file: 'LUX_10.wav' },
      { name: 'lux_11', file: 'LUX_11.wav' },
      { name: 'lux_12', file: 'LUX_12.wav' },
      { name: 'lux_13', file: 'LUX_13.wav' },
      { name: 'lux_14', file: 'LUX_14.wav' },
      { name: 'lux_15', file: 'LUX_15.wav' },
    ],
    'microexpressions': [
      { name: 'microexpressions_1', file: 'MICROEXPRESSIONS_1.wav' },
    ],
    'neural': [
      { name: 'neural_1', file: 'NEURAL_1.wav' },
      { name: 'neural_2', file: 'NEURAL_2.wav' },
      { name: 'neural_3', file: 'NEURAL_3.wav' },
      { name: 'neural_4', file: 'NEURAL_4.wav' },
      { name: 'neural_5', file: 'NEURAL_5.wav' },
      { name: 'neural_6', file: 'NEURAL_6.wav' },
      { name: 'neural_7', file: 'NEURAL_7.wav' },
      { name: 'neural_8', file: 'NEURAL_8.wav' },
    ],
    'n dimensional': [
      { name: 'n-dimensional_1', file: 'N-DIMENSIONAL_1.wav' },
    ],
    'radical': [
      { name: 'radical_1', file: 'RADICAL_1.wav' },
    ],
  }

  const handlePlay = (category, fileName) => {
    const audioId = `${category}/${fileName}`
    
    // Stop currently playing audio if any
    if (currentlyPlaying && currentlyPlaying !== audioId) {
      const currentAudio = audioRefs.current[currentlyPlaying]
      if (currentAudio) {
        currentAudio.pause()
        currentAudio.currentTime = 0
      }
    }

    const audio = audioRefs.current[audioId]
    if (audio) {
      if (currentlyPlaying === audioId && !audio.paused) {
        audio.pause()
        setCurrentlyPlaying(null)
      } else {
        audio.play()
        setCurrentlyPlaying(audioId)
      }
    }
  }

  const handleTimeUpdate = (category, fileName) => {
    const audioId = `${category}/${fileName}`
    const audio = audioRefs.current[audioId]
    if (audio) {
      setPlaybackProgress(prev => ({
        ...prev,
        [audioId]: (audio.currentTime / audio.duration) * 100
      }))
    }
  }

  const handleEnded = (category, fileName) => {
    const audioId = `${category}/${fileName}`
    setCurrentlyPlaying(null)
    setPlaybackProgress(prev => ({
      ...prev,
      [audioId]: 0
    }))
  }

  const handleLoadedMetadata = (category, fileName) => {
    const audioId = `${category}/${fileName}`
    const audio = audioRefs.current[audioId]
    if (audio && audio.duration) {
      setDurations(prev => ({
        ...prev,
        [audioId]: formatDuration(audio.duration)
      }))
    }
  }

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`
  }

  const handleProgressClick = (e, category, fileName) => {
    const audioId = `${category}/${fileName}`
    const audio = audioRefs.current[audioId]
    if (audio) {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const clickedValue = (x / rect.width) * audio.duration
      audio.currentTime = clickedValue
    }
  }

  const getFilteredCategories = () => {
    if (selectedCategory === 'all') {
      return Object.keys(audioCorpus)
    }
    return [selectedCategory]
  }

  return (
    <section className="audio-section">
      <h2>audio</h2>
      
      <div className="audio-controls">
        <div className="category-filter">
          <button 
            className={selectedCategory === 'all' ? 'active' : ''}
            onClick={() => setSelectedCategory('all')}
          >
            all
          </button>
          {Object.keys(audioCorpus).map(category => (
            <button
              key={category}
              className={selectedCategory === category ? 'active' : ''}
              onClick={() => setSelectedCategory(category)}
            >
              {category}
            </button>
          ))}
        </div>
      </div>

      <div className="audio-library">
        {getFilteredCategories().map(category => (
          <div key={category} className="audio-category">
            <h3>{category}</h3>
            <div className="audio-table">
              {audioCorpus[category].map((audio, index) => {
                const categoryKey = category.toUpperCase().replace(' ', '_')
                const audioId = `${category}/${audio.file}`
                const isPlaying = currentlyPlaying === audioId
                const progress = playbackProgress[audioId] || 0
                const duration = durations[audioId] || '00:00'
                const trackNumber = String(index + 1).padStart(2, '0')

                return (
                  <div key={audio.file} className="audio-row">
                    <span className="track-number">{trackNumber}</span>
                    <span className="track-name">{audio.name}</span>
                    <span className="track-duration">{duration}</span>
                    <button
                      className={`play-button ${isPlaying ? 'playing' : ''}`}
                      onClick={() => handlePlay(category, audio.file)}
                    >
                      {isPlaying ? '▶' : '▶'}
                    </button>
                    {isPlaying && (
                      <div className="progress-bar-container">
                        <div 
                          className="progress-bar"
                          onClick={(e) => handleProgressClick(e, category, audio.file)}
                        >
                          <div 
                            className="progress-fill"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                    <audio
                      ref={el => audioRefs.current[audioId] = el}
                      src={`/audio/${categoryKey}/${audio.file}`}
                      onTimeUpdate={() => handleTimeUpdate(category, audio.file)}
                      onEnded={() => handleEnded(category, audio.file)}
                      onLoadedMetadata={() => handleLoadedMetadata(category, audio.file)}
                    />
                  </div>
                )
              })}
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default Audio
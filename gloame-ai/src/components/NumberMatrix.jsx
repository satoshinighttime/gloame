import { useEffect, useRef, memo, useMemo, useCallback } from 'react'

const NumberItem = memo(({ value, baseOpacity, x, y, id }) => {
  const ref = useRef(null)
  
  useEffect(() => {
    if (ref.current) {
      ref.current.style.setProperty('--base-opacity', baseOpacity)
    }
  }, [baseOpacity])
  
  return (
    <span
      ref={ref}
      className="matrix-number"
      data-id={id}
      data-x={x * 30 + 15}
      data-y={y * 30 + 15}
    >
      {value}
    </span>
  )
})

function NumberMatrix() {
  const containerRef = useRef(null)
  const mouseRef = useRef({ x: -1000, y: -1000 })
  const rafRef = useRef(null)
  const lastUpdateRef = useRef(0)

  // Generate a large fixed grid once
  const numbers = useMemo(() => {
    const rows = 60 // Enough for most screen sizes
    const cols = 50
    const nums = []
    
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        nums.push({
          id: `${i}-${j}`,
          value: Math.floor(Math.random() * 10),
          x: j,
          y: i,
          baseOpacity: 0.08 + Math.random() * 0.12
        })
      }
    }
    
    return nums
  }, [])

  const updateMouseEffects = useCallback(() => {
    if (!containerRef.current) return
    
    const mouseX = mouseRef.current.x
    const mouseY = mouseRef.current.y
    const maxDistance = 150
    
    // Get only visible numbers
    const numbers = containerRef.current.querySelectorAll('.matrix-number')
    
    for (let i = 0; i < numbers.length; i++) {
      const num = numbers[i]
      const numX = parseFloat(num.dataset.x)
      const numY = parseFloat(num.dataset.y)
      
      // Quick boundary check
      if (Math.abs(mouseX - numX) > maxDistance || Math.abs(mouseY - numY) > maxDistance) {
        if (num.dataset.affected === 'true') {
          num.style.transform = ''
          num.style.opacity = ''
          num.dataset.affected = 'false'
        }
        continue
      }
      
      const dx = mouseX - numX
      const dy = mouseY - numY
      const distance = Math.sqrt(dx * dx + dy * dy)
      
      if (distance < maxDistance) {
        const influence = 1 - distance / maxDistance
        const offsetX = -dx * influence * 0.3
        const offsetY = -dy * influence * 0.3
        const scale = 1 + influence * 0.5
        const baseOpacity = parseFloat(getComputedStyle(num).getPropertyValue('--base-opacity'))
        
        num.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`
        num.style.opacity = baseOpacity + influence * 0.6
        num.dataset.affected = 'true'
      } else if (num.dataset.affected === 'true') {
        num.style.transform = ''
        num.style.opacity = ''
        num.dataset.affected = 'false'
      }
    }
  }, [])

  const handleMouseMove = useCallback((e) => {
    if (!containerRef.current) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const newX = e.clientX - rect.left
    const newY = e.clientY - rect.top
    
    // Only update if moved significantly
    if (Math.abs(newX - mouseRef.current.x) > 2 || Math.abs(newY - mouseRef.current.y) > 2) {
      mouseRef.current.x = newX
      mouseRef.current.y = newY
      
      if (!rafRef.current) {
        rafRef.current = requestAnimationFrame(() => {
          updateMouseEffects()
          rafRef.current = null
        })
      }
    }
  }, [updateMouseEffects])

  const handleMouseLeave = useCallback(() => {
    mouseRef.current = { x: -1000, y: -1000 }
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    // Reset all affected numbers
    if (containerRef.current) {
      const numbers = containerRef.current.querySelectorAll('.matrix-number[data-affected="true"]')
      numbers.forEach(num => {
        num.style.transform = ''
        num.style.opacity = ''
        num.dataset.affected = 'false'
      })
    }
  }, [])

  useEffect(() => {
    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
      }
    }
  }, [])

  return (
    <div 
      className="number-matrix"
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      {numbers.map(num => (
        <NumberItem 
          key={num.id}
          id={num.id}
          value={num.value}
          baseOpacity={num.baseOpacity}
          x={num.x}
          y={num.y}
        />
      ))}
    </div>
  )
}

export default NumberMatrix
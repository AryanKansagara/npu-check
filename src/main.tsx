import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles.css'
import { App } from './App.tsx'
import { loadProjects } from './store.ts'
import { initSync } from './sync.ts'

void loadProjects().then(() => initSync())
createRoot(document.getElementById('root')!).render(<StrictMode><App /></StrictMode>)

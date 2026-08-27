import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import UploadPage from './pages/UploadPage'
import HistoryPage from './pages/HistoryPage'
import './App.css'

function App() {
  return (
    <BrowserRouter>
      <div className="app">
        <header className="app-header">
          <h1>Steganography Detection</h1>
          <nav>
            <NavLink to="/" end>Upload</NavLink>
            <NavLink to="/history">History</NavLink>
          </nav>
        </header>
        <main className="app-main">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/history" element={<HistoryPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

export default App

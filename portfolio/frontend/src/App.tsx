
import MainPage from './pages/mainPage.tsx'
import DownloadsPage from './pages/downloadsPage.tsx'
import CompetencesPage from './pages/competencesPage.tsx'
import CentreInteret from './pages/loisirs.tsx'
import RocketDemo from './pages/rocketDemo.tsx'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'


function App() {

  return (
    <Router basename="/ali-portfolio">
      <Routes>
        <Route path="/" element={<Navigate to="/home" replace />} />
        {/* <Route path='/educationExperience' element={<EducationExperience />} /> */}
        <Route path="/home" element={<MainPage />} />
        <Route path='/downloadsPage' element={<DownloadsPage />} />
        <Route path='/competencesPage' element={<CompetencesPage />} />
        <Route path='/loisirs' element={<CentreInteret />} />
        <Route path='/rocketDemo' element={<RocketDemo />} />
      </Routes>
    </Router>
  )
}

export default App

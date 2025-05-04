import { Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import TryLisaPage from './pages/TryLisaPage';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

function App() {
  return (
    <div className="flex flex-col min-h-screen">
      <Navbar />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/try-lisa" element={<TryLisaPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
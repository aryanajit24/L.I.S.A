import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X } from 'lucide-react';

const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const location = useLocation();

  useEffect(() => {
    const handleScroll = () => {
      if (window.scrollY > 10) {
        setIsScrolled(true);
      } else {
        setIsScrolled(false);
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  // Close mobile menu when changing routes
  useEffect(() => {
    setIsOpen(false);
  }, [location.pathname]);

  return (
    <nav 
      className={`fixed top-0 w-full z-50 transition-all duration-300 ${
        isScrolled ? 'bg-neutral-950/90 backdrop-blur-md border-b border-neutral-800' : 'bg-transparent'
      }`}
    >
      <div className="container-custom">
        <div className="flex justify-between items-center py-4">
          <Link 
            to="/" 
            className="flex items-center space-x-2"
          >
            <div className="w-8 h-8">
              <svg width="32" height="32" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect width="40" height="40" rx="8" fill="currentColor" />
                <path d="M8 10H32V12H8V10Z" fill="white" />
                <path d="M8 18H20V20H8V18Z" fill="white" />
                <path d="M8 26H32V28H8V26Z" fill="white" />
                <path d="M24 18H32V20H24V18Z" fill="#7dd3fc" />
                <circle cx="30" cy="19" r="6" fill="black" stroke="#d946ef" strokeWidth="2" />
              </svg>
            </div>
            <span className="text-xl font-medium">L.I.S.A</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-10">
            <Link to="/" className="text-sm hover:text-primary-300 transition-colors">
              Home
            </Link>
            <a href="#features" className="text-sm hover:text-primary-300 transition-colors">
              Features
            </a>
            <a href="#about" className="text-sm hover:text-primary-300 transition-colors">
              About
            </a>
            <Link to="/try-lisa" className="btn-primary">
              Try L.I.S.A
            </Link>
          </div>

          {/* Mobile Navigation Toggle */}
          <button 
            className="md:hidden text-white"
            onClick={() => setIsOpen(!isOpen)}
            aria-label="Toggle menu"
          >
            {isOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Mobile Navigation Menu */}
        {isOpen && (
          <div className="md:hidden absolute top-16 left-0 right-0 bg-neutral-900 border-b border-neutral-800 py-4">
            <div className="flex flex-col space-y-4 px-4">
              <Link to="/" className="text-base py-2 hover:text-primary-300 transition-colors">
                Home
              </Link>
              <a href="#features" className="text-base py-2 hover:text-primary-300 transition-colors">
                Features
              </a>
              <a href="#about" className="text-base py-2 hover:text-primary-300 transition-colors">
                About
              </a>
              <Link to="/try-lisa" className="btn-primary text-center">
                Try L.I.S.A
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;
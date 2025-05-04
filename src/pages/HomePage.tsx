import { useEffect } from 'react';
import HeroSection from '../components/HeroSection';
import FeaturesSection from '../components/FeaturesSection';
import AboutSection from '../components/AboutSection';
import DemoSection from '../components/DemoSection';
import SmoothScroll from '../components/SmoothScroll';

const HomePage = () => {
  useEffect(() => {
    // Update the title when component mounts
    document.title = 'L.I.S.A - Local Integrated Systems Architecture';
    
    // Scroll to top when component mounts
    window.scrollTo(0, 0);
  }, []);

  return (
    <SmoothScroll>
      <div className="overflow-hidden">
        <HeroSection />
        <FeaturesSection />
        <AboutSection />
        <DemoSection />
      </div>
    </SmoothScroll>
  );
};

export default HomePage;
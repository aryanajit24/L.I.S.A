import { useEffect, useRef } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import AnimatedThreeDModel from './AnimatedThreeDModel';

const HeroSection = () => {
  const heroRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleScroll = () => {
      if (!heroRef.current) return;
      
      const scrollTop = window.scrollY;
      const opacity = 1 - Math.min(1, scrollTop / 700);
      
      if (heroRef.current) {
        heroRef.current.style.opacity = String(opacity);
        heroRef.current.style.transform = `translateY(${scrollTop * 0.3}px)`;
      }
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div 
      ref={heroRef}
      className="relative min-h-screen flex items-center justify-center overflow-hidden"
    >
      {/* Clean, modern background */}
      <div className="absolute inset-0">
        <div className="absolute inset-0 bg-neutral-950" />
        <div className="absolute inset-0 bg-gradient-to-t from-primary-500/5 via-transparent to-transparent" />
        <div className="absolute inset-0 bg-gradient-to-r from-accent-500/5 via-transparent to-transparent" />
      </div>
      
      {/* Subtle animated gradient */}
      <div className="absolute inset-0">
        <div className="absolute -inset-[100%] bg-gradient-to-r from-transparent via-primary-500/5 to-transparent animate-[gradient_8s_linear_infinite] blur-3xl" />
        <div className="absolute -inset-[100%] bg-gradient-to-r from-transparent via-accent-500/5 to-transparent animate-[gradient_12s_linear_infinite] blur-3xl" />
      </div>
      
      {/* 3D Model with reduced opacity */}
      <div className="absolute inset-0 z-0 opacity-60">
        <AnimatedThreeDModel />
      </div>
      
      {/* Content */}
      <div className="container-custom relative z-10 pt-24 md:pt-0">
        <div className="text-center max-w-4xl mx-auto px-4">
          <motion.div 
            className="mb-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold tracking-tight">
              Meet <span className="relative inline-block">
                <span className="gradient-text">L.I.S.A</span>
                <div className="absolute -inset-2 bg-gradient-to-r from-primary-500/20 to-accent-500/20 blur-xl -z-10" />
              </span>
            </h1>
          </motion.div>
          
          <motion.p 
            className="text-xl md:text-2xl text-neutral-200 mb-10 max-w-3xl mx-auto"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            Local Integrated Systems Architecture —
            <br />A revolutionary AI model for processing complex data
          </motion.p>
          
          <motion.div 
            className="flex flex-col sm:flex-row items-center justify-center space-y-4 sm:space-y-0 sm:space-x-4"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            <Link to="/try-lisa" className="btn-primary">
              Try L.I.S.A Now
            </Link>
            <Link to="/#features" className="btn-secondary">
              Learn More
            </Link>
          </motion.div>
        </div>
      </div>
      
      {/* Scroll indicator */}
      <motion.div 
        className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1, delay: 1.2 }}
      >
        <div className="flex flex-col items-center">
          <span className="text-sm text-neutral-400 mb-2">Scroll to explore</span>
          <div className="w-6 h-10 border-2 border-neutral-400 rounded-full flex justify-center">
            <div className="w-1 h-2 bg-white rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default HeroSection;
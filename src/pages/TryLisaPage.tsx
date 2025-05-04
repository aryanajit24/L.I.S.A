import { useEffect } from 'react';
import { motion } from 'framer-motion';
import ChatInterface from '../components/ChatInterface';

const TryLisaPage = () => {
  useEffect(() => {
    // Update the title when component mounts
    document.title = 'Try L.I.S.A - Interactive Demo';
    
    // Scroll to top when component mounts
    window.scrollTo(0, 0);
  }, []);

  return (
    <div className="pt-24 pb-16 min-h-screen bg-neutral-950">
      <div className="container-custom">
        <div className="max-w-4xl mx-auto">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center mb-12"
          >
            <h1 className="text-3xl md:text-4xl font-bold mb-4">
              Try <span className="gradient-text">L.I.S.A</span> Now
            </h1>
            <p className="text-neutral-300 max-w-2xl mx-auto">
              Upload files, images, or simply chat with L.I.S.A to experience the power of Local Integrated Systems Architecture.
            </p>
          </motion.div>
          
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <ChatInterface />
          </motion.div>
          
          <motion.div 
            className="mt-8 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <p className="text-neutral-500 text-sm">
              Note: This is a demo interface. In the full version, you would have access to complete processing capabilities.
            </p>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default TryLisaPage;
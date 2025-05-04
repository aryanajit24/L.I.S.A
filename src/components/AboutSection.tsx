import { motion } from 'framer-motion';

const AboutSection = () => {
  return (
    <section id="about" className="py-24 bg-neutral-950 relative">
      {/* Background gradient element */}
      <div className="absolute top-0 left-1/2 transform -translate-x-1/2 w-full max-w-6xl h-full opacity-30">
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-primary-500/20 rounded-full filter blur-3xl"></div>
        <div className="absolute top-1/3 right-1/4 w-64 h-64 bg-accent-500/20 rounded-full filter blur-3xl"></div>
      </div>

      <div className="container-custom relative z-10">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
          {/* Left column - Description */}
          <div>
            <motion.span 
              className="text-sm uppercase tracking-wider text-primary-400"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5 }}
            >
              About
            </motion.span>
            <motion.h2 
              className="text-3xl md:text-4xl font-bold mt-2 mb-6"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              The Story Behind L.I.S.A
            </motion.h2>
            
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <p className="text-neutral-300 mb-4">
                L.I.S.A (Local Integrated Systems Architecture) represents a breakthrough in AI technology that operates with a focus on local processing and data privacy.
              </p>
              
              <p className="text-neutral-400 mb-4">
                Developed by a team of AI researchers and engineers, L.I.S.A aims to bring powerful AI capabilities to your devices without compromising on privacy or performance.
              </p>
              
              <p className="text-neutral-400 mb-4">
                Our mission is to make advanced AI accessible while preserving user privacy and data sovereignty. L.I.S.A represents the next generation of AI systems that work for you, not the other way around.
              </p>
              
              <div className="mt-8 space-y-4">
                <div className="flex items-center">
                  <div className="w-16 h-1 bg-primary-500"></div>
                  <div className="w-full h-px bg-neutral-800"></div>
                </div>
                
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-4xl font-bold gradient-text">99%</p>
                    <p className="text-sm text-neutral-400 mt-1">Privacy Preservation</p>
                  </div>
                  <div>
                    <p className="text-4xl font-bold gradient-text">5x</p>
                    <p className="text-sm text-neutral-400 mt-1">Faster Than Alternatives</p>
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
          
          {/* Right column - Image */}
          <motion.div
            className="rounded-2xl overflow-hidden shadow-xl"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7, delay: 0.3 }}
          >
            <img 
              src="https://images.pexels.com/photos/3048527/pexels-photo-3048527.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" 
              alt="AI Technology Visualization" 
              className="w-full h-auto rounded-2xl object-cover"
            />
          </motion.div>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
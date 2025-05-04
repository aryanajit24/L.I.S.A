import { useState } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const DemoSection = () => {
  const [isHovered, setIsHovered] = useState(false);
  
  return (
    <section className="py-24 bg-gradient-to-b from-neutral-900 to-neutral-950">
      <div className="container-custom">
        <div className="bg-neutral-800 rounded-3xl overflow-hidden">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="p-8 lg:p-12 flex flex-col justify-center">
              <motion.h2 
                className="text-3xl md:text-4xl font-bold mb-6"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5 }}
              >
                Experience L.I.S.A In Action
              </motion.h2>
              
              <motion.p 
                className="text-neutral-300 mb-8"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.1 }}
              >
                Upload files, images, and documents to see how L.I.S.A processes and understands complex data. Get intelligent insights and responses in real-time.
              </motion.p>
              
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: 0.2 }}
              >
                <Link 
                  to="/try-lisa" 
                  className="group inline-flex items-center text-white"
                  onMouseEnter={() => setIsHovered(true)}
                  onMouseLeave={() => setIsHovered(false)}
                >
                  <span className="text-lg font-medium">Try L.I.S.A Now</span>
                  <span className={`ml-2 transform transition-transform duration-300 ${isHovered ? 'translate-x-2' : ''}`}>
                    <ArrowRight size={20} />
                  </span>
                </Link>
              </motion.div>
            </div>
            
            <motion.div 
              className="relative h-full min-h-[300px] lg:min-h-0"
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.7, delay: 0.3 }}
            >
              <img 
                src="https://images.pexels.com/photos/8386440/pexels-photo-8386440.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2" 
                alt="L.I.S.A Demo" 
                className="absolute inset-0 w-full h-full object-cover"
              />
              <div className="absolute inset-0 bg-gradient-to-r from-neutral-800/90 to-transparent" />
            </motion.div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default DemoSection;
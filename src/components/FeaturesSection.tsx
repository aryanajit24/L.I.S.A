import { motion } from 'framer-motion';
import { Brain, Database, PenTool, Shield, Zap, RefreshCw } from 'lucide-react';

const features = [
  {
    icon: <Brain size={24} />,
    title: 'Advanced AI',
    description: 'Cutting-edge language model fine-tuned for precise responses and deep understanding.'
  },
  {
    icon: <Database size={24} />,
    title: 'Local Processing',
    description: 'Process sensitive data directly on your device without sending it to external servers.'
  },
  {
    icon: <PenTool size={24} />,
    title: 'Creative Assistant',
    description: 'Generate creative content, from marketing copy to imaginative stories.'
  },
  {
    icon: <Shield size={24} />,
    title: 'Privacy Focused',
    description: 'Keep your data private with local processing and encrypted communication.'
  },
  {
    icon: <Zap size={24} />,
    title: 'Lightning Fast',
    description: 'Optimized for performance with rapid response times even for complex queries.'
  },
  {
    icon: <RefreshCw size={24} />,
    title: 'Continuous Learning',
    description: 'Improves over time through usage patterns and feedback systems.'
  }
];

const FeaturesSection = () => {
  return (
    <section id="features" className="py-24 bg-neutral-900">
      <div className="container-custom">
        <div className="text-center mb-16">
          <motion.span 
            className="text-sm uppercase tracking-wider text-primary-400"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5 }}
          >
            Features
          </motion.span>
          <motion.h2 
            className="text-3xl md:text-4xl font-bold mt-2"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            What Makes L.I.S.A Special
          </motion.h2>
          <motion.div 
            className="w-24 h-1 bg-primary-500 mx-auto mt-6"
            initial={{ opacity: 0, width: 0 }}
            whileInView={{ opacity: 1, width: 96 }}
            viewport={{ once: true }}
            transition={{ duration: 0.5, delay: 0.2 }}
          ></motion.div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {features.map((feature, index) => (
            <motion.div 
              key={index}
              className="bg-neutral-800/50 rounded-2xl p-6 border border-neutral-700 hover:border-primary-500 transition-all duration-300"
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: 0.1 * index }}
            >
              <div className="bg-neutral-700/50 w-12 h-12 rounded-lg flex items-center justify-center text-primary-400 mb-4">
                {feature.icon}
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-neutral-400">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
};

export default FeaturesSection;
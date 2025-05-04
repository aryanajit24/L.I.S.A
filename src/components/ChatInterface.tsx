import { useState, useRef, useEffect } from 'react';
import { Send, Paperclip, XCircle, RotateCw } from 'lucide-react';
import { motion } from 'framer-motion';

type Message = {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
};

type FileUpload = {
  id: string;
  name: string;
  type: string;
  size: number;
};

const ChatInterface = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      type: 'assistant',
      content: 'Hello! I\'m L.I.S.A, your Local Integrated Systems Architecture assistant. How can I help you today?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [files, setFiles] = useState<FileUpload[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll to the bottom of messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (input.trim() === '' && files.length === 0) return;
    
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsProcessing(true);
    
    // Simulate response (in a real app, you'd call your API here)
    setTimeout(() => {
      let responseContent = `I've received your message${files.length > 0 ? ' and ' + files.length + ' file(s)' : ''}. As a demo, I'm showing a simulated response. In the full version, I would process your query and provide an intelligent response based on all information provided.`;
      
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'assistant',
        content: responseContent,
        timestamp: new Date()
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      setIsProcessing(false);
      setFiles([]);
    }, 1500);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map(file => ({
        id: Math.random().toString(36).substring(7),
        name: file.name,
        type: file.type,
        size: file.size
      }));
      
      setFiles(prev => [...prev, ...newFiles]);
    }
  };

  const removeFile = (id: string) => {
    setFiles(files.filter(file => file.id !== id));
  };

  const triggerFileInput = () => {
    fileInputRef.current?.click();
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[70vh] lg:h-[80vh] bg-neutral-900 rounded-xl border border-neutral-700 overflow-hidden">
      {/* Chat messages */}
      <div className="flex-grow overflow-y-auto p-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <motion.div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div 
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.type === 'user' 
                    ? 'bg-primary-600 text-white' 
                    : 'bg-neutral-800 text-neutral-200'
                }`}
              >
                {message.content}
              </div>
            </motion.div>
          ))}
          {isProcessing && (
            <div className="flex justify-start">
              <div className="bg-neutral-800 text-neutral-200 rounded-2xl px-4 py-3">
                <div className="flex items-center space-x-2">
                  <RotateCw className="animate-spin h-4 w-4" />
                  <span>Processing...</span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>
      
      {/* File uploads area */}
      {files.length > 0 && (
        <div className="bg-neutral-800 p-3 border-t border-neutral-700">
          <div className="flex flex-wrap gap-2">
            {files.map(file => (
              <div key={file.id} className="flex items-center bg-neutral-700 rounded px-3 py-1 text-sm">
                <span className="truncate max-w-40">{file.name}</span>
                <button 
                  className="ml-2 text-neutral-400 hover:text-white transition-colors"
                  onClick={() => removeFile(file.id)}
                >
                  <XCircle size={16} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
      
      {/* Input area */}
      <div className="border-t border-neutral-700 p-3 bg-neutral-800">
        <div className="flex items-end">
          <button 
            className="p-2 text-neutral-400 hover:text-white transition-colors"
            onClick={triggerFileInput}
          >
            <Paperclip size={20} />
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              className="hidden"
              multiple
            />
          </button>
          
          <textarea
            className="flex-grow bg-neutral-700 text-white rounded-xl px-3 py-2 outline-none resize-none min-h-[40px] max-h-32"
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            rows={1}
          />
          
          <button 
            className="p-2 ml-2 text-neutral-400 hover:text-white transition-colors"
            onClick={handleSendMessage}
            disabled={input.trim() === '' && files.length === 0}
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
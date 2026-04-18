import { Sparkles } from 'lucide-react';
import { motion } from "framer-motion";

export function Header() {
  return (
    <header className="bg-[#ffffff] relative overflow-hidden">
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `radial-gradient(#151616 1px, transparent 1px)`,
            backgroundSize: "24px 24px",
            opacity: "0.1",
          }}
        />
      </div>

      <div className="container mx-auto px-6 py-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="inline-flex items-center gap-2 bg-[#151616] text-white rounded-full px-4 py-2"
          >
            <Sparkles className="w-4 h-4 text-[#D6F32F]" />
            <span className="text-sm font-medium">Powered by Vision Transformer & CNN-LSTM</span>
          </motion.div>
        </div>
      </div>
    </header>
  );
} 
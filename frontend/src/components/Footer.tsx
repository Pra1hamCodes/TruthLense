import { useState } from "react";
import { motion } from "framer-motion";
import type { ComponentType } from "react";
import { Link } from "react-router-dom";
import {
  Bot,
  ArrowRight,
  Twitter,
  Github,
  Instagram,
  MessageSquare,
  Heart,
} from "lucide-react";

interface SocialButtonProps {
  icon: ComponentType<{ className?: string }>;
  label: string;
  href: string;
}

// Newsletter Input Component
const NewsletterInput = () => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      className="relative flex"
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}>
      <input
        type="email"
        placeholder="Stay updated with latest AI detection features"
        className="w-full px-4 py-3 rounded-l-xl border-2 border-r-0 border-[#151616] 
          focus:outline-none focus:ring-2 ring-[#D6F32F]"
      />
      <button
        className="bg-[#D6F32F] px-6 rounded-r-xl border-2 border-[#151616] flex items-center gap-2 
          font-bold hover:bg-[#D6F32F]/80 transition-colors">
        Subscribe
        <motion.div
          animate={isHovered ? { x: [0, 5, 0] } : {}}
          transition={{ duration: 1, repeat: Infinity }}>
          <ArrowRight className="w-5 h-5" />
        </motion.div>
      </button>
    </motion.div>
  );
};

// Social Button Component
const SocialButton = ({ icon: Icon, label, href }: SocialButtonProps) => (
  <motion.a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    whileHover={{ rotate: 10 }}
    whileTap={{ scale: 0.9 }}
    className="group relative w-10 h-10 bg-white rounded-xl flex items-center justify-center 
      border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[2px_2px_0px_0px_#151616] 
      hover:translate-y-[2px] hover:translate-x-[2px] transition-all">
    <Icon className="w-5 h-5 text-[#151616]" />
    <span
      className="absolute -top-8 scale-0 group-hover:scale-100 transition-transform 
      bg-[#151616] text-white text-xs py-1 px-2 rounded">
      {label}
    </span>
  </motion.a>
);

const Footer = () => {
  return (
    <footer className="relative bg-[#ffffff] pt-20 overflow-hidden">
      {/* Top Border Design */}
      <div className="absolute top-0 left-0 right-0 h-2 bg-[#151616]" />
      <div className="absolute top-2 left-0 right-0 h-1 bg-[#D6F32F]" />

      <div className="container mx-auto px-6">
        {/* Newsletter Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center max-w-xl mx-auto mb-20"
        >
          <h2 className="text-3xl font-bold mb-4">
            Join the Fight Against Deepfakes
          </h2>
          <p className="text-[#151616]/70 mb-6">
            Stay updated with the latest in AI-powered deepfake detection
          </p>
          <NewsletterInput />
        </motion.div>

        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 py-16 border-t-2 border-[#151616]">
          {/* Brand Section */}
          <div className="md:col-span-2 space-y-6">
            <motion.div
              className="flex items-center gap-3"
              whileHover={{ scale: 1.02 }}>
              <div
                className="w-12 h-12 bg-[#D6F32F] rounded-xl flex items-center justify-center 
                border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616]">
                <Bot className="w-7 h-7 text-[#151616]" />
              </div>
              <h2 className="text-3xl font-bold">DeepShield</h2>
            </motion.div>
            <p className="text-[#151616]/70 max-w-md">
              Empowering truth in the digital age. Our AI-powered platform helps you detect and combat deepfakes, ensuring authenticity in online content.
            </p>
            <div className="flex gap-4">
              <SocialButton icon={Twitter} label="Twitter" href="#" />
              <SocialButton icon={Github} label="GitHub" href="#" />
              <SocialButton icon={Instagram} label="Instagram" href="#" />
              <SocialButton icon={MessageSquare} label="Discord" href="#" />
            </div>
          </div>

          {/* Approach CTA */}
          <div className="md:col-span-1 flex flex-col items-start md:justify-end gap-4">
            <div className="w-full md:max-w-sm border-2 border-[#151616] rounded-2xl p-6 bg-[#f4f8ff] shadow-[4px_4px_0px_0px_#151616]">
              <h3 className="font-bold text-xl mb-3">Our Approach</h3>
              <p className="text-[#151616]/70 mb-5 text-sm">
                Explore our complete AI pipeline with model choices and methodology.
              </p>
              <Link
                to="/our-approach"
                className="inline-flex items-center gap-2 bg-[#D6F32F] px-4 py-2 rounded-lg border-2 border-[#151616] font-bold text-[#151616] shadow-[3px_3px_0px_0px_#151616] hover:shadow-[1px_1px_0px_0px_#151616] hover:translate-x-[2px] hover:translate-y-[2px] transition-all text-sm"
              >
                View Approach
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
            
            <div className="w-full md:max-w-sm border-2 border-[#151616] rounded-2xl p-6 bg-[#fff4f8] shadow-[4px_4px_0px_0px_#151616]">
              <h3 className="font-bold text-xl mb-3">Model Evaluation</h3>
              <p className="text-[#151616]/70 mb-5 text-sm">
                See comprehensive performance metrics and evaluation results.
              </p>
              <Link
                to="/model-evaluation"
                className="inline-flex items-center gap-2 bg-[#D6F32F] px-4 py-2 rounded-lg border-2 border-[#151616] font-bold text-[#151616] shadow-[3px_3px_0px_0px_#151616] hover:shadow-[1px_1px_0px_0px_#151616] hover:translate-x-[2px] hover:translate-y-[2px] transition-all text-sm"
              >
                View Metrics
                <ArrowRight className="w-3 h-3" />
              </Link>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div
          className="border-t-2 border-[#151616]/10 py-6 flex flex-col md:flex-row 
          justify-between items-center gap-4">
          <div className="flex items-center gap-2 text-sm text-[#151616]/70">
            <span>© 2024 DeepShield. All rights reserved.</span>
            <span className="flex items-center gap-1">
              Made with <Heart className="w-4 h-4 text-[#D6F32F]" /> by Shubham
            </span>
          </div>

          <motion.div
            className="flex items-center gap-3 bg-white px-4 py-2 rounded-full border-2 
              border-[#151616] shadow-[4px_4px_0px_0px_#151616]"
            whileHover={{ y: -2 }}>
            <span className="w-2 h-2 bg-[#D6F32F] rounded-full animate-pulse" />
            <span className="text-sm font-medium">AI Detection Active 24/7</span>
          </motion.div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
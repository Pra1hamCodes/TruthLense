import { useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { ComponentType, CSSProperties } from "react";
import { useAuth } from "@/contexts/auth-context";
import {
  Scan,
  Brain,
  Video,
  Shield,
  BarChart2,
  BadgeCheck,
  ChevronRight,
  Sparkles,
} from "lucide-react";

// Feature colors for spiritual theme
const featureColors = [
  { bg: "#FFE8EC", accent: "#FFB6C1", icon: "#FF69B4" }, // Rose for Love & Relationships
  { bg: "#E8F4FF", accent: "#B6E0FF", icon: "#4DA6FF" }, // Blue for Meditation & Peace
  { bg: "#E8FFE8", accent: "#B6FFB6", icon: "#4CAF50" }, // Green for Growth
  { bg: "#FFF4E8", accent: "#FFD7B6", icon: "#FF9800" }, // Orange for Energy
  { bg: "#F4E8FF", accent: "#E0B6FF", icon: "#9C27B0" }, // Purple for Spirituality
  { bg: "#FFFFE8", accent: "#FFFFA6", icon: "#D6F32F" }, // Yellow for Enlightenment
];

interface Feature {
  icon: ComponentType<{ className?: string; style?: CSSProperties }>;
  title: string;
  description: string;
}

interface FeatureCardProps {
  feature: Feature;
  index: number;
}

interface FloatingElementProps {
  className: string;
  color: string;
  delay?: number;
}

const FeatureCard = ({ feature, index }: FeatureCardProps) => {
  const [isHovered, setIsHovered] = useState(false);
  const colors = featureColors[index];

  return (
    <motion.div
      className="relative group"
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}>
      <motion.div
        className="relative bg-white rounded-3xl p-8 h-full border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[8px_8px_0px_0px_#151616] transition-all duration-300"
        onHoverStart={() => setIsHovered(true)}
        onHoverEnd={() => setIsHovered(false)}
        whileHover={{ y: -5 }}
        style={{
          background: `linear-gradient(135deg, white 0%, ${colors.bg} 100%)`,
        }}>
        <div className="absolute top-4 right-4 text-sm font-bold text-[#151616]/30">
          0{index + 1}
        </div>

        <motion.div
          className="relative w-16 h-16 rounded-2xl bg-white flex items-center justify-center border-2 border-[#151616] mb-6 overflow-hidden"
          animate={isHovered ? { rotate: 360 } : {}}
          transition={{ duration: 0.5 }}>
          <div
            className="absolute inset-0 opacity-50"
            style={{
              background: `linear-gradient(135deg, ${colors.bg} 0%, ${colors.accent} 100%)`,
            }}
          />
          <feature.icon
            className="w-8 h-8 relative z-10"
            style={{ color: colors.icon }}
          />
        </motion.div>

        <div className="relative">
          <h3 className="text-xl font-bold text-[#151616] mb-3">
            {feature.title}
          </h3>
          <p className="text-[#151616]/70 leading-relaxed">
            {feature.description}
          </p>
        </div>

        <motion.div
          className="absolute -top-2 -right-2 w-6 h-6 rounded-full border-2 border-[#151616]"
          style={{ background: colors.accent }}
          animate={isHovered ? { scale: [1, 1.2, 1] } : {}}
          transition={{ duration: 1, repeat: Infinity }}
        />
      </motion.div>
    </motion.div>
  );
};

const FloatingElement = ({ className, color, delay = 0 }: FloatingElementProps) => (
  <motion.div
    className={`absolute ${className}`}
    style={{ background: color }}
    animate={{
      y: [-10, 10, -10],
      rotate: [0, 5, -5, 0],
    }}
    transition={{
      duration: 6,
      repeat: Infinity,
      delay,
    }}
  />
);

const FeatureSection = () => {
  const { session } = useAuth();
  const ctaPath = session ? '/create-post' : '/login';

  const features = [
    {
      icon: Brain,
      title: "Vision Transformer",
      description:
        "State-of-the-art Vision Transformer model that analyzes spatial relationships between video frames to detect manipulation patterns.",
    },
    {
      icon: Scan,
      title: "CNN-LSTM Analysis",
      description:
        "Dual-model approach combining CNN-LSTM architecture to analyze temporal inconsistencies and detect sophisticated deepfake patterns.",
    },
    {
      icon: Video,
      title: "Frame-by-Frame Detection",
      description:
        "Detailed frame-by-frame analysis providing comprehensive insights into potential manipulations within your videos.",
    },
    {
      icon: Shield,
      title: "Real-Time Results",
      description:
        "Get instant analysis results with confidence scores and clear indicators showing which parts of the video may be manipulated.",
    },
    {
      icon: BadgeCheck,
      title: "Verification Badges",
      description:
        "Trusted verification system that provides clear real/fake badges, helping maintain authenticity in our community.",
    },
    {
      icon: BarChart2,
      title: "Advanced Metrics",
      description:
        "Detailed metrics including confidence percentages, frame counts, and comprehensive analysis summaries for each video.",
    },
  ];

  return (
    <section className="py-24 bg-[#ffffff] relative overflow-hidden">
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

      <FloatingElement
        className="top-20 left-10 w-20 h-20 rounded-full opacity-20 blur-xl"
        color={featureColors[0].accent}
      />
      <FloatingElement
        className="bottom-40 right-20 w-32 h-32 rounded-full opacity-20 blur-xl"
        color={featureColors[2].accent}
        delay={1}
      />

      <div className="container mx-auto px-6 relative z-10">
        <div className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-flex items-center gap-2 bg-[#151616] text-white rounded-full px-4 py-2 mb-4"
          >
            <Sparkles className="w-4 h-4 text-[#D6F32F]" />
            <span className="text-sm font-medium">Advanced Detection Features</span>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-4xl md:text-5xl font-black text-[#151616] mb-6"
          >
            Comprehensive Analysis
            <span className="inline-block ml-2">
              <motion.div
                animate={{ rotate: [0, 10, 0] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="inline-block"
              >
                ✨
              </motion.div>
            </span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.3 }}
            className="text-xl text-[#151616]/70 max-w-2xl mx-auto"
          >
            Cutting-edge AI technology combined with user-friendly features for reliable deepfake detection
          </motion.p>
        </div>

        <motion.div
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.4 }}
        >
          {features.map((feature, index) => (
            <FeatureCard key={index} feature={feature} index={index} />
          ))}
        </motion.div>

        <motion.div
          className="mt-20 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.5 }}
        >
          <Link
            to={ctaPath}
            className="group bg-[#D6F32F] px-8 py-4 rounded-2xl text-xl font-bold text-[#151616] border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[2px_2px_0px_0px_#151616] hover:translate-y-[2px] hover:translate-x-[2px] transition-all duration-200 inline-flex items-center gap-2 mx-auto"
          >
            Start Detecting Now
            <motion.div
              animate={{ x: [0, 5, 0] }}
              transition={{ duration: 1, repeat: Infinity }}
            >
              <ChevronRight className="w-5 h-5" />
            </motion.div>
          </Link>
        </motion.div>
      </div>
    </section>
  );
};

export default FeatureSection;

import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { ComponentType, ReactNode } from "react";
import { useAuth } from "@/contexts/auth-context";
import {
  Moon,
  Stars,
  Sparkles,
  ArrowRight,
  Flower,
  ScrollText,
  CircleDot,
  Triangle,
  Square,
} from "lucide-react";

interface FeatureCardProps {
  icon: ComponentType<{ className?: string }>;
  title: string;
  description: string;
  delay: number;
}

interface FloatingElementProps {
  children: ReactNode;
  delay?: number;
  rotate?: boolean;
}

const FeatureCard = ({ icon: Icon, title, description, delay }: FeatureCardProps) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.5, delay }}
    className="bg-white p-6 rounded-2xl border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:translate-y-1 hover:shadow-[2px_2px_0px_0px_#151616] transition-all duration-200 text-center"
  >
    <div className="flex justify-center mb-4">
      <div className="w-12 h-12 bg-[#D6F32F] rounded-xl border-2 border-[#151616] flex items-center justify-center">
        <Icon className="w-6 h-6 text-[#151616]" />
      </div>
    </div>
    <h3 className="text-lg font-bold text-[#151616] mb-2">{title}</h3>
    <p className="text-[#151616]/70">{description}</p>
  </motion.div>
);

const FloatingElement = ({ children, delay = 0, rotate = false }: FloatingElementProps) => (
  <motion.div
    initial={{ y: 10 }}
    animate={{
      y: [-10, 10],
      rotate: rotate ? [-10, 10] : 0,
    }}
    transition={{
      y: {
        duration: 2,
        repeat: Infinity,
        repeatType: "reverse",
        ease: "easeInOut",
        delay,
      },
      rotate: {
        duration: 3,
        repeat: Infinity,
        repeatType: "reverse",
        ease: "easeInOut",
        delay,
      },
    }}>
    {children}
  </motion.div>
);

const GeometricBackground = () => {
  const shapes = Array(6).fill(null);

  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden">
      {shapes.map((_, index) => (
        <motion.div
          key={index}
          className="absolute"
          initial={{ opacity: 0 }}
          animate={{ opacity: [0.1, 0.3, 0.1] }}
          transition={{
            duration: 3,
            repeat: Infinity,
            delay: index * 0.5,
          }}
          style={{
            left: `${Math.random() * 100}%`,
            top: `${Math.random() * 100}%`,
          }}>
          {index % 3 === 0 ? (
            <CircleDot className="w-16 h-16 text-[#D6F32F]" />
          ) : index % 3 === 1 ? (
            <Triangle className="w-16 h-16 text-[#151616]" />
          ) : (
            <Square className="w-16 h-16 text-[#D6F32F]" />
          )}
        </motion.div>
      ))}
    </div>
  );
};

const ParticleEffect = () => {
  const particles = Array(20).fill(null);

  return (
    <div className="absolute inset-0 pointer-events-none">
      {particles.map((_, index) => (
        <motion.div
          key={index}
          className="absolute w-2 h-2 bg-[#D6F32F] rounded-full"
          initial={{
            x: Math.random() * window.innerWidth,
            y: Math.random() * window.innerHeight,
            scale: 0,
          }}
          animate={{
            y: [0, -30, 0],
            scale: [0, 1, 0],
            opacity: [0, 1, 0],
          }}
          transition={{
            duration: 2 + Math.random() * 2,
            repeat: Infinity,
            delay: index * 0.2,
          }}
        />
      ))}
    </div>
  );
};

const Hero = () => {
  const { session } = useAuth();
  const ctaPath = session ? '/create-post' : '/login';

  return (
    <div className="min-h-screen bg-[#ffffff] relative overflow-hidden pt-4">
      {/* Background Pattern */}
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

      <GeometricBackground />
      <ParticleEffect />

      {/* Floating Decorative Elements */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        <FloatingElement delay={0.2} rotate>
          <Stars className="absolute top-20 left-1/4 w-8 h-8 text-[#D6F32F]" />
        </FloatingElement>
        <FloatingElement delay={0.4} rotate>
          <Moon className="absolute top-40 right-1/4 w-8 h-8 text-[#151616]" />
        </FloatingElement>
        <FloatingElement delay={0.6} rotate>
          <Flower className="absolute bottom-20 left-1/3 w-8 h-8 text-[#D6F32F]" />
        </FloatingElement>
        <FloatingElement delay={0.8} rotate>
          <Sparkles className="absolute top-32 right-1/3 w-8 h-8 text-[#151616]" />
        </FloatingElement>
      </div>

      <div className="container mx-auto px-6 py-8 relative z-10">
        <motion.div className="text-center max-w-4xl mx-auto">
          <h1 className="text-5xl md:text-6xl font-black text-[#151616] mb-4">
            Detect Deepfakes with
            <div className="relative inline-block px-2">
              DeepSeek
              <motion.div
                className="absolute inset-0 bg-[#D6F32F]/40 -rotate-2"
                initial={{ width: 0 }}
                animate={{ width: "100%" }}
                transition={{ duration: 0.8, delay: 0.5 }}
              />
            </div>
            <div className="mt-2">Trust What You See</div>
          </h1>

          <p className="text-xl text-[#151616]/70 mb-6 max-w-2xl mx-auto">
            Our advanced models analyze every frame of your videos to detect deepfakes with high accuracy. Share and verify content with confidence in our trusted community.
          </p>

          <div className="flex gap-4 justify-center mb-16">
            <motion.div
              className="bg-[#D6F32F] px-8 py-4 rounded-2xl text-xl font-bold text-[#151616] border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:translate-y-1 hover:shadow-[2px_2px_0px_0px_#151616] transition-all duration-200 flex items-center gap-2"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}>
              <Link to={ctaPath} className="flex items-center gap-2">
                Start Detecting
                <ArrowRight className="w-5 h-5" />
              </Link>
            </motion.div>
          </div>

          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <FeatureCard
              icon={Stars}
              title="Frame Analysis"
              description="Advanced frame-by-frame analysis using Vision Transformer technology"
              delay={0.2}
            />
            <FeatureCard
              icon={Sparkles}
              title="Real-time Detection"
              description="Get instant results with our dual-model approach using ViT and CNN-LSTM"
              delay={0.4}
            />
            <FeatureCard
              icon={ScrollText}
              title="Verification Badges"
              description="Clear real/fake badges and confidence scores for each analyzed video"
              delay={0.6}
            />
          </div>

          {/* Bottom Decoration */}
          <motion.div
            className="mt-16 text-center"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.8 }}>
            <div className="inline-flex gap-4">
              <motion.div
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}>
                ✨
              </motion.div>
              <motion.div
                animate={{ rotate: [0, -360] }}
                transition={{ duration: 20, repeat: Infinity, ease: "linear" }}>
                🌟
              </motion.div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

export default Hero;

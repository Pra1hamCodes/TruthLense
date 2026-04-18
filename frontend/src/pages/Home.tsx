import Hero from '../components/Hero';
import FeatureSection from '../components/FeatureSection';
import { Header } from '../components/Header';

export default function Home() {
  return (
    <div className="pt-16">
      <Header />
      <Hero />
      <FeatureSection />
    </div>
  );
} 
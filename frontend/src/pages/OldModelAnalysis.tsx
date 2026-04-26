import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, BarChart2, Film, Percent } from 'lucide-react';
import axios from 'axios';
import { API_BASE_URL } from '../lib/config';

interface Frame {
  frame: string;
  frame_path: string;
  confidence: number;
  is_fake: boolean;
}

interface Analysis {
  frames_analysis: Frame[];
  confidence: number;
  is_fake: boolean;
  summary: {
    status: string;
    confidence_percentage: number;
    total_frames: number;
    real_frames: number;
    fake_frames: number;
  };
}

const API_URL = API_BASE_URL;

const resolveFrameUrl = (framePath: string) => {
  if (framePath.startsWith('http://') || framePath.startsWith('https://')) {
    return framePath;
  }
  return `${API_URL}${framePath.startsWith('/') ? framePath : `/${framePath}`}`;
};

export default function OldModelAnalysis() {
  const { postId } = useParams();
  const navigate = useNavigate();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());

  const handleImageLoad = (imagePath: string) => {
    setLoadedImages(prev => new Set(prev).add(imagePath));
  };

  useEffect(() => {
    const fetchAnalysis = async () => {
      const token = localStorage.getItem('token');
      
      if (!token) {
        navigate('/login');
        return;
      }

      try {
        const response = await axios.get(
          `${API_URL}/api/posts/old-model-analysis/${postId}`,
          {
            headers: {
              'Authorization': `Bearer ${token}`
            }
          }
        );
        setAnalysis(response.data);
      } catch (error) {
        console.error('Error fetching analysis:', error);
        if (axios.isAxiosError(error) && error.response?.status === 401) {
          navigate('/login');
        }
      }
    };
    fetchAnalysis();
  }, [postId, navigate]);

  if (!analysis) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#D6F32F]"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#FAFAFA] py-24 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616]"
        >
          <div className="flex items-center gap-2 mb-6">
            <span className="text-lg font-semibold text-[#151616]">Old Model Analysis Results</span>
            <div className="px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
              Completed
            </div>
          </div>

          {/* Analysis Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className={`p-6 rounded-xl border-2 ${
                analysis.summary.status === 'REAL' 
                  ? 'border-green-500 bg-green-50' 
                  : 'border-red-500 bg-red-50'
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                {analysis.summary.status === 'REAL' ? (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                ) : (
                  <AlertTriangle className="w-6 h-6 text-red-500" />
                )}
                <h3 className="font-bold">Final Verdict</h3>
              </div>
              <p className={`text-2xl font-bold ${
                analysis.summary.status === 'REAL' ? 'text-green-600' : 'text-red-600'
              }`}>
                {analysis.summary.status}
              </p>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="p-6 rounded-xl border-2 border-[#151616] bg-white"
            >
              <div className="flex items-center gap-3 mb-3">
                <Percent className="w-6 h-6 text-[#151616]" />
                <h3 className="font-bold">Confidence Score</h3>
              </div>
              <p className="text-2xl font-bold text-[#151616]">
                {analysis.summary.confidence_percentage}%
              </p>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
              className="p-6 rounded-xl border-2 border-[#151616] bg-white"
            >
              <div className="flex items-center gap-3 mb-3">
                <Film className="w-6 h-6 text-[#151616]" />
                <h3 className="font-bold">Frame Analysis</h3>
              </div>
              <p className="text-2xl font-bold text-[#151616]">
                {analysis.summary.real_frames}/{analysis.summary.total_frames}
                <span className="text-base font-normal text-[#151616]/70 ml-2">real frames</span>
              </p>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.4 }}
              className="p-6 rounded-xl border-2 border-[#151616] bg-white"
            >
              <div className="flex items-center gap-3 mb-3">
                <BarChart2 className="w-6 h-6 text-[#151616]" />
                <h3 className="font-bold">Fake Frames</h3>
              </div>
              <p className="text-2xl font-bold text-[#151616]">
                {analysis.summary.fake_frames}
                <span className="text-base font-normal text-[#151616]/70 ml-2">frames</span>
              </p>
            </motion.div>
          </div>
        </motion.div>

        {/* Frames Grid Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616]"
        >
          <div className="flex items-center gap-3 mb-6">
            <Film className="w-6 h-6 text-[#151616]" />
            <h2 className="text-xl font-bold">Analyzed Frames</h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {analysis.frames_analysis.map((frame, index) => {
              const imagePath = resolveFrameUrl(frame.frame_path);
              const isLoaded = loadedImages.has(imagePath);

              return (
                <div key={index} className="relative">
                  <div className="aspect-video rounded-xl overflow-hidden border-2 border-[#151616] bg-gray-100">
                    {!isLoaded && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-6 h-6 border-2 border-[#D6F32F] border-t-transparent rounded-full animate-spin"></div>
                      </div>
                    )}
                    <img
                      src={imagePath}
                      alt={`Frame ${index + 1}`}
                      className={`w-full h-full object-cover transition-opacity duration-200 ${
                        isLoaded ? 'opacity-100' : 'opacity-0'
                      }`}
                      onLoad={() => handleImageLoad(imagePath)}
                      onError={() => console.error('Image load error:', imagePath)}
                    />
                  </div>
                  <div className={`absolute top-2 right-2 px-2 py-1 rounded-full text-xs font-bold ${
                    frame.is_fake 
                      ? 'bg-red-500 text-white' 
                      : 'bg-green-500 text-white'
                  }`}>
                    {frame.is_fake ? 'FAKE' : 'REAL'}
                  </div>
                </div>
              );
            })}
          </div>
        </motion.div>

        {/* Detailed Analysis Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616]"
        >
          <div className="flex items-center gap-3 mb-6">
            <BarChart2 className="w-6 h-6 text-[#151616]" />
            <h2 className="text-xl font-bold">Frame-by-Frame Analysis</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b-2 border-[#151616]">
                  <th className="px-4 py-3 font-bold">Frame</th>
                  <th className="px-4 py-3 font-bold">Status</th>
                  <th className="px-4 py-3 font-bold">Confidence</th>
                  <th className="px-4 py-3 font-bold">Preview</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#151616]/10">
                {analysis.frames_analysis.map((frame, index) => (
                  <tr key={index}>
                    <td className="px-4 py-3">Frame {index + 1}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
                        frame.is_fake 
                          ? 'bg-red-100 text-red-700' 
                          : 'bg-green-100 text-green-700'
                      }`}>
                        {frame.is_fake ? 'FAKE' : 'REAL'}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-medium">
                      {(frame.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3">
                      <div className="w-24 h-16 rounded-lg overflow-hidden border-2 border-[#151616]">
                        <img 
                          src={resolveFrameUrl(frame.frame_path)}
                          alt={`Frame ${index + 1}`}
                          className="w-full h-full object-cover"
                        />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </motion.div>
      </div>
    </div>
  );
} 
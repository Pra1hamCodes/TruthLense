import { useCallback, useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, CheckCircle, BarChart2, Film, Percent, Video, ImageOff } from 'lucide-react';
import { API_BASE_URL, FLASK_MAIN_URL } from '../lib/config';

interface Frame {
  frame: string;
  frame_path: string;
  confidence: number;
  is_fake: boolean;
}

interface Post {
  id: string;
  title: string;
  content: string;
  media_url: string;
  media_type: 'image' | 'video';
  created_at: string;
  analysis_status: string;
  deepfake_analysis?: {
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
  };
  profiles?: {
    username: string;
  };
}

const API_URL = API_BASE_URL;
const FLASK_URL = FLASK_MAIN_URL;

export default function PostAnalysis() {
  const { postId } = useParams();
  const [post, setPost] = useState<Post | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());
  const [failedImages, setFailedImages] = useState<Set<string>>(new Set());
  const [error, setError] = useState<string | null>(null);
  const [isLoadingOldModel, setIsLoadingOldModel] = useState(false);

  const fetchPost = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/api/posts/${postId}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        throw new Error('Failed to fetch post data');
      }

      const data = await response.json();
      setPost(data);
    } catch (fetchError) {
      console.error('Error fetching post:', fetchError);
      setError('Failed to load post data. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  }, [postId]);

  useEffect(() => {
    fetchPost();
  }, [fetchPost]);

  const handleImageLoad = (imagePath: string) => {
    setLoadedImages(prev => new Set(prev).add(imagePath));
    setFailedImages(prev => {
      const newSet = new Set(prev);
      newSet.delete(imagePath);
      return newSet;
    });
  };

  const handleImageError = (imagePath: string) => {
    console.error(`Failed to load image: ${imagePath}`);
    setFailedImages(prev => new Set(prev).add(imagePath));
    setLoadedImages(prev => {
      const newSet = new Set(prev);
      newSet.delete(imagePath);
      return newSet;
    });
  };

  const determineVideoStatus = (frames: Frame[]) => {
    const realFrames = frames.filter(f => !f.is_fake).length;
    const fakeFrames = frames.filter(f => f.is_fake).length;
    const totalConfidence = frames.reduce((sum, frame) => sum + frame.confidence, 0);
    const averageConfidence = totalConfidence / frames.length;
    
    return {
      isReal: realFrames >= fakeFrames,
      realCount: realFrames,
      fakeCount: fakeFrames,
      totalFrames: frames.length,
      confidence: averageConfidence
    };
  };

  const handleOldModelAnalysis = async () => {
    try {
      setIsLoadingOldModel(true);
      const response = await fetch(`${API_URL}/api/posts/analyze-old-model/${postId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Analysis failed');
      }
      
      const data = await response.json();
      
      if (!data || !data.frames_analysis) {
        throw new Error('Invalid analysis data received');
      }

      // Store the analysis result in localStorage for the new page
      localStorage.setItem(`oldModelAnalysis_${postId}`, JSON.stringify(data));
      
      // Open new window with results
      window.open(`/old-model-analysis/${postId}`, '_blank');
    } catch (error: unknown) {
      console.error('Error:', error);
      const message = error instanceof Error ? error.message : 'Analysis failed';
      alert(`Analysis failed: ${message}. Please ensure both Flask servers are running.`);
    } finally {
      setIsLoadingOldModel(false);
    }
  };

  const handleAnalyzeNow = async () => {
    try {
      setIsAnalyzing(true);
      setError(null);

      const response = await fetch(`${API_URL}/api/posts/analyze/${postId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.message || 'Video analysis failed');
      }

      await fetchPost();
    } catch (analyzeError) {
      console.error('Error re-analyzing video:', analyzeError);
      setError(analyzeError instanceof Error ? analyzeError.message : 'Video analysis failed');
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-[#D6F32F]"></div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="min-h-screen bg-gray-50 pt-24 px-4">
        <div className="max-w-3xl mx-auto text-center">
          <AlertTriangle className="w-16 h-16 mx-auto mb-4 text-red-500" />
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            {error || 'Post not found'}
          </h2>
          <p className="text-gray-600">Please try again later or contact support if the issue persists.</p>
        </div>
      </div>
    );
  }

  const status = determineVideoStatus(post.deepfake_analysis?.frames_analysis || []);

  if (!post?.deepfake_analysis?.frames_analysis) {
    const isVideo = post.media_type === 'video';
    return (
      <div className="min-h-screen bg-gray-50 pt-24 px-4">
        <div className="max-w-6xl mx-auto text-center">
          <AlertTriangle className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-800 mb-2">No Analysis Available</h2>
          <p className="text-gray-600 mb-4">
            {isVideo
              ? `Current status: ${post.analysis_status || 'unknown'}. You can run analysis now.`
              : 'This post is an image. Frame-by-frame deepfake analysis is available for videos only.'}
          </p>
          {isVideo && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleAnalyzeNow}
              disabled={isAnalyzing || post.analysis_status === 'processing'}
              className="px-6 py-3 rounded-lg bg-[#D6F32F] border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[1px_1px_0px_0px_#151616] hover:translate-x-[2px] hover:translate-y-[2px] transition-all text-sm font-medium disabled:opacity-50"
            >
              {isAnalyzing ? 'Analyzing Video...' : 'Run Analysis Now'}
            </motion.button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 pt-24 pb-24 px-4">
      <div className="max-w-6xl mx-auto">
        {/* Header Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616] mb-8"
        >
          <div className="flex flex-col md:flex-row md:items-center justify-between mb-8">
            <div className="flex items-center gap-4 mb-4 md:mb-0">
              <div className="w-12 h-12 bg-[#D6F32F] rounded-xl border-2 border-[#151616] flex items-center justify-center">
                <Shield className="w-6 h-6 text-[#151616]" />
              </div>
              <div>
                <h1 className="text-3xl font-bold text-[#151616]">{post.title}</h1>
                <div className="flex items-center gap-2 mt-1">
                  <span className="text-[#151616]/70">Posted by</span>
                  <span className="font-medium text-[#151616]">@{post.profiles?.username}</span>
                  <span className="text-[#151616]/70">•</span>
                  <span className="text-[#151616]/70">
                    {new Date(post.created_at).toLocaleDateString('en-US', {
                      month: 'long',
                      day: 'numeric',
                      year: 'numeric'
                    })}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-lg font-semibold text-[#151616]">Analysis Results</span>
              <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                post.analysis_status === 'completed' 
                  ? 'bg-green-100 text-green-800' 
                  : 'bg-yellow-100 text-yellow-800'
              }`}>
                {post.analysis_status === 'completed' ? 'Completed' : 'Processing'}
              </div>
            </div>
          </div>

          {/* Analysis Summary */}
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <motion.div 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 }}
              className={`p-6 rounded-xl border-2 ${
                status.isReal ? 'border-green-500 bg-green-50' : 'border-red-500 bg-red-50'
              }`}
            >
              <div className="flex items-center gap-3 mb-3">
                {status.isReal ? (
                  <CheckCircle className="w-6 h-6 text-green-500" />
                ) : (
                  <AlertTriangle className="w-6 h-6 text-red-500" />
                )}
                <h3 className="font-bold">Final Verdict</h3>
              </div>
              <p className={`text-2xl font-bold ${
                status.isReal ? 'text-green-600' : 'text-red-600'
              }`}>
                {status.isReal ? 'AUTHENTIC' : 'DEEPFAKE'}
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
                {(status.confidence * 100).toFixed(1)}%
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
                {status.realCount}/{status.totalFrames}
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
                <Shield className="w-6 h-6 text-[#151616]" />
                <h3 className="font-bold">Old Model Analysis</h3>
              </div>
              <div className="flex gap-4 mt-6">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleOldModelAnalysis}
                  disabled={isLoadingOldModel}
                  className="px-6 py-3 rounded-lg bg-[#D6F32F] border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[1px_1px_0px_0px_#151616] hover:translate-x-[2px] hover:translate-y-[2px] transition-all text-sm font-medium disabled:opacity-50 flex items-center gap-2"
                >
                  {isLoadingOldModel ? (
                    <>
                      <div className="w-4 h-4 border-2 border-[#151616] border-t-transparent rounded-full animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Shield className="w-4 h-4" />
                      See Result (Old Model)
                    </>
                  )}
                </motion.button>
              </div>
            </motion.div>
          </div>
        </motion.div>

        {/* Video Preview Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616] mb-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <Video className="w-6 h-6 text-[#151616]" />
            <h2 className="text-xl font-bold">Original Video</h2>
          </div>

          <div className="aspect-video rounded-xl overflow-hidden border-2 border-[#151616] bg-gray-100">
            <video 
              src={post.media_url.startsWith('http') ? post.media_url : `${API_URL}${post.media_url}`}
              className="w-full h-full object-contain"
              controls
              onError={(e) => {
                console.error('Video loading error:', e);
                setError('Failed to load video');
              }}
            />
          </div>
        </motion.div>

        {/* Frames Grid Section */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616] mb-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <Film className="w-6 h-6 text-[#151616]" />
            <h2 className="text-xl font-bold">Analyzed Frames</h2>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            {post.deepfake_analysis.frames_analysis.map((frame, index) => {
              const imagePath = frame.frame_path.startsWith('http') 
                ? frame.frame_path 
                : `${FLASK_URL}${frame.frame_path}`;
              const isLoaded = loadedImages.has(imagePath);
              const hasFailed = failedImages.has(imagePath);

              return (
                <div key={index} className="relative">
                  <div className="aspect-video rounded-xl overflow-hidden border-2 border-[#151616] bg-gray-100">
                    {!isLoaded && !hasFailed && (
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-6 h-6 border-2 border-[#D6F32F] border-t-transparent rounded-full animate-spin"></div>
                      </div>
                    )}
                    {hasFailed ? (
                      <div className="absolute inset-0 flex flex-col items-center justify-center bg-gray-100">
                        <ImageOff className="w-8 h-8 text-gray-400 mb-2" />
                        <span className="text-sm text-gray-500">Failed to load frame</span>
                      </div>
                    ) : (
                      <img
                        src={imagePath}
                        alt={`Frame ${index + 1}`}
                        className={`w-full h-full object-cover transition-opacity duration-200 ${
                          isLoaded ? 'opacity-100' : 'opacity-0'
                        }`}
                        onLoad={() => handleImageLoad(imagePath)}
                        onError={() => handleImageError(imagePath)}
                      />
                    )}
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

        {/* Frames Analysis Table */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-2xl p-8 shadow-lg border-2 border-[#151616]"
        >
          <div className="flex items-center gap-3 mb-6">
            <BarChart2 className="w-6 h-6 text-[#151616]" />
            <h2 className="text-xl font-bold">Detailed Frame Analysis</h2>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-[#151616]/10">
                  <th className="px-4 py-3 text-left">Frame</th>
                  <th className="px-4 py-3 text-left">Result</th>
                  <th className="px-4 py-3 text-left">Confidence</th>
                  <th className="px-4 py-3 text-left">Preview</th>
                </tr>
              </thead>
              <tbody>
                {post.deepfake_analysis.frames_analysis.map((frame, index) => (
                  <tr key={index} className="border-b border-[#151616]/10">
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
                    <td className="px-4 py-3">
                      {(frame.confidence * 100).toFixed(1)}%
                    </td>
                    <td className="px-4 py-3">
                      <div className="w-24 h-16 rounded-lg overflow-hidden border border-[#151616]/10">
                        <img 
                          src={frame.frame_path.startsWith('http') 
                            ? frame.frame_path 
                            : `${FLASK_URL}${frame.frame_path}`}
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
import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Link } from 'react-router-dom';
import { API_BASE_URL } from '../lib/config';

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
  };
}

const API_URL = API_BASE_URL;

const resolveFrameUrl = (framePath: string) => {
  if (framePath.startsWith('http://') || framePath.startsWith('https://')) {
    return framePath;
  }
  return `${API_URL}${framePath.startsWith('/') ? framePath : `/${framePath}`}`;
};

export default function VideoAnalysisDashboard() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [selectedPost, setSelectedPost] = useState<Post | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [loadedImages, setLoadedImages] = useState<Set<string>>(new Set());

  // Memoize the fetch function
  const fetchUserPosts = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${API_URL}/api/posts/user/posts`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      const data = await response.json();
      console.log('Fetched posts:', data);
      if (!Array.isArray(data)) {
        setPosts([]);
        return;
      }
      setPosts(data.filter((post: Post) => post.media_type === 'video'));
    } catch (error) {
      console.error('Error fetching posts:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchUserPosts();
  }, [fetchUserPosts]);

  const handleCloseModal = () => {
    setShowModal(false);
    setTimeout(() => setSelectedPost(null), 200); // Clear selected post after animation
  };

  const handleAnalyze = async (postId: string) => {
    try {
      const response = await fetch(`${API_URL}/api/posts/analyze/${postId}`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) throw new Error('Analysis failed');
      
      const updatedPost = await response.json();
      setPosts(posts.map(p => p.id === postId ? updatedPost : p));
    } catch (error) {
      console.error('Error analyzing frames:', error);
    }
  };

  const handleImageLoad = (imagePath: string) => {
    setLoadedImages(prev => new Set(prev).add(imagePath));
  };

  const handleImageError = (imagePath: string) => {
    console.error(`Error loading image: ${imagePath}`);
  };

  // Add the same helper function
  const determineVideoStatus = (analysis?: Post['deepfake_analysis']) => {
    const frames = analysis?.frames_analysis || [];
    const realFrames = frames.filter(f => !f.is_fake).length;
    const fakeFrames = frames.filter(f => f.is_fake).length;
    const summary = analysis && 'summary' in analysis ? (analysis as Post['deepfake_analysis'] & { summary?: { status: string; confidence_percentage: number; total_frames: number; real_frames: number; fake_frames: number } }).summary : undefined;
    const verdictConfidence = frames.length > 0
      ? Math.round((Math.max(realFrames, fakeFrames) / frames.length) * 100)
      : 0;
    
    return {
      isReal: summary ? summary.status === 'REAL' : realFrames >= fakeFrames,
      realCount: summary?.real_frames ?? realFrames,
      fakeCount: summary?.fake_frames ?? fakeFrames,
      totalFrames: summary?.total_frames ?? frames.length,
      confidencePercentage: verdictConfidence
    };
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <h2 className="text-2xl font-bold mb-6">Video Analysis</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {posts.map(post => (
          <div key={post.id} className="bg-white rounded-lg shadow p-4">
            <video 
              src={`${API_URL}${post.media_url}`}
              className="w-full h-48 object-cover mb-4"
              controls
            />
            <h3 className="font-semibold mb-2">{post.title}</h3>
            
            {post.analysis_status === 'none' ? (
              <div className="space-y-2">
                <button
                  onClick={() => handleAnalyze(post.id)}
                  className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded transition-colors w-full mb-2"
                >
                  Analyze Video
                </button>
              </div>
            ) : post.analysis_status === 'processing' ? (
              <div className="flex items-center justify-center space-x-2 py-2">
                <div className="animate-spin h-4 w-4 border-2 border-blue-500 rounded-full border-t-transparent"></div>
                <span>Processing...</span>
              </div>
            ) : post.analysis_status === 'completed' && post.deepfake_analysis ? (
              <div className="space-y-4">
                {/* Status */}
                {(() => {
                  const status = determineVideoStatus(post.deepfake_analysis);
                  return (
                    <div className={`text-lg font-bold ${
                      status.isReal ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {status.isReal ? 'REAL' : 'FAKE'} 
                      ({status.realCount}/{status.totalFrames} real frames)
                    </div>
                  );
                })()}
                
                {/* Preview first few frames */}
                <div className="grid grid-cols-3 gap-2">
                  {post.deepfake_analysis.frames_analysis.slice(0, 3).map((frame, index) => {
                    const imagePath = resolveFrameUrl(frame.frame_path);
                    const isLoaded = loadedImages.has(imagePath);

                    return (
                      <div key={index} className="relative aspect-video bg-gray-100">
                        {!isLoaded && (
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        )}
                        <img 
                          src={imagePath}
                          alt={`Frame ${index + 1}`}
                          className={`w-full h-full object-cover rounded transition-opacity duration-200 ${
                            isLoaded ? 'opacity-100' : 'opacity-0'
                          }`}
                          onLoad={() => handleImageLoad(imagePath)}
                          onError={() => handleImageError(imagePath)}
                        />
                      </div>
                    );
                  })}
                </div>
                
                <Link
                  to={`/posts/${post.id}/analysis`}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded transition-colors block text-center"
                >
                  View Full Analysis ({post.deepfake_analysis.frames_analysis.length} frames)
                </Link>
              </div>
            ) : (
              <div className="text-red-500 text-center py-2">Analysis failed</div>
            )}
          </div>
        ))}
      </div>

      {/* Frames Modal */}
      <AnimatePresence>
        {showModal && selectedPost?.deepfake_analysis && (
          (() => {
            const status = determineVideoStatus(selectedPost.deepfake_analysis);
            return (
          <div className="fixed inset-0 z-50">
            <div className="absolute inset-0 bg-black bg-opacity-50" onClick={handleCloseModal} />
            <div className="flex items-center justify-center min-h-screen p-4">
              <motion.div 
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto relative"
              >
                <div className="sticky top-0 bg-white pb-4 mb-4 border-b flex justify-between items-center">
                  <h3 className="text-xl font-bold">Extracted Frames</h3>
                  <button 
                    onClick={handleCloseModal}
                    className="p-2 hover:bg-gray-100 rounded-full transition-colors"
                  >
                    ✕
                  </button>
                </div>
                
                <div className="mb-6">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="text-lg font-semibold mb-2">Analysis Summary</h4>
                    <div className="flex items-center space-x-4">
                      <div className={`text-lg font-bold ${
                        selectedPost.deepfake_analysis.is_fake ? 'text-red-600' : 'text-green-600'
                      }`}>
                        {selectedPost.deepfake_analysis.is_fake ? 'FAKE' : 'REAL'}
                      </div>
                      <div className="text-gray-600">
                        Confidence Score: {status.confidencePercentage}%
                      </div>
                    </div>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="min-w-full bg-white border">
                    <thead>
                      <tr className="bg-gray-100">
                        <th className="px-4 py-2 text-left">Frame</th>
                        <th className="px-4 py-2 text-left">Confidence</th>
                        <th className="px-4 py-2 text-left">Result</th>
                      </tr>
                    </thead>
                    <tbody>
                      {selectedPost.deepfake_analysis.frames_analysis.map((frame, index) => (
                        <tr key={index} className="border-t">
                          <td className="px-4 py-2">Frame {index + 1}</td>
                          <td className="px-4 py-2">{(frame.confidence * 100).toFixed(1)}%</td>
                          <td className={`px-4 py-2 font-medium ${
                            frame.is_fake ? 'text-red-600' : 'text-green-600'
                          }`}>
                            {frame.is_fake ? 'FAKE' : 'REAL'}
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
          })()
        )}
      </AnimatePresence>
    </div>
  );
} 
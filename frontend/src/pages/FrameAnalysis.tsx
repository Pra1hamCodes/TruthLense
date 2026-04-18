import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { API_BASE_URL } from '../lib/config';

interface Frame {
  frame: string;
  frame_path: string;
}

interface Post {
  id: string;
  title: string;
  media_url: string;
  media_type: 'image' | 'video';
  created_at: string;
  analysis_status: string;
  deepfake_analysis?: {
    frames_analysis: Frame[];
  };
}

const API_URL = API_BASE_URL;

export default function FrameAnalysis() {
  const { postId } = useParams();
  const [post, setPost] = useState<Post | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [frameLoadErrors, setFrameLoadErrors] = useState<Set<number>>(new Set());

  useEffect(() => {
    let isMounted = true;

    async function fetchPost() {
      try {
        setLoading(true);
        const response = await fetch(`${API_URL}/api/posts/${postId}`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });

        if (!response.ok) {
          throw new Error('Failed to fetch post');
        }

        const data = await response.json();
        if (isMounted) {
          setPost(data);
          setError(null);
        }
      } catch (error) {
        console.error('Error fetching post:', error);
        if (isMounted) {
          setError('Failed to load post analysis');
        }
      } finally {
        if (isMounted) {
          setLoading(false);
        }
      }
    }

    fetchPost();
    return () => {
      isMounted = false;
    };
  }, [postId]);

  const handleFrameError = (index: number) => {
    setFrameLoadErrors(prev => new Set([...prev, index]));
  };

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="animate-pulse text-xl text-blue-600">Loading analysis...</div>
      </div>
    );
  }

  if (error || !post) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-red-500 text-xl">{error || 'Post not found'}</div>
      </div>
    );
  }

  const isProcessing = post.analysis_status === 'processing';
  const frames = post.deepfake_analysis?.frames_analysis ?? [];
  const hasFrames = frames.length > 0;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4">{post.title}</h1>
          
          <div className="bg-white rounded-lg shadow-lg overflow-hidden mb-6">
            <video 
              src={post.media_url.startsWith('http') ? post.media_url : `${API_URL}${post.media_url}`}
              className="w-full h-96 object-contain"
              controls
              playsInline
            />
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-2xl font-bold mb-4">Frame Analysis</h2>
          {isProcessing ? (
            <div className="text-center py-8">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
              <p className="text-lg text-gray-600">Processing video frames...</p>
            </div>
          ) : !hasFrames ? (
            <div className="text-center py-8">
              <p className="text-lg text-gray-600">No frames available for analysis</p>
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {frames.map((frame, index) => {
                if (frameLoadErrors.has(index)) {
                  return null;
                }

                const frameSrc = frame.frame_path.startsWith('http') 
                  ? frame.frame_path 
                  : `${API_URL}${frame.frame_path}`;

                return (
                  <motion.div 
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-white rounded-lg shadow-lg overflow-hidden hover:shadow-xl transition-shadow duration-300"
                  >
                    <div className="relative pt-[75%]">
                      <img 
                        src={frameSrc}
                        alt={`Frame ${index + 1}`}
                        className="absolute top-0 left-0 w-full h-full object-cover"
                        onError={() => handleFrameError(index)}
                      />
                    </div>
                    <div className="p-4">
                      <p className="text-sm font-semibold text-gray-800">Frame {index + 1}</p>
                      <p className="text-xs text-gray-500 mt-1 truncate">{frame.frame}</p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
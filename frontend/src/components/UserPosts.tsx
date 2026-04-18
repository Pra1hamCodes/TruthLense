import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Heart, MessageCircle, Sparkles } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { API_BASE_URL } from '../lib/config';

interface Frame {
  frame: string;
  confidence: number;
  is_fake: boolean;
  frame_path: string;
}

interface Post {
  id: string;
  title: string;
  content: string;
  media_url: string;
  media_type: 'image' | 'video';
  created_at: string;
  profiles: {
    username: string;
  };
  likes_count: number;
  comments_count: number;
  analysis_status: string;
  deepfake_analysis?: {
    is_fake: boolean;
    confidence: number;
    frames_analysis: Frame[];
  };
}

const API_URL = API_BASE_URL;

const PostCard: React.FC<{ post: Post; index: number }> = ({ post, index }) => {
  const [isHovered, setIsHovered] = useState(false);
  const hasCompletedAnalysis =
    post.media_type === 'video' &&
    post.analysis_status === 'completed' &&
    Array.isArray(post.deepfake_analysis?.frames_analysis) &&
    post.deepfake_analysis.frames_analysis.length > 0;
  const postLink = hasCompletedAnalysis ? `/posts/${post.id}/analysis` : `/posts/${post.id}`;

  const mediaUrl = post?.media_url 
    ? (post.media_url.startsWith('http') ? post.media_url : `${API_URL}${post.media_url}`)
    : '';

  if (!post || !post.media_url) {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.1 }}
    >
      <Link
        to={postLink}
        className="block"
        onMouseEnter={() => setIsHovered(true)}
        onMouseLeave={() => setIsHovered(false)}
      >
        <motion.div
          className="bg-white rounded-3xl overflow-hidden border-2 border-[#151616] shadow-[4px_4px_0px_0px_#151616] hover:shadow-[8px_8px_0px_0px_#151616] transition-all duration-300"
          whileHover={{ y: -5 }}
        >
          <div className="relative h-48">
            {post.media_type === 'video' ? (
              <video
                src={mediaUrl}
                className="w-full h-full object-cover"
                controls
              >
                Your browser does not support the video tag.
              </video>
            ) : (
              <img
                src={mediaUrl}
                alt={post.title}
                className="w-full h-full object-cover"
              />
            )}
            <div className="absolute top-4 right-4 text-sm font-bold text-white bg-[#151616]/50 px-2 py-1 rounded-full">
              {new Date(post.created_at).toLocaleDateString()}
            </div>
          </div>
          
          <div className="p-6">
            <h2 className="text-xl font-bold text-[#151616] mb-3">{post.title}</h2>
            <p className="text-[#151616]/70 mb-4 line-clamp-2">{post.content}</p>
            {post.media_type === 'video' && (
              <p className="text-xs text-[#151616]/70 mb-4">
                {hasCompletedAnalysis ? 'Click to view analysis details.' : `Analysis status: ${post.analysis_status || 'none'}`}
              </p>
            )}
            
            <div className="flex items-center justify-between">
              <div className="text-sm text-[#151616]/70">
                Posted by {post.profiles?.username || 'Anonymous'}
              </div>
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1 text-[#151616]/70">
                  <Heart className={`h-5 w-5 ${isHovered ? 'text-red-500' : ''}`} />
                  <span>{post.likes_count || 0}</span>
                </div>
                <div className="flex items-center space-x-1 text-[#151616]/70">
                  <MessageCircle className="h-5 w-5" />
                  <span>{post.comments_count || 0}</span>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </Link>
    </motion.div>
  );
};

export default function UserPosts() {
  const [posts, setPosts] = useState<Post[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    async function fetchUserPosts() {
      try {
        const response = await fetch(`${API_URL}/api/posts/user/posts`, {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });

        if (response.status === 401) {
          localStorage.removeItem('token');
          localStorage.removeItem('userId');
          navigate('/login');
          return;
        }

        const data = await response.json();
        console.log('Fetched posts:', data);
        if (!Array.isArray(data)) {
          console.error('Expected array of posts, got:', typeof data);
          setError(data?.message || 'Invalid data received from server');
          return;
        }
        setPosts(data);
      } catch (error) {
        console.error('Error fetching user posts:', error);
        setError('Failed to load posts');
      } finally {
        setLoading(false);
      }
    }
    fetchUserPosts();
  }, [navigate]);

  if (loading) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-gray-500">Loading posts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex justify-center items-center min-h-screen">
        <div className="text-red-500">{error}</div>
      </div>
    );
  }

  console.log('Current posts state:', posts);

  return (
    <section className="py-24 bg-[#ffffff] relative overflow-hidden mt-16">
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

      <div className="container mx-auto px-6">
        <div className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="inline-flex items-center gap-2 bg-[#151616] text-white rounded-full px-4 py-2 mb-4"
          >
            <Sparkles className="w-4 h-4 text-[#D6F32F]" />
            <span className="text-sm font-medium">Your Posts</span>
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="text-4xl md:text-5xl font-black text-[#151616] mb-6"
          >
            Your Content Dashboard
          </motion.h2>
        </div>

        <div className="grid gap-8 md:grid-cols-2 lg:grid-cols-3">
          {posts.length === 0 ? (
            <div className="col-span-full text-center text-gray-500">
              You haven't created any posts yet.
            </div>
          ) : (
            posts.map((post, index) => (
              <PostCard 
                key={post.id} 
                post={post} 
                index={index}
              />
            ))
          )}
        </div>
      </div>
    </section>
  );
} 
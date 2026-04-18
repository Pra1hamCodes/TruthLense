import { useEffect, useState, type ReactNode } from 'react';
import { jwtDecode } from 'jwt-decode';
import { AuthContext } from './auth-context';

interface Session {
  user: {
    id: string;
    username: string;
    email: string;
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      try {
        const decoded = jwtDecode<{ userId?: string; username?: string; email?: string }>(token);
        setSession({
          user: {
            id: decoded.userId || '',
            username: decoded.username || 'User',
            email: decoded.email || ''
          }
        });
      } catch (error) {
        console.error('Invalid token:', error);
        localStorage.removeItem('token');
      }
    }
    setLoading(false);
  }, []);

  return (
    <AuthContext.Provider value={{ session, loading, setSession }}>
      {children}
    </AuthContext.Provider>
  );
}
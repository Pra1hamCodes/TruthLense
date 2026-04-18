import { createContext, useContext } from 'react';

interface User {
  id: string;
  username: string;
  email: string;
}

interface Session {
  user: User;
}

export interface AuthContextType {
  session: Session | null;
  loading: boolean;
  setSession: (session: Session | null) => void;
}

export const AuthContext = createContext<AuthContextType>({
  session: null,
  loading: true,
  setSession: () => {},
});

export const useAuth = () => useContext(AuthContext);

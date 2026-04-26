const trimTrailingSlash = (value?: string) => (value ? value.replace(/\/+$/, '') : '');

const apiBaseFromEnv = trimTrailingSlash(import.meta.env.VITE_API_BASE_URL);
const backendTarget = trimTrailingSlash(import.meta.env.VITE_BACKEND_TARGET);

export const API_BASE_URL = apiBaseFromEnv || backendTarget || '';
export const API_API_PREFIX = `${API_BASE_URL}/api`;

export const FLASK_MAIN_URL = trimTrailingSlash(import.meta.env.VITE_FLASK_MAIN_URL) || 'http://localhost:5000';
export const FLASK_OLD_URL = trimTrailingSlash(import.meta.env.VITE_FLASK_OLD_URL) || 'http://localhost:5001';

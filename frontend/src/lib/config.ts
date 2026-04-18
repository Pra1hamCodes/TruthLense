const trimTrailingSlash = (value?: string) => (value ? value.replace(/\/+$/, '') : '');

const apiBaseFromEnv = trimTrailingSlash(import.meta.env.VITE_API_BASE_URL);

export const API_BASE_URL = apiBaseFromEnv || '';
export const API_API_PREFIX = `${API_BASE_URL}/api`;

export const FLASK_MAIN_URL = trimTrailingSlash(import.meta.env.VITE_FLASK_MAIN_URL) || 'http://localhost:5000';
export const FLASK_OLD_URL = trimTrailingSlash(import.meta.env.VITE_FLASK_OLD_URL) || 'http://localhost:5001';

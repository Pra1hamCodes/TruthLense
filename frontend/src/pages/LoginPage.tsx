import type React from "react"
import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import { motion } from "framer-motion"
import { Mail, Lock, ArrowRight, AlertCircle } from "lucide-react"
import axios, { AxiosError } from "axios"
import { jwtDecode } from "jwt-decode"
import { API_BASE_URL } from "../lib/config"
import { useAuth } from "@/contexts/auth-context"

import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import { Checkbox } from "../components/ui/checkbox"

export function LoginPage() {
    const [email, setEmail] = useState("")
    const [password, setPassword] = useState("")
    const [error, setError] = useState("")
    const [isLoading, setIsLoading] = useState(false)
    const navigate = useNavigate()
    const { setSession } = useAuth()

    interface TokenPayload {
        userId: string;
        username?: string;
        email?: string;
    }

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault()
        setError("")
        setIsLoading(true)

        try {
            const response = await axios.post(`${API_BASE_URL}/signin`, {
                username: email,
                password: password
            })

            const token = response.data?.token
            if (!token || typeof token !== 'string') {
                throw new Error('Missing authentication token in response')
            }

            let decoded: TokenPayload
            try {
                decoded = jwtDecode<TokenPayload>(token)
            } catch (decodeError) {
                console.error('Failed to decode token:', decodeError)
                throw new Error('Invalid authentication token')
            }

            localStorage.setItem("token", token)
            if (decoded.userId) {
                localStorage.setItem("userId", decoded.userId)
            }

            setSession({
                user: {
                    id: decoded.userId || '',
                    username: decoded.username || 'User',
                    email: decoded.email || ''
                }
            })

            navigate("/dashboard")
        } catch (err: unknown) {
            const axiosError = err as AxiosError<{ message?: string }>;
            setError(axiosError.response?.data?.message || (err instanceof Error ? err.message : "Sign in failed. Please try again."))
            setIsLoading(false)
        }
    }

    return (
        <div className="min-h-screen bg-[#FFFFF4] relative">
            {/* Dot Grid Background */}
            <div
                className="absolute inset-0 pointer-events-none"
                style={{
                    backgroundImage: `radial-gradient(#151616 1px, transparent 1px)`,
                    backgroundSize: "24px 24px",
                    opacity: "0.1",
                }}
            />

            {/* Main Content */}
            <div className="container mx-auto px-4 h-screen flex items-center justify-center relative">
                <motion.div 
                    initial={{ opacity: 0, y: 20 }} 
                    animate={{ opacity: 1, y: 0 }} 
                    className="w-full max-w-md"
                >
                    <div className="bg-white p-8 rounded-lg shadow-md border-2 border-[#151616]">
                        <div className="text-center space-y-2 mb-6">
                            <h2 className="text-3xl font-bold text-[#151616]">Welcome Back</h2>
                            <p className="text-[#151616]">
                                Don't have an account?{" "}
                                <Link
                                    to="/register"
                                    className="font-medium hover:underline"
                                    style={{ color: "#151616", textDecorationColor: "#D6F32F" }}
                                >
                                    Sign up here
                                </Link>
                            </p>
                        </div>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            {error && (
                                <motion.div
                                    initial={{ opacity: 0, y: -10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    className="flex items-start gap-3 p-4 bg-red-50 rounded-xl border-2 border-red-500"
                                >
                                    <AlertCircle className="w-5 h-5 text-red-500 shrink-0 mt-0.5" />
                                    <p className="text-sm text-red-600">{error}</p>
                                </motion.div>
                            )}

                            <div className="space-y-2">
                                <Label htmlFor="email">Email Address</Label>
                                <div className="relative">
                                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                                    <Input
                                        id="email"
                                        type="email"
                                        placeholder="Enter your email"
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        className="pl-10"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="space-y-2">
                                <Label htmlFor="password">Password</Label>
                                <div className="relative">
                                    <Lock className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                                    <Input
                                        id="password"
                                        type="password"
                                        placeholder="Enter your password"
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        className="pl-10"
                                        required
                                    />
                                </div>
                            </div>

                            <div className="flex items-center justify-between">
                                <div className="flex items-center space-x-2">
                                    <Checkbox id="remember" />
                                    <Label htmlFor="remember" className="text-sm text-gray-600">
                                        Remember me
                                    </Label>
                                </div>
                                <Link to="/forgot-password" className="text-sm text-gray-600 hover:underline">
                                    Forgot password?
                                </Link>
                            </div>

                            <Button
                                type="submit"
                                disabled={isLoading}
                                className="w-full"
                                style={{ backgroundColor: "#D6F32F", color: "#151616" }}
                            >
                                {isLoading ? (
                                    "Signing in..."
                                ) : (
                                    <>
                                        Sign in
                                        <ArrowRight className="ml-2 h-4 w-4" />
                                    </>
                                )}
                            </Button>
                        </form>
                    </div>
                </motion.div>
            </div>
        </div>
    )
}
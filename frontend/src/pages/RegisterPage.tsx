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

export function RegisterPage() {
  const [formData, setFormData] = useState({
    email: "",
    password: "",
    firstName: "",
    lastName: "",
  })
  const [error, setError] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const navigate = useNavigate()
  const { setSession } = useAuth()

  interface TokenPayload {
    userId: string;
    username?: string;
    email?: string;
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError("")
    setIsLoading(true)

    try {
      const response = await axios.post(`${API_BASE_URL}/signup`, {
        username: formData.email,
        firstName: formData.firstName,
        lastName: formData.lastName,
        password: formData.password
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
          username: decoded.username || formData.email,
          email: decoded.email || formData.email
        }
      })

      navigate("/dashboard")
    } catch (err: unknown) {
      const axiosError = err as AxiosError<{ message?: string }>;
      setError(axiosError.response?.data?.message || (err instanceof Error ? err.message : "Sign up failed. Please try again."))
      setIsLoading(false)
    }
  }

  return (
    <div className="relative min-h-screen flex flex-row items-center justify-center">
      {/* Background pattern layer */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage: `radial-gradient(#151616 1px, transparent 1px)`,
          backgroundSize: "24px 24px",
          opacity: "0.1",
        }}
      />
      
      {/* Content layer */}
      <div className="relative container mx-auto px-4 py-8">
        <motion.div 
          initial={{ opacity: 0, y: 20 }} 
          animate={{ opacity: 1, y: 0 }} 
          className="w-full max-w-2xl mx-auto"
        >
          <div className="bg-white p-8 rounded-lg shadow-md border-2 border-[#151616]">
            <div className="text-center space-y-2 mb-6">
              <h2 className="text-3xl font-bold text-[#151616]">Create Account</h2>
              <p className="text-[#151616]">
                Already have an account?{" "}
                <Link
                  to="/login"
                  className="font-medium hover:underline"
                  style={{ color: "#151616", textDecorationColor: "#D6F32F" }}
                >
                  Sign in here
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

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <Input
                      id="firstName"
                      name="firstName"
                      type="text"
                      placeholder="Enter your first name"
                      value={formData.firstName}
                      onChange={handleChange}
                      className="pl-10"
                      required
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="lastName">Last Name</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                    <Input
                      id="lastName"
                      name="lastName"
                      type="text"
                      placeholder="Enter your last name"
                      value={formData.lastName}
                      onChange={handleChange}
                      className="pl-10"
                      required
                    />
                  </div>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    placeholder="Enter your email"
                    value={formData.email}
                    onChange={handleChange}
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
                    name="password"
                    type="password"
                    placeholder="Create a password"
                    value={formData.password}
                    onChange={handleChange}
                    className="pl-10"
                    required
                  />
                </div>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full"
                style={{ backgroundColor: "#D6F32F", color: "#151616" }}
              >
                {isLoading ? (
                  "Creating Account..."
                ) : (
                  <>
                    Create Account
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

export default RegisterPage
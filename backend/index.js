// backend/index.js
const express = require('express');
const cors = require("cors");
const jwt = require("jsonwebtoken");
const zod = require("zod");
const bcrypt = require('bcryptjs');
const crypto = require('crypto');
require('dotenv').config();
const path = require('path');
const fs = require('fs');

const normalizeUsername = (raw) =>
  typeof raw === 'string' ? raw.trim().toLowerCase() : raw;

// Import db connection
require('./db');

const { User } = require("./db");
const postRoutes = require('./routes/post');
const evaluationRoutes = require('./routes/evaluation');
const authMiddleware = require('./middleware/auth');

const JWT_SECRET = process.env.JWT_SECRET;
if (!JWT_SECRET) {
  throw new Error('JWT_SECRET is not configured. Set it in backend/.env before starting.');
}
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '7d';
const BCRYPT_ROUNDS = 10;

const allowedOrigins = (process.env.CORS_ORIGINS || 'http://localhost:5173,http://localhost:3000')
  .split(',')
  .map((o) => o.trim())
  .filter(Boolean);

const app = express();
app.use(cors({
    origin: (origin, cb) => {
        if (!origin || allowedOrigins.includes(origin)) return cb(null, true);
        return cb(null, false);
    },
    credentials: true,
}));
app.use(express.json());

// Create uploads directory if it doesn't exist
const uploadsDir = path.join(__dirname, 'uploads');
if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
}

// Serve static files from the uploads directory
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// Serve frames from BOTH Flask servers' frames dirs. express.static calls next()
// on miss, so the second mount catches files that live in flask_server_2/frames
// (e.g. frames returned by S1's /analyze-frames or S2's /analyze).
app.use('/uploads/frames', express.static(path.join(__dirname, 'flask_server/frames')));
app.use('/uploads/frames', express.static(path.join(__dirname, 'flask_server_2/frames')));

// SIGNUP ROUTE //
const signupBody = zod.object({
    username: zod.string().email(),
    firstName: zod.string(),
    lastName: zod.string(),
    password: zod.string()
})

app.post("/signup", async (req, res) => {
    try {
        const { success } = signupBody.safeParse(req.body)
        if (!success) {
            return res.status(400).json({
                message: "Incorrect inputs"
            })
        }

        const username = normalizeUsername(req.body.username);

        const existingUser = await User.findOne({
            username
        })

        if (existingUser) {
            return res.status(409).json({
                message: "Email already taken"
            })
        }

        const passwordHash = await bcrypt.hash(req.body.password, BCRYPT_ROUNDS);

        const user = await User.create({
            username,
            password: passwordHash,
            firstName: req.body.firstName,
            lastName: req.body.lastName,
        })
        const userId = user._id;

        const token = jwt.sign({
            userId,
            username: user.username,
            email: user.username
        }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });

        res.json({
            message: "User created successfully",
            token: token
        })
    } catch (error) {
        console.error('Signup error:', error);
        if (error.code === 'USER_EXISTS') {
            return res.status(409).json({
                message: "Email already taken"
            });
        }

        res.status(500).json({
            message: 'Unable to create user'
        });
    }
})

// SIGNIN ROUTE //
const signinBody = zod.object({
    username: zod.string().email(),
    password: zod.string()
})

app.post("/signin", async (req, res) => {
    try {
        const { success } = signinBody.safeParse(req.body)
        if (!success) {
            return res.status(400).json({
                message: "Incorrect inputs"
            })
        }

        const username = normalizeUsername(req.body.username);

        const user = await User.findOne({
            username
        });

        if (!user) {
            return res.status(401).json({
                message: "Invalid credentials"
            });
        }

        const storedPassword = typeof user.password === 'string' ? user.password : '';
        const looksHashed = /^\$2[aby]\$/.test(storedPassword);

        let passwordMatches = false;
        if (looksHashed) {
            passwordMatches = await bcrypt.compare(req.body.password, storedPassword);
        } else if (storedPassword) {
            const a = Buffer.from(storedPassword, 'utf8');
            const b = Buffer.from(req.body.password, 'utf8');
            if (a.length === b.length && crypto.timingSafeEqual(a, b)) {
                passwordMatches = true;
                try {
                    const newHash = await bcrypt.hash(req.body.password, BCRYPT_ROUNDS);
                    const updated = await User.updatePassword(user._id, newHash);
                    if (!updated) {
                        console.error('Password hash migration returned no record for user', String(user._id));
                    }
                } catch (migrationError) {
                    console.error('Password hash migration failed:', migrationError);
                }
            }
        }

        if (!passwordMatches) {
            return res.status(401).json({
                message: "Invalid credentials"
            });
        }

        const token = jwt.sign({
            userId: user._id,
            username: user.username,
            email: user.username
        }, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });

        res.json({
            token: token,
            user: {
                username: user.username,
                email: user.username
            }
        })
    } catch (error) {
        console.error('Signin error:', error);
        res.status(500).json({
            message: 'Unable to log in'
        });
    }
})

// Routes
app.use('/api/posts', postRoutes);
app.use('/api/evaluation', evaluationRoutes);

const PORT = Number(process.env.PORT) || 3000;

function startServer(port, attemptsLeft) {
  const server = app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
  });
  server.on('error', (err) => {
    if (err.code === 'EADDRINUSE' && attemptsLeft > 0) {
      console.log(`Port ${port} is busy, trying ${port + 1}`);
      startServer(port + 1, attemptsLeft - 1);
    } else {
      console.error('Server error:', err);
      process.exit(1);
    }
  });
}

startServer(PORT, 5);
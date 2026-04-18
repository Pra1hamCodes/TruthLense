// backend/db.js
const fs = require('fs');
const path = require('path');
const dns = require('dns');
const mongoose = require('mongoose');
require('dotenv').config();

dns.setServers(['8.8.8.8', '1.1.1.1']);

const hasMongoUri = typeof process.env.MONGO_URI === 'string' && process.env.MONGO_URI.trim().length > 0;
const usersStorePath = path.join(__dirname, 'data', 'users.json');
let mongoAvailable = false;

function isMongoReady() {
    return mongoAvailable;
}

function ensureUsersStore() {
    const storeDir = path.dirname(usersStorePath);
    if (!fs.existsSync(storeDir)) {
        fs.mkdirSync(storeDir, { recursive: true });
    }

    if (!fs.existsSync(usersStorePath)) {
        fs.writeFileSync(usersStorePath, '[]', 'utf8');
    }
}

function readUsers() {
    ensureUsersStore();
    const raw = fs.readFileSync(usersStorePath, 'utf8');
    return JSON.parse(raw || '[]');
}

function writeUsers(users) {
    ensureUsersStore();
    fs.writeFileSync(usersStorePath, JSON.stringify(users, null, 2), 'utf8');
}

function normalizeUsername(username) {
    return String(username || '').trim().toLowerCase();
}

if (hasMongoUri) {
    mongoose.connect(process.env.MONGO_URI)
    .then(() => {
        mongoAvailable = true;
        console.log('Successfully connected to DB');
    })
    .catch((error) => {
        mongoAvailable = false;
        console.error('Error connecting to DB:', error);
    });

    mongoose.connection.on('disconnected', () => {
        mongoAvailable = false;
        console.warn('MongoDB disconnected — falling back to file store.');
    });
    mongoose.connection.on('reconnected', () => {
        mongoAvailable = true;
        console.log('MongoDB reconnected.');
    });
    mongoose.connection.on('error', (err) => {
        mongoAvailable = false;
        console.error('MongoDB connection error:', err);
    });
} else {
    console.warn('MONGO_URI is not configured. Using local file-backed users store for signup/signin.');
}

const userSchema = new mongoose.Schema({
    username: {
        type: String,
        required: true,
        unique: true,
        trim: true,
        lowercase: true,
    },
    password: {
        type: String,
        required: true,
        minLength: 6,
    },
    firstName: {
        type: String,
        required: true,
        trim: true,
        maxLength: 50,
    },
    lastName: {
        type: String,
        required: true,
        trim: true,
        maxLength: 50,
    },
}, { timestamps: true });

const mongoUserModel = hasMongoUri ? mongoose.model('User', userSchema) : null;

const fileUserStore = {
    async findOne(query) {
        const users = readUsers();
        const username = normalizeUsername(query?.username);

        return users.find((user) => {
            return username ? user.username === username : true;
        }) || null;
    },

    async create(data) {
        const users = readUsers();
        const username = normalizeUsername(data.username);

        const existingUser = users.find((user) => user.username === username);
        if (existingUser) {
            const error = new Error('User already exists');
            error.code = 'USER_EXISTS';
            throw error;
        }

        const newUser = {
            _id: new mongoose.Types.ObjectId().toString(),
            username,
            password: data.password,
            firstName: data.firstName,
            lastName: data.lastName,
            createdAt: new Date().toISOString(),
            updatedAt: new Date().toISOString(),
        };

        users.push(newUser);
        writeUsers(users);
        return newUser;
    },

    async updatePassword(userId, newPassword) {
        const users = readUsers();
        const index = users.findIndex((u) => u._id === userId);
        if (index === -1) return null;
        users[index].password = newPassword;
        users[index].updatedAt = new Date().toISOString();
        writeUsers(users);
        return users[index];
    },
};

const User = {
    async findOne(query) {
        if (isMongoReady() && mongoUserModel) {
            return mongoUserModel.findOne(query);
        }

        return fileUserStore.findOne(query);
    },

    async create(data) {
        if (isMongoReady() && mongoUserModel) {
            return mongoUserModel.create(data);
        }

        return fileUserStore.create(data);
    },

    async updatePassword(userId, newPassword) {
        if (isMongoReady() && mongoUserModel) {
            return mongoUserModel.findByIdAndUpdate(
                userId,
                { password: newPassword },
                { new: true }
            );
        }
        return fileUserStore.updatePassword(userId, newPassword);
    },
};

module.exports = {
    User,
    isMongoReady,
};
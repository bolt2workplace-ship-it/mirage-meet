import express from 'express';
import { createServer } from 'http';
import { Server } from 'socket.io';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const httpServer = createServer(app);
const io = new Server(httpServer, {
  cors: {
    origin: ['http://localhost:5173', 'http://localhost:3000'],
    methods: ['GET', 'POST'],
    credentials: true,
  },
});

app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.sendStatus(200);
  next();
});

const rooms = new Map();

app.get('/', (req, res) => {
  res.json({
    name: 'Mirage Meet Signaling Server',
    status: 'running',
    version: '1.0.0',
    endpoints: {
      'POST /create-room': 'Create a new meeting room',
      'Socket.IO': 'WebRTC signaling and real-time communication'
    }
  });
});

app.post('/create-room', (req, res) => {
  const roomId = generateRoomId();
  rooms.set(roomId, {
    id: roomId,
    admin: null,
    participants: new Map(),
  });
  res.json({ roomId });
});

io.on('connection', (socket) => {
  console.log('User connected:', socket.id);

  socket.on('create-room', () => {
    const roomId = generateRoomId();
    rooms.set(roomId, {
      id: roomId,
      admin: socket.id,
      participants: new Map([[socket.id, { id: socket.id, isAdmin: true, displayName: 'Host' }]]),
    });
    socket.join(roomId);
    socket.emit('room-created', { roomId, isAdmin: true });
    console.log('Room created:', roomId);
  });

  socket.on('join-room', ({ roomId, displayName }) => {
    const room = rooms.get(roomId);
    if (!room) {
      socket.emit('error', { message: 'Room not found' });
      return;
    }

    const isAdmin = room.admin === null || room.admin === socket.id;
    if (isAdmin && room.admin === null) {
      room.admin = socket.id;
    }

    room.participants.set(socket.id, {
      id: socket.id,
      isAdmin,
      displayName: displayName || 'Guest',
    });

    socket.join(roomId);
    socket.emit('room-joined', {
      roomId,
      isAdmin,
      participant: { id: socket.id, isAdmin, displayName: displayName || 'Guest' },
      participants: Array.from(room.participants.values()),
    });

    socket.to(roomId).emit('user-joined', {
      user: { id: socket.id, isAdmin, displayName: displayName || 'Guest' },
    });

    console.log('User joined room:', roomId, socket.id);
  });

  socket.on('signal', ({ to, signal }) => {
    io.to(to).emit('signal', { from: socket.id, signal });
  });

  socket.on('disconnect', () => {
    rooms.forEach((room, roomId) => {
      if (room.participants.has(socket.id)) {
        room.participants.delete(socket.id);
        socket.to(roomId).emit('user-left', { userId: socket.id });
        console.log('User left room:', roomId, socket.id);
      }
    });
  });

  socket.on('toggle-camera', ({ roomId, enabled }) => {
    const room = rooms.get(roomId);
    if (room && room.participants.has(socket.id)) {
      const participant = room.participants.get(socket.id);
      participant.cameraEnabled = enabled;
      socket.to(roomId).emit('participant-updated', {
        userId: socket.id,
        updates: { cameraEnabled: enabled },
      });
    }
  });

  socket.on('toggle-microphone', ({ roomId, enabled }) => {
    const room = rooms.get(roomId);
    if (room && room.participants.has(socket.id)) {
      const participant = room.participants.get(socket.id);
      participant.microphoneEnabled = enabled;
      socket.to(roomId).emit('participant-updated', {
        userId: socket.id,
        updates: { microphoneEnabled: enabled },
      });
    }
  });
});

function generateRoomId() {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let result = '';
  for (let i = 0; i < 9; i++) {
    if (i > 0 && i % 3 === 0) result += '-';
    result += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return result;
}

const PORT = process.env.PORT || 3001;
httpServer.listen(PORT, () => {
  console.log(`Signaling server running on port ${PORT}`);
});

const express = require('express');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);

// Configure Socket.IO server with CORS enabled for simplicity
const io = new Server(server, {
  cors: {
    origin: "*", // Allow all origins for local testing
    methods: ["GET", "POST"]
  }
});

// Serve HTML page from the current directory
app.use(express.static("public"));

io.on('connection', (socket) => {
  console.log('Client connected');

  socket.on('detection_data', (data) => {
    if (data && data.frame && data.frame.length > 1000) { 
      io.emit('detection_data', data);
    }
  });

  socket.on('disconnect', () => {
    console.log('Client disconnected');
  });
});

const PORT = 3000;
server.listen(PORT, () => console.log(`Server running at http://localhost:${PORT}`));
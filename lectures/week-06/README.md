# Week 6: Frontend Integration & User Experience

## 📚 Lecture Overview

This week we focus on creating beautiful, responsive frontends for AI applications. You'll learn how to build modern web interfaces using React, Next.js, and Typescript, with real-time communication, intuitive UX design, and performance optimization.

## 🎯 Learning Objectives

By the end of this week, you will:
- Build modern frontend applications for AI systems
- Implement real-time communication with WebSockets
- Design intuitive user interfaces and experiences
- Optimize frontend performance and accessibility
- Create responsive and mobile-friendly applications

## 📖 Lecture Content

### 1. Frontend Frameworks and Technologies

**Modern Frontend Stack:**
- **React**: Component-based UI library
- **Next.js**: Progressive JavaScript framework
- **Vanilla JavaScript**: Modern ES6+ features
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework

**Framework Comparison:**
```javascript
// React Component
function AIChatInterface() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');

  const sendMessage = async () => {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input })
    });
    const data = await response.json();
    setMessages([...messages, { user: input, ai: data.response }]);
    setInput('');
  };

  return (
    <div className="chat-container">
      {messages.map((msg, i) => (
        <div key={i} className="message">
          <div className="user-message">{msg.user}</div>
          <div className="ai-message">{msg.ai}</div>
        </div>
      ))}
      <input 
        value={input} 
        onChange={(e) => setInput(e.target.value)}
        placeholder="Type your message..."
      />
      <button onClick={sendMessage}>Send</button>
    </div>
  );
}
```

### 2. React for AI Applications

**React Setup with Vite:**
```bash
npm create vite@latest ai-chat-app -- --template react
cd ai-chat-app
npm install
npm run dev
```

**AI Chat Component:**
```jsx
import React, { useState, useEffect, useRef } from 'react';
import './ChatInterface.css';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { type: 'user', content: input, timestamp: new Date() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input })
      });

      const data = await response.json();
      const aiMessage = { 
        type: 'ai', 
        content: data.response, 
        timestamp: new Date() 
      };
      
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>AI Assistant</h2>
        <div className="status-indicator">
          {isLoading ? 'Thinking...' : 'Online'}
        </div>
      </div>

      <div className="messages-container">
        {messages.map((message, index) => (
          <div 
            key={index} 
            className={`message ${message.type}-message`}
          >
            <div className="message-content">{message.content}</div>
            <div className="message-timestamp">
              {message.timestamp.toLocaleTimeString()}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="message ai-message">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-container">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Type your message..."
          disabled={isLoading}
        />
        <button 
          onClick={sendMessage} 
          disabled={isLoading || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
};

export default ChatInterface;
```

**CSS Styling:**
```css
.chat-container {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f5f5f5;
}

.chat-header {
  background: #2c3e50;
  color: white;
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  max-width: 70%;
  padding: 0.75rem;
  border-radius: 1rem;
  animation: fadeIn 0.3s ease-in;
}

.user-message {
  align-self: flex-end;
  background: #3498db;
  color: white;
}

.ai-message {
  align-self: flex-start;
  background: white;
  border: 1px solid #ddd;
}

.input-container {
  padding: 1rem;
  background: white;
  border-top: 1px solid #ddd;
  display: flex;
  gap: 0.5rem;
}

.input-container textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #ddd;
  border-radius: 0.5rem;
  resize: none;
  min-height: 40px;
  max-height: 120px;
}

.input-container button {
  padding: 0.75rem 1.5rem;
  background: #3498db;
  color: white;
  border: none;
  border-radius: 0.5rem;
  cursor: pointer;
  transition: background 0.2s;
}

.input-container button:hover {
  background: #2980b9;
}

.input-container button:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
}

.typing-indicator {
  display: flex;
  gap: 0.25rem;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  background: #95a5a6;
  border-radius: 50%;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }

@keyframes typing {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1); }
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
```

### 3. Real-time Communication

**WebSocket Implementation:**
```javascript
// WebSocket hook for React
import { useEffect, useRef, useState } from 'react';

const useWebSocket = (url) => {
  const [isConnected, setIsConnected] = useState(false);
  const [messages, setMessages] = useState([]);
  const wsRef = useRef(null);

  useEffect(() => {
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setMessages(prev => [...prev, data]);
    };

    ws.onclose = () => {
      setIsConnected(false);
      console.log('WebSocket disconnected');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      ws.close();
    };
  }, [url]);

  const sendMessage = (message) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  };

  return { isConnected, messages, sendMessage };
};

// Real-time chat component
const RealTimeChat = () => {
  const { isConnected, messages, sendMessage } = useWebSocket('ws://localhost:8000/ws');
  const [input, setInput] = useState('');

  const handleSend = () => {
    if (input.trim()) {
      sendMessage({ type: 'message', content: input });
      setInput('');
    }
  };

  return (
    <div className="realtime-chat">
      <div className="connection-status">
        Status: {isConnected ? 'Connected' : 'Disconnected'}
      </div>
      
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.type}`}>
            {msg.content}
          </div>
        ))}
      </div>

      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Type a message..."
        />
        <button onClick={handleSend}>Send</button>
      </div>
    </div>
  );
};
```

### 4. Vue.js Alternative

**Vue.js Chat Component:**
```vue
<template>
  <div class="chat-app">
    <div class="chat-header">
      <h2>AI Assistant</h2>
      <div class="status" :class="{ online: isConnected }">
        {{ isConnected ? 'Online' : 'Offline' }}
      </div>
    </div>

    <div class="messages" ref="messagesContainer">
      <div
        v-for="(message, index) in messages"
        :key="index"
        :class="['message', message.type]"
      >
        <div class="message-content">{{ message.content }}</div>
        <div class="message-time">{{ formatTime(message.timestamp) }}</div>
      </div>
      
      <div v-if="isTyping" class="message ai typing">
        <div class="typing-indicator">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>

    <div class="input-area">
      <textarea
        v-model="inputMessage"
        @keydown.enter.prevent="sendMessage"
        placeholder="Type your message..."
        :disabled="!isConnected"
      ></textarea>
      <button @click="sendMessage" :disabled="!isConnected || !inputMessage.trim()">
        Send
      </button>
    </div>
  </div>
</template>

<script>
export default {
  name: 'ChatApp',
  data() {
    return {
      messages: [],
      inputMessage: '',
      isConnected: false,
      isTyping: false,
      ws: null
    }
  },
  
  mounted() {
    this.connectWebSocket();
  },
  
  methods: {
    connectWebSocket() {
      this.ws = new WebSocket('ws://localhost:8000/ws');
      
      this.ws.onopen = () => {
        this.isConnected = true;
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleIncomingMessage(data);
      };
      
      this.ws.onclose = () => {
        this.isConnected = false;
        // Reconnect after 5 seconds
        setTimeout(() => this.connectWebSocket(), 5000);
      };
    },
    
    sendMessage() {
      if (!this.inputMessage.trim() || !this.isConnected) return;
      
      const message = {
        type: 'user',
        content: this.inputMessage,
        timestamp: new Date()
      };
      
      this.messages.push(message);
      this.ws.send(JSON.stringify(message));
      this.inputMessage = '';
    },
    
    handleIncomingMessage(data) {
      if (data.type === 'typing') {
        this.isTyping = data.isTyping;
      } else {
        this.messages.push({
          type: 'ai',
          content: data.content,
          timestamp: new Date()
        });
        this.isTyping = false;
      }
      
      this.$nextTick(() => {
        this.scrollToBottom();
      });
    },
    
    scrollToBottom() {
      const container = this.$refs.messagesContainer;
      container.scrollTop = container.scrollHeight;
    },
    
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString();
    }
  },
  
  beforeUnmount() {
    if (this.ws) {
      this.ws.close();
    }
  }
}
</script>

<style scoped>
.chat-app {
  max-width: 800px;
  margin: 0 auto;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f8f9fa;
}

.chat-header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.message {
  max-width: 75%;
  padding: 0.75rem 1rem;
  border-radius: 1.5rem;
  position: relative;
}

.message.user {
  align-self: flex-end;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.message.ai {
  align-self: flex-start;
  background: white;
  border: 1px solid #e9ecef;
  color: #333;
}

.input-area {
  padding: 1rem;
  background: white;
  border-top: 1px solid #e9ecef;
  display: flex;
  gap: 0.5rem;
}

.input-area textarea {
  flex: 1;
  padding: 0.75rem;
  border: 1px solid #e9ecef;
  border-radius: 1.5rem;
  resize: none;
  min-height: 40px;
  font-family: inherit;
}

.input-area button {
  padding: 0.75rem 1.5rem;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 1.5rem;
  cursor: pointer;
  transition: transform 0.2s;
}

.input-area button:hover {
  transform: translateY(-2px);
}

.input-area button:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
}
</style>
```

### 5. Advanced UI Components

**File Upload Component:**
```jsx
import React, { useState, useRef } from 'react';

const FileUpload = ({ onUpload, acceptedTypes = ['image/*'] }) => {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFiles(e.dataTransfer.files);
    }
  };

  const handleFiles = async (files) => {
    setUploading(true);
    setProgress(0);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      await uploadFile(file);
      setProgress(((i + 1) / files.length) * 100);
    }

    setUploading(false);
    setProgress(0);
  };

  const uploadFile = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const result = await response.json();
        onUpload(result);
      }
    } catch (error) {
      console.error('Upload failed:', error);
    }
  };

  return (
    <div
      className={`file-upload ${dragActive ? 'drag-active' : ''}`}
      onDragEnter={handleDrag}
      onDragLeave={handleDrag}
      onDragOver={handleDrag}
      onDrop={handleDrop}
    >
      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept={acceptedTypes.join(',')}
        onChange={(e) => handleFiles(e.target.files)}
        style={{ display: 'none' }}
      />
      
      <div className="upload-content">
        <div className="upload-icon">📁</div>
        <p>Drag and drop files here or click to browse</p>
        <button 
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
        >
          Choose Files
        </button>
      </div>

      {uploading && (
        <div className="upload-progress">
          <div className="progress-bar">
            <div 
              className="progress-fill" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <span>{Math.round(progress)}%</span>
        </div>
      )}
    </div>
  );
};
```

### 6. Performance Optimization

**React Performance Optimization:**
```jsx
import React, { memo, useCallback, useMemo } from 'react';

// Memoized component
const MessageItem = memo(({ message, onDelete }) => {
  const handleDelete = useCallback(() => {
    onDelete(message.id);
  }, [message.id, onDelete]);

  const formattedTime = useMemo(() => {
    return new Date(message.timestamp).toLocaleTimeString();
  }, [message.timestamp]);

  return (
    <div className="message-item">
      <div className="message-content">{message.content}</div>
      <div className="message-meta">
        <span className="time">{formattedTime}</span>
        <button onClick={handleDelete}>Delete</button>
      </div>
    </div>
  );
});

// Virtual scrolling for large lists
import { FixedSizeList as List } from 'react-window';

const VirtualizedMessageList = ({ messages }) => {
  const Row = ({ index, style }) => (
    <div style={style}>
      <MessageItem message={messages[index]} />
    </div>
  );

  return (
    <List
      height={400}
      itemCount={messages.length}
      itemSize={80}
      width="100%"
    >
      {Row}
    </List>
  );
};
```

### 7. Accessibility and UX

**Accessible Chat Interface:**
```jsx
const AccessibleChat = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const messagesRef = useRef(null);

  // Announce new messages to screen readers
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      const announcement = document.createElement('div');
      announcement.setAttribute('aria-live', 'polite');
      announcement.setAttribute('aria-atomic', 'true');
      announcement.className = 'sr-only';
      announcement.textContent = `New message: ${lastMessage.content}`;
      document.body.appendChild(announcement);
      
      setTimeout(() => {
        document.body.removeChild(announcement);
      }, 1000);
    }
  }, [messages]);

  return (
    <div className="accessible-chat" role="main">
      <h1 id="chat-title">AI Assistant Chat</h1>
      
      <div 
        ref={messagesRef}
        className="messages"
        role="log"
        aria-labelledby="chat-title"
        aria-live="polite"
      >
        {messages.map((message, index) => (
          <div
            key={index}
            className={`message ${message.type}`}
            role="article"
            aria-label={`${message.type} message`}
          >
            <div className="message-content">{message.content}</div>
            <div className="message-time" aria-label="Message time">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
      </div>

      <form 
        className="input-form"
        onSubmit={(e) => {
          e.preventDefault();
          // Handle message submission
        }}
      >
        <label htmlFor="message-input" className="sr-only">
          Type your message
        </label>
        <textarea
          id="message-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          aria-describedby="message-help"
        />
        <div id="message-help" className="help-text">
          Press Enter to send, Shift+Enter for new line
        </div>
        <button type="submit" aria-label="Send message">
          Send
        </button>
      </form>
    </div>
  );
};
```

## 🛠️ Workshop Preparation

**Pre-workshop Tasks:**
1. Set up React development environment
2. Install UI libraries: `npm install @mui/material @emotion/react @emotion/styled`
3. Review modern CSS and JavaScript features
4. Practice with WebSocket connections

**Workshop Goals:**
- Build a complete chat interface
- Implement real-time communication
- Create responsive design
- Add accessibility features

## 📚 Additional Resources

**Reading Materials:**
- [React Documentation](https://react.dev/)
- [Vue.js Guide](https://vuejs.org/guide/)
- [WebSocket API](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket)
- [Accessibility Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)

**Videos:**
- [React Tutorial](https://www.youtube.com/watch?v=bMknfKXIFA8)
- [Vue.js Course](https://www.youtube.com/watch?v=FXpIoQ_rT_c)
- [WebSocket Guide](https://www.youtube.com/watch?v=8ARodQ4Wlf4)

**Tools & Platforms:**
- [Vite](https://vitejs.dev/) - Fast build tool
- [Tailwind CSS](https://tailwindcss.com/) - Utility-first CSS
- [Framer Motion](https://www.framer.com/motion/) - Animation library
- [Storybook](https://storybook.js.org/) - Component development

## 📝 Assignment

**Due Date**: End of Week 6

**Tasks:**
1. Build a complete frontend for your AI application
2. Implement real-time features
3. Ensure responsive design
4. Add accessibility features

**Submission Requirements:**
- Working frontend application
- Responsive design implementation
- Accessibility compliance
- Performance optimization

## 🎯 Next Week Preview

**Week 7: Automation & CI/CD Pipelines**
- GitHub Actions workflows
- Automated testing and deployment
- Continuous integration
- DevOps best practices

## 💡 Advanced Concepts

**Progressive Web Apps (PWA):**
- Service workers
- Offline functionality
- Push notifications
- App-like experience

**State Management:**
- Redux and Zustand
- Context API
- Server state management
- Optimistic updates

## 🚀 Best Practices

**Performance:**
- Code splitting and lazy loading
- Image optimization
- Bundle size optimization
- Caching strategies

**User Experience:**
- Loading states and skeletons
- Error boundaries
- Progressive enhancement
- Mobile-first design

**Accessibility:**
- Semantic HTML
- ARIA labels and roles
- Keyboard navigation
- Screen reader support

## 📞 Support

- **Element Support**: `#support`
- **Office Hours**: TBD
- **Email**: course-instructor@example.com

---

*Remember: Great UX is invisible. Focus on making your AI application intuitive and delightful to use!* 
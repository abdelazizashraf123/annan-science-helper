// AnnanScience Chat Frontend JavaScript

document.addEventListener('DOMContentLoaded', () => {
  const chatMessages = document.getElementById('chat-messages');
  const messageInput = document.getElementById('message-input');
  const loadingIndicator = document.getElementById('loading');
  const settingsPanel = document.getElementById('settings-panel');
  const apiEndpointInput = document.getElementById('api-endpoint');
  const apiKeyInput = document.getElementById('api-key');
  const systemPromptInput = document.getElementById('system-prompt');

  // Load saved settings or defaults
  loadSettings();

  // Send message on Enter key (Shift+Enter for newline)
  messageInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  window.toggleSettings = function() {
    settingsPanel.classList.toggle('hidden');
  };

  window.saveSettings = function() {
    localStorage.setItem('apiEndpoint', apiEndpointInput.value.trim());
    localStorage.setItem('apiKey', apiKeyInput.value.trim());
    localStorage.setItem('systemPrompt', systemPromptInput.value.trim());
    alert('Settings saved!');
    toggleSettings();
  };

  function loadSettings() {
    const savedEndpoint = localStorage.getItem('apiEndpoint');
    const savedKey = localStorage.getItem('apiKey');
    const savedPrompt = localStorage.getItem('systemPrompt');

    if (savedEndpoint) apiEndpointInput.value = savedEndpoint;
    if (savedKey) apiKeyInput.value = savedKey;
    if (savedPrompt) systemPromptInput.value = savedPrompt;
  }

  window.attachFile = function() {
    alert('Attach feature is not implemented yet.');
  };

  window.searchWeb = function() {
    alert('Search feature is not implemented yet.');
  };

  window.voiceInput = function() {
    alert('Voice input feature is not implemented yet.');
  };

  window.sendMessage = async function() {
    const message = messageInput.value.trim();
    if (!message) return;

    appendMessage('user', message);
    messageInput.value = '';
    scrollToBottom();

    showLoading(true);

    try {
      const response = await callRagApi(message);
      if (response && response.answer) {
        appendMessage('assistant', response.answer);
      } else {
        appendMessage('assistant', "Oops! I didn't get that. Can you try again?");
      }
    } catch (error) {
      appendMessage('assistant', 'Sorry, something went wrong. Please try again later.');
      console.error('API error:', error);
    } finally {
      showLoading(false);
      scrollToBottom();
    }
  };

  function appendMessage(sender, text) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message');
    if (sender === 'user') {
      messageDiv.classList.add('user-message');
    } else {
      messageDiv.classList.add('assistant-message');
    }

    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('message-avatar');
    avatarDiv.textContent = sender === 'user' ? '👧' : '🧪';

    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.innerHTML = sanitizeText(text);

    messageDiv.appendChild(avatarDiv);
    messageDiv.appendChild(contentDiv);

    chatMessages.appendChild(messageDiv);
  }

  function sanitizeText(text) {
    // Simple sanitizer to prevent HTML injection
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, '<br>');
  }

  function showLoading(show) {
    if (show) {
      loadingIndicator.classList.remove('hidden');
    } else {
      loadingIndicator.classList.add('hidden');
    }
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  async function callRagApi(userMessage) {
    const endpoint = apiEndpointInput.value.trim() || 'http://localhost:5000/chat';
    const sessionId = getSessionId();

    const payload = {
      message: userMessage,
      session_id: sessionId
    };

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();
    return data;
  }

  function getSessionId() {
    // Generate or get existing session ID for conversation continuity
    let sessionId = localStorage.getItem('annanscience_session_id');
    if (!sessionId) {
      sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
      localStorage.setItem('annanscience_session_id', sessionId);
    }
    return sessionId;
  }

  window.resetConversation = async function() {
    try {
      const endpoint = apiEndpointInput.value.trim() || 'http://localhost:5000/reset';
      const sessionId = getSessionId();

      await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ session_id: sessionId })
      });

      // Clear chat messages
      chatMessages.innerHTML = `
        <div class="message assistant-message">
          <div class="message-avatar">🧪</div>
          <div class="message-content">
            <p>مرحباً أصدقائي الصغار! 👋 أنا AnnanScience، معلمكم الودود في العلوم!</p>
            <p>اسألوني عن أي شيء في العلوم - من لماذا السماء زرقاء 🌌 إلى كيف تنمو النباتات 🌱!</p>
            <p>ماذا تريدون أن نستكشف اليوم؟ 🔍</p>
          </div>
        </div>
      `;
      
      alert('تم إعادة تعيين المحادثة! 🎉');
    } catch (error) {
      console.error('Error resetting conversation:', error);
      alert('حدث خطأ أثناء إعادة تعيين المحادثة.');
    }
  };
});

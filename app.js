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
    const endpoint = apiEndpointInput.value.trim() || 'https://your-api-endpoint.com/chat';
    const apiKey = apiKeyInput.value.trim() || 'YOUR_API_KEY_HERE';
    const systemPrompt = systemPromptInput.value.trim() || 'You are AnnanScience, a friendly science teacher for kids aged 6-12.';

    const payload = {
      model: 'rag-model-placeholder',
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userMessage }
      ]
    };

    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }

    const data = await response.json();

    // Assuming the RAG API returns { answer: "text" }
    return data;
  }
});

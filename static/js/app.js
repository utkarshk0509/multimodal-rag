// State management
let state = {
  apiKey: localStorage.getItem('rag_google_api_key') || '',
  activeCollection: localStorage.getItem('rag_active_collection') || '',
  activeDocumentName: localStorage.getItem('rag_active_doc_name') || '',
  collections: [],
  messages: [],
  selectedFile: null,
  isIngesting: false,
  isQuerying: false,
  isRecording: false,
  recognition: null
};

// DOM references
const apiKeyInput = document.getElementById('apiKeyInput');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const filePreview = document.getElementById('filePreview');
const fileNameDisplay = document.getElementById('fileNameDisplay');
const removeFileBtn = document.getElementById('removeFileBtn');
const buildBtn = document.getElementById('buildBtn');
const collectionsList = document.getElementById('collectionsList');
const sessionBanner = document.getElementById('sessionBanner');
const sessionDocName = document.getElementById('sessionDocName');
const messagesContainer = document.getElementById('messagesContainer');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const micBtn = document.getElementById('micBtn');
const toastContainer = document.getElementById('toastContainer');

// Notification helper
function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// Initialize Speech Recognition
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    if (micBtn) micBtn.title = "Voice dictation not supported in this browser";
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = false;
  state.recognition.lang = 'en-US';

  state.recognition.onstart = () => {
    state.isRecording = true;
    micBtn.classList.add('recording');
    showToast("Listening... Speak your query clearly.", "info");
  };

  state.recognition.onresult = (event) => {
    const transcript = event.results[0][0].transcript;
    chatInput.value = transcript;
    handleSendQuery();
  };

  state.recognition.onerror = (event) => {
    console.error("Speech Recognition Error:", event.error);
    showToast(`Speech recognition error: ${event.error}`, 'error');
    stopRecording();
  };

  state.recognition.onend = () => {
    stopRecording();
  };
}

function toggleRecording() {
  if (!state.recognition) {
    showToast("Web Speech API is not supported in your browser.", "error");
    return;
  }
  if (state.isRecording) {
    state.recognition.stop();
  } else {
    try {
      state.recognition.start();
    } catch (e) {
      console.warn("Speech start issue:", e);
    }
  }
}

function stopRecording() {
  state.isRecording = false;
  micBtn.classList.remove('recording');
}

// Fetch API Key check and collections
async function initApp() {
  if (state.apiKey) {
    apiKeyInput.value = state.apiKey;
  }

  apiKeyInput.addEventListener('input', (e) => {
    state.apiKey = e.target.value.trim();
    localStorage.setItem('rag_google_api_key', state.apiKey);
  });

  setupSpeechRecognition();
  await loadCollections();
  renderSession();
}

async function loadCollections() {
  try {
    const res = await fetch('/api/collections');
    if (!res.ok) return;
    const data = await res.json();
    state.collections = data.collections || [];
    renderCollections();
  } catch (err) {
    console.warn("Could not load collections:", err);
  }
}

function renderCollections() {
  collectionsList.innerHTML = '';
  if (state.collections.length === 0) {
    collectionsList.innerHTML = '<p style="font-size:0.75rem; color: var(--text-dim); text-align:center; padding: 10px;">No index collections yet</p>';
    return;
  }

  state.collections.forEach(col => {
    const card = document.createElement('div');
    card.className = `collection-card ${col.name === state.activeCollection ? 'active' : ''}`;
    
    // Clean display name
    const displayName = col.name.replace(/^col_/, '').replace(/_/g, ' ');

    card.innerHTML = `
      <div class="collection-info">
        <h4>${displayName}</h4>
        <p>${col.name}</p>
      </div>
      <div class="badge-count">${col.count} chunks</div>
    `;

    card.addEventListener('click', () => {
      selectCollection(col.name, displayName);
    });

    collectionsList.appendChild(card);
  });
}

function selectCollection(colName, displayName) {
  state.activeCollection = colName;
  state.activeDocumentName = displayName;
  localStorage.setItem('rag_active_collection', colName);
  localStorage.setItem('rag_active_doc_name', displayName);
  renderCollections();
  renderSession();
  showToast(`Active collection set to "${displayName}"`, "info");
}

function renderSession() {
  if (state.activeCollection) {
    sessionBanner.style.display = 'flex';
    sessionDocName.textContent = state.activeDocumentName || state.activeCollection;
  } else {
    sessionBanner.style.display = 'none';
  }
}

// File dropzone management
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    handleFileSelected(files[0]);
  }
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileSelected(e.target.files[0]);
  }
});

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast("Only PDF documents are supported.", "error");
    return;
  }
  state.selectedFile = file;
  fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  dropzone.style.display = 'none';
  filePreview.style.display = 'flex';
  buildBtn.disabled = false;
}

removeFileBtn.addEventListener('click', () => {
  state.selectedFile = null;
  fileInput.value = '';
  filePreview.style.display = 'none';
  dropzone.style.display = 'block';
  buildBtn.disabled = true;
});

// Document Ingestion
buildBtn.addEventListener('click', async () => {
  if (!state.selectedFile) return;

  const formData = new FormData();
  formData.append('file', state.selectedFile);
  if (state.apiKey) {
    formData.append('api_key', state.apiKey);
  }

  state.isIngesting = true;
  buildBtn.disabled = true;
  buildBtn.innerHTML = '<div class="spinner"></div> Indexing Knowledge Base...';

  try {
    const res = await fetch('/api/ingest', {
      method: 'POST',
      body: formData
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || 'Ingestion failed');
    }

    showToast(`Indexed ${data.chunks_indexed} chunks across ${data.total_pages} pages!`, 'success');
    
    // Auto-select active collection
    state.activeCollection = data.collection_name;
    state.activeDocumentName = state.selectedFile.name;
    localStorage.setItem('rag_active_collection', state.activeCollection);
    localStorage.setItem('rag_active_doc_name', state.activeDocumentName);
    
    await loadCollections();
    renderSession();
  } catch (err) {
    showToast(err.message, 'error');
  } finally {
    state.isIngesting = false;
    buildBtn.disabled = false;
    buildBtn.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="7 10 12 15 17 10"/>
        <line x1="12" y1="15" x2="12" y2="3"/>
      </svg>
      Build Knowledge Base
    `;
  }
});

// Markdown parser helper for rendering formatted answer
function formatMarkdown(text) {
  if (!text) return '';
  let formatted = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/\*([^*]+)\*/g, '<em>$1</em>')
    .replace(/\n\n/g, '<br><br>')
    .replace(/\n/g, '<br>');
  return formatted;
}

// Append Chat Message
function appendMessage(role, content, sources = []) {
  // Clear welcome screen if present
  const welcome = document.querySelector('.welcome-screen');
  if (welcome) welcome.remove();

  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  avatar.textContent = role === 'user' ? 'U' : 'AI';

  const contentBox = document.createElement('div');
  contentBox.className = 'message-content';
  contentBox.innerHTML = role === 'user' ? content : formatMarkdown(content);

  // Render sources accordion for assistant responses
  if (role === 'assistant' && sources && sources.length > 0) {
    const sourcesBox = document.createElement('div');
    sourcesBox.className = 'sources-box';
    
    let sourcesHtml = `
      <details>
        <summary>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
            <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
          </svg>
          ${sources.length} Context Citations Retrieved
        </summary>
        <div class="sources-list">
    `;

    sources.forEach((s, idx) => {
      sourcesHtml += `
        <div class="source-item">
          <div class="source-item-header">
            <span>#${idx + 1} Source: ${s.source}</span>
            <span>Page ${s.page || 'N/A'}</span>
          </div>
          <div class="source-item-text">${s.content}</div>
        </div>
      `;
    });

    sourcesHtml += `</div></details>`;
    sourcesBox.innerHTML = sourcesHtml;
    contentBox.appendChild(sourcesBox);
  }

  row.appendChild(avatar);
  row.appendChild(contentBox);
  messagesContainer.appendChild(row);

  // Auto scroll to bottom
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Query Execution
async function handleSendQuery() {
  const query = chatInput.value.trim();
  if (!query || state.isQuerying) return;

  if (!state.activeCollection) {
    showToast("Please upload a document or select an indexed collection first.", "error");
    return;
  }

  appendMessage('user', query);
  chatInput.value = '';

  state.isQuerying = true;
  sendBtn.disabled = true;

  // Placeholder thinking bubble
  const thinkingRow = document.createElement('div');
  thinkingRow.className = 'message-row assistant';
  thinkingRow.id = 'thinkingBubble';
  thinkingRow.innerHTML = `
    <div class="avatar bot-avatar">AI</div>
    <div class="message-content" style="display:flex; align-items:center; gap:8px;">
      <div class="spinner" style="border-top-color: var(--accent);"></div>
      <span style="color:var(--text-muted); font-size:0.85rem;">Retrieving context & synthesizing multimodal response...</span>
    </div>
  `;
  messagesContainer.appendChild(thinkingRow);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        collection_name: state.activeCollection,
        api_key: state.apiKey || undefined,
        top_k: 5
      })
    });

    const data = await res.json();
    thinkingRow.remove();

    if (!res.ok) {
      throw new Error(data.detail || 'Query failed');
    }

    appendMessage('assistant', data.answer, data.sources);
  } catch (err) {
    thinkingRow.remove();
    appendMessage('assistant', `⚠️ **Error encountered:** ${err.message}`);
    showToast(err.message, 'error');
  } finally {
    state.isQuerying = false;
    sendBtn.disabled = false;
    chatInput.focus();
  }
}

// Event Listeners for Chat
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    handleSendQuery();
  }
});

sendBtn.addEventListener('click', handleSendQuery);
micBtn.addEventListener('click', toggleRecording);

// Start app
window.addEventListener('DOMContentLoaded', initApp);

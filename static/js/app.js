// ==========================================================================
// AeroRAG frontend
// ==========================================================================

// ---------- Safe storage (private windows can throw) ----------
const store = {
  get(key, fallback = null) {
    try { const v = localStorage.getItem(key); return v === null ? fallback : v; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* ignore quota / privacy errors */ }
  },
  remove(key) {
    try { localStorage.removeItem(key); } catch { /* ignore */ }
  },
};

// ---------- State ----------
const state = {
  apiKey: store.get('rag_google_api_key', ''),
  activeCollection: store.get('rag_active_collection', ''),
  activeDocumentName: store.get('rag_active_doc_name', ''),
  topK: parseInt(store.get('rag_top_k', '5'), 10) || 5,
  collections: [],
  messages: [],
  selectedFile: null,
  isIngesting: false,
  isQuerying: false,
  isRecording: false,
  recognition: null,
  abortController: null,
};

const MAX_STORED_MESSAGES = 60;
const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// ---------- DOM references ----------
const $ = (id) => document.getElementById(id);
const apiKeyInput = $('apiKeyInput');
const dropzone = $('dropzone');
const fileInput = $('fileInput');
const filePreview = $('filePreview');
const fileNameDisplay = $('fileNameDisplay');
const removeFileBtn = $('removeFileBtn');
const buildBtn = $('buildBtn');
const collectionsList = $('collectionsList');
const sessionBanner = $('sessionBanner');
const sessionDocName = $('sessionDocName');
const chatStatus = $('chatStatus');
const chatStatusText = $('chatStatusText');
const messagesContainer = $('messagesContainer');
const welcomeTemplate = $('welcomeTemplate');
const scrollBottomBtn = $('scrollBottomBtn');
const chatInput = $('chatInput');
const chatInputBox = document.querySelector('.chat-input-box');
const sendBtn = $('sendBtn');
const micBtn = $('micBtn');
const topKSelect = $('topKSelect');
const clearChatBtn = $('clearChatBtn');
const exportChatBtn = $('exportChatBtn');
const toastContainer = $('toastContainer');

// ---------- Motion bridge ----------
// Calls into static/js/motion.js when loaded; no-op if Motion is unavailable
// or the visitor prefers reduced motion.
function fx(name, ...args) {
  const m = window.AeroMotion;
  return m && typeof m[name] === 'function' ? m[name](...args) : undefined;
}

// ---------- Small helpers ----------
function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function uid() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
}

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for non-secure contexts (plain http on a LAN IP)
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    let ok = false;
    try { ok = document.execCommand('copy'); } catch { ok = false; }
    ta.remove();
    return ok;
  }
}

const ICONS = {
  copy: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>',
  check: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  retry: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg>',
  edit: '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4 12.5-12.5z"/></svg>',
  chevron: '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>',
};

// ---------- Toasts ----------
function showToast(message, type = 'info', action = null) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.setAttribute('role', type === 'error' ? 'alert' : 'status');

  const text = document.createElement('span');
  text.textContent = message;
  toast.appendChild(text);

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    Promise.resolve(fx('toastOut', toast)).finally(() => toast.remove());
  };

  if (action) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'toast-action';
    btn.textContent = action.label;
    btn.addEventListener('click', () => { action.onClick(); close(); });
    toast.appendChild(btn);
  }

  toastContainer.appendChild(toast);
  fx('toastIn', toast);
  setTimeout(close, action ? 6000 : 4000);
}

// ==========================================================================
// Markdown rendering (escape-first: model output can never inject HTML)
// ==========================================================================
const CITE_RE = /\b(pages?|pp?\.)\s*(\d{1,4})(?:\s*(?:-|–|to)\s*(\d{1,4}))?/gi;

function inline(text) {
  return String(text)
    .split(/(`[^`\n]+`)/g)
    .map((part) => {
      if (/^`[^`\n]+`$/.test(part)) return `<code>${esc(part.slice(1, -1))}</code>`;
      let h = esc(part);
      h = h.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
        '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
      h = h.replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>');
      h = h.replace(/(^|[^*\w])\*([^*\n]+?)\*(?!\*)/g, '$1<em>$2</em>');
      // Page references become clickable citations (text segments only, not inside tags)
      h = h.split(/(<[^>]+>)/g).map((seg) => (seg.startsWith('<') ? seg :
        seg.replace(CITE_RE, (m, _w, p) => `<button type="button" class="cite-ref" data-page="${p}">${m}</button>`)
      )).join('');
      return h;
    })
    .join('');
}

function renderMarkdown(src) {
  const lines = String(src || '').replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  let i = 0;

  const LIST_RE = /^(\s*)([-*+•]|\d+[.)])\s+(.*)$/;
  const HR_RE = /^\s*(-{3,}|\*{3,}|_{3,})\s*$/;
  const TABLE_SEP_RE = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;
  const isTableAt = (k) => lines[k]?.includes('|') && TABLE_SEP_RE.test(lines[k + 1] || '');
  const isBlockStart = (k) => {
    const l = lines[k];
    return /^\s*```/.test(l) || /^#{1,6}\s/.test(l) || HR_RE.test(l) || LIST_RE.test(l) || /^>\s?/.test(l) || isTableAt(k);
  };
  const splitRow = (l) => l.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());

  while (i < lines.length) {
    const line = lines[i];

    if (/^\s*$/.test(line)) { i++; continue; }

    // Fenced code
    if (/^\s*```/.test(line)) {
      const lang = line.trim().slice(3).trim();
      const code = [];
      i++;
      while (i < lines.length && !/^\s*```/.test(lines[i])) code.push(lines[i++]);
      i++; // closing fence (or EOF while typing)
      out.push(
        `<div class="code-block"><div class="code-block-header"><span>${esc(lang || 'code')}</span>` +
        `<button type="button" class="code-copy">Copy</button></div><pre><code>${esc(code.join('\n'))}</code></pre></div>`
      );
      continue;
    }

    // Headings
    const hm = line.match(/^(#{1,6})\s+(.*)$/);
    if (hm) {
      const tag = hm[1].length === 1 ? 'h3' : hm[1].length === 2 ? 'h4' : 'h5';
      out.push(`<${tag}>${inline(hm[2].replace(/#+\s*$/, ''))}</${tag}>`);
      i++;
      continue;
    }

    if (HR_RE.test(line)) { out.push('<hr>'); i++; continue; }

    // Tables
    if (isTableAt(i)) {
      const head = splitRow(lines[i]);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].includes('|') && !/^\s*$/.test(lines[i])) rows.push(splitRow(lines[i++]));
      out.push(
        '<div class="table-wrap"><table><thead><tr>' +
        head.map((c) => `<th>${inline(c)}</th>`).join('') +
        '</tr></thead><tbody>' +
        rows.map((r) => '<tr>' + head.map((_, ci) => `<td>${inline(r[ci] ?? '')}</td>`).join('') + '</tr>').join('') +
        '</tbody></table></div>'
      );
      continue;
    }

    // Blockquote
    if (/^>\s?/.test(line)) {
      const quote = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) quote.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<blockquote>${quote.map(inline).join('<br>')}</blockquote>`);
      continue;
    }

    // Lists (nested by indentation)
    if (LIST_RE.test(line)) {
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(LIST_RE);
        if (m) {
          items.push({ indent: m[1].replace(/\t/g, '    ').length, ordered: /\d/.test(m[2]), start: parseInt(m[2], 10), text: m[3] });
          i++;
        } else if (items.length && /^\s{2,}\S/.test(lines[i])) {
          items[items.length - 1].text += ' ' + lines[i].trim();
          i++;
        } else if (/^\s*$/.test(lines[i]) && LIST_RE.test(lines[i + 1] || '')) {
          i++;
        } else break;
      }
      let html = '';
      const stack = [];
      items.forEach((it) => {
        while (stack.length && it.indent < stack[stack.length - 1].indent) html += `</li></${stack.pop().tag}>`;
        const top = stack[stack.length - 1];
        if (!top || it.indent > top.indent) {
          const tag = it.ordered ? 'ol' : 'ul';
          html += `<${tag}${it.ordered && it.start > 1 ? ` start="${it.start}"` : ''}>`;
          stack.push({ indent: it.indent, tag });
        } else {
          html += '</li>';
        }
        html += `<li>${inline(it.text)}`;
      });
      while (stack.length) html += `</li></${stack.pop().tag}>`;
      out.push(html);
      continue;
    }

    // Paragraph
    const para = [];
    while (i < lines.length && !/^\s*$/.test(lines[i]) && !(para.length && isBlockStart(i))) para.push(lines[i++]);
    out.push(`<p>${para.map(inline).join('<br>')}</p>`);
  }

  return out.join('');
}

// ==========================================================================
// Speech recognition
// ==========================================================================
function setupSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.title = 'Voice dictation is not supported in this browser';
    return;
  }

  state.recognition = new SpeechRecognition();
  state.recognition.continuous = false;
  state.recognition.interimResults = true;
  state.recognition.lang = 'en-US';

  let baseText = '';
  state.recognition.onstart = () => {
    state.isRecording = true;
    baseText = chatInput.value ? chatInput.value.trimEnd() + ' ' : '';
    micBtn.classList.add('recording');
    micBtn.setAttribute('aria-pressed', 'true');
    chatInput.placeholder = 'Listening… speak your question';
  };

  // Live transcript into the box; the user reviews and presses Enter to send
  state.recognition.onresult = (event) => {
    const transcript = Array.from(event.results).map((r) => r[0].transcript).join('');
    chatInput.value = baseText + transcript;
    autoGrow();
    updateComposer();
  };

  state.recognition.onerror = (event) => {
    if (event.error !== 'aborted' && event.error !== 'no-speech') {
      showToast(`Speech recognition error: ${event.error}`, 'error');
    }
    stopRecording();
  };

  state.recognition.onend = () => {
    stopRecording();
    chatInput.focus();
  };
}

function toggleRecording() {
  if (!state.recognition) {
    showToast('Voice dictation is not supported in your browser.', 'error');
    return;
  }
  if (state.isRecording) {
    state.recognition.stop();
  } else {
    try { state.recognition.start(); } catch (e) { console.warn('Speech start issue:', e); }
  }
}

function stopRecording() {
  state.isRecording = false;
  micBtn.classList.remove('recording');
  micBtn.setAttribute('aria-pressed', 'false');
  updateComposer();
}

// ==========================================================================
// Collections & session
// ==========================================================================
async function loadCollections() {
  try {
    const res = await fetch('/api/collections');
    if (!res.ok) return;
    const data = await res.json();
    state.collections = data.collections || [];
    renderCollections({ animate: true });
  } catch (err) {
    console.warn('Could not load collections:', err);
  }
}

function renderCollections({ animate = false } = {}) {
  collectionsList.innerHTML = '';
  if (state.collections.length === 0) {
    collectionsList.innerHTML = '<p style="font-size:0.75rem; color: var(--text-muted); text-align:center; padding: 10px;">No index collections yet</p>';
    return;
  }

  state.collections.forEach((col) => {
    const card = document.createElement('div');
    card.className = `collection-card ${col.name === state.activeCollection ? 'active' : ''}`;
    card.setAttribute('data-press', '0.97');
    card.setAttribute('role', 'button');
    card.tabIndex = 0;

    const displayName = col.name.replace(/^col_/, '').replace(/_/g, ' ');
    card.innerHTML = `
      <div class="collection-info">
        <h4>${esc(displayName)}</h4>
        <p>${esc(col.name)}</p>
      </div>
      <div class="badge-count">${esc(col.count)} chunks</div>
    `;

    const choose = () => selectCollection(col.name, displayName);
    card.addEventListener('click', choose);
    card.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); choose(); }
    });
    collectionsList.appendChild(card);
  });

  if (animate) fx('stagger', collectionsList.children, 'slide-left');
}

function selectCollection(colName, displayName) {
  if (colName === state.activeCollection) return;
  if (state.isQuerying) state.abortController?.abort();
  state.activeCollection = colName;
  state.activeDocumentName = displayName;
  store.set('rag_active_collection', colName);
  store.set('rag_active_doc_name', displayName);
  renderCollections();
  renderSession();
  loadHistory();
  renderChat({ animate: true });
  showToast(`Now chatting with "${displayName}"`, 'info');
}

function renderSession() {
  if (state.activeCollection) {
    const wasHidden = sessionBanner.hidden;
    sessionBanner.hidden = false;
    sessionDocName.textContent = state.activeDocumentName || state.activeCollection;
    sessionBanner.title = state.activeDocumentName || state.activeCollection;
    if (wasHidden) fx('enter', sessionBanner, 'fade-down');
    else fx('pop', sessionBanner);
  } else {
    sessionBanner.hidden = true;
  }
  updateChatChrome();
}

// ==========================================================================
// Document upload & ingestion
// ==========================================================================
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});

['dragleave', 'drop'].forEach((eventName) => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});

dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if (files && files.length > 0) handleFileSelected(files[0]);
});

fileInput.addEventListener('change', (e) => {
  if (e.target.files && e.target.files.length > 0) handleFileSelected(e.target.files[0]);
});

function handleFileSelected(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Only PDF documents are supported.', 'error');
    fx('shake', dropzone);
    return;
  }
  state.selectedFile = file;
  fileNameDisplay.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
  dropzone.style.display = 'none';
  filePreview.style.display = 'flex';
  buildBtn.disabled = false;
  fx('enter', filePreview, 'scale');
  fx('pop', buildBtn);
}

removeFileBtn.addEventListener('click', async () => {
  state.selectedFile = null;
  fileInput.value = '';
  buildBtn.disabled = true;
  await fx('exit', filePreview, 'scale');
  filePreview.style.display = 'none';
  dropzone.style.display = 'block';
  fx('enter', dropzone, 'scale');
});

const BUILD_BTN_HTML = buildBtn.innerHTML;

buildBtn.addEventListener('click', async () => {
  if (!state.selectedFile) return;

  const formData = new FormData();
  formData.append('file', state.selectedFile);
  if (state.apiKey) formData.append('api_key', state.apiKey);

  state.isIngesting = true;
  buildBtn.disabled = true;
  buildBtn.innerHTML = '<div class="spinner"></div> Indexing Knowledge Base...';
  updateChatChrome();

  try {
    const res = await fetch('/api/ingest', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Ingestion failed');

    showToast(`Indexed ${data.chunks_indexed} chunks across ${data.total_pages} pages!`, 'success');
    fx('pop', buildBtn);

    state.activeCollection = data.collection_name;
    state.activeDocumentName = state.selectedFile.name;
    store.set('rag_active_collection', state.activeCollection);
    store.set('rag_active_doc_name', state.activeDocumentName);

    await loadCollections();
    renderSession();
    loadHistory();
    renderChat({ animate: true });
    chatInput.focus();
  } catch (err) {
    showToast(err.message, 'error');
    fx('shake', buildBtn);
  } finally {
    state.isIngesting = false;
    buildBtn.disabled = !state.selectedFile;
    buildBtn.innerHTML = BUILD_BTN_HTML;
    updateChatChrome();
  }
});

// ==========================================================================
// Chat: history persistence
// ==========================================================================
const historyKey = () => `rag_chat_${state.activeCollection}`;

function loadHistory() {
  state.messages = [];
  if (!state.activeCollection) return;
  try {
    const parsed = JSON.parse(store.get(historyKey(), '[]'));
    if (Array.isArray(parsed)) state.messages = parsed.filter((m) => m && m.role && typeof m.content === 'string');
  } catch { state.messages = []; }
}

function saveHistory() {
  if (!state.activeCollection) return;
  if (!state.messages.length) { store.remove(historyKey()); return; }
  store.set(historyKey(), JSON.stringify(state.messages.slice(-MAX_STORED_MESSAGES)));
}

// ==========================================================================
// Chat: rendering
// ==========================================================================
function isNearBottom(threshold = 120) {
  const el = messagesContainer;
  return el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
}

function scrollToBottom({ smooth = true } = {}) {
  requestAnimationFrame(() => {
    messagesContainer.scrollTo({ top: messagesContainer.scrollHeight, behavior: smooth && !prefersReducedMotion ? 'smooth' : 'auto' });
  });
}

function updateScrollButton() {
  scrollBottomBtn.hidden = isNearBottom(160) || !state.messages.length;
}

function updateChatChrome() {
  const hasDoc = !!state.activeCollection;
  let stateName = 'idle';
  let text = 'No document selected';
  if (state.isIngesting) { stateName = 'busy'; text = 'Indexing document…'; }
  else if (state.isQuerying) { stateName = 'busy'; text = 'Searching your document…'; }
  else if (hasDoc) { stateName = 'ready'; text = `Ready · ${state.topK} sources per answer`; }
  chatStatus.dataset.state = stateName;
  chatStatusText.textContent = text;

  clearChatBtn.disabled = !state.messages.length || state.isQuerying;
  exportChatBtn.disabled = !state.messages.length;
  chatInput.placeholder = state.isRecording
    ? 'Listening… speak your question'
    : hasDoc
      ? `Ask about ${state.activeDocumentName || 'your document'}…`
      : 'Upload or select a document to start asking questions…';
  updateComposer();
}

function updateComposer() {
  const hasText = chatInput.value.trim().length > 0;
  sendBtn.classList.toggle('is-stop', state.isQuerying);
  sendBtn.disabled = !state.isQuerying && !hasText;
  sendBtn.title = state.isQuerying ? 'Stop generating' : 'Send question';
  sendBtn.setAttribute('aria-label', sendBtn.title);
}

function autoGrow() {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 168) + 'px';
}

function showWelcome({ animate = false } = {}) {
  messagesContainer.innerHTML = '';
  const node = welcomeTemplate.content.firstElementChild.cloneNode(true);
  const subtitle = node.querySelector('[data-welcome-subtitle]');
  if (state.activeCollection) {
    subtitle.innerHTML = `Chatting with <strong>${esc(state.activeDocumentName || state.activeCollection)}</strong>. ` +
      'Every answer links back to the exact pages it was drawn from.';
  } else {
    subtitle.innerHTML = 'Upload a technical PDF on the left and click <strong>Build Knowledge Base</strong> ' +
      '— or pick an existing collection — then try one of these:';
  }
  messagesContainer.appendChild(node);
  if (animate) {
    fx('enter', node, 'fade-up');
    fx('stagger', node.querySelectorAll('.suggestion-card'), 'fade-up', { delay: 0.1 });
  }
}

function renderChat({ animate = false } = {}) {
  if (!state.messages.length) {
    showWelcome({ animate });
  } else {
    messagesContainer.innerHTML = '';
    const divider = document.createElement('div');
    divider.className = 'chat-divider';
    divider.textContent = `Earlier conversation · ${state.activeDocumentName || 'document'}`;
    messagesContainer.appendChild(divider);
    state.messages.forEach((msg) => messagesContainer.appendChild(buildMessageRow(msg)));
    markLatest();
    scrollToBottom({ smooth: false });
  }
  updateChatChrome();
  updateScrollButton();
}

function markLatest() {
  messagesContainer.querySelectorAll('.message-row.is-latest').forEach((r) => r.classList.remove('is-latest'));
  const rows = messagesContainer.querySelectorAll('.message-row.assistant:not(.thinking)');
  const last = rows[rows.length - 1];
  const lastMsg = state.messages[state.messages.length - 1];
  if (last && lastMsg && lastMsg.role === 'assistant' && last.dataset.id === lastMsg.id) last.classList.add('is-latest');

  // Regenerate is only offered on the newest answer
  messagesContainer.querySelectorAll('[data-action="regenerate"]').forEach((b) => {
    b.hidden = !b.closest('.message-row').classList.contains('is-latest');
  });
}

function buildSources(sources) {
  const panel = document.createElement('div');
  panel.className = 'sources-panel';

  const pages = [...new Set(sources.map((s) => s.page).filter((p) => p !== null && p !== undefined))]
    .sort((a, b) => a - b);

  const chips = pages.map((p) => `<button type="button" class="cite-chip" data-page="${esc(p)}">p.${esc(p)}</button>`).join('');

  panel.innerHTML = `
    <div class="sources-row">
      <span class="sources-label">Sources</span>
      ${chips}
      <button type="button" class="sources-toggle" aria-expanded="false">
        <span>${sources.length} passage${sources.length === 1 ? '' : 's'}</span>${ICONS.chevron}
      </button>
    </div>
    <div class="sources-list">
      ${sources.map((s, idx) => `
        <div class="source-item" data-page="${esc(s.page ?? '')}">
          <div class="source-item-header">
            <span class="source-index">${idx + 1}</span>
            <span class="source-name" title="${esc(s.source)}">${esc(s.source)}</span>
            <span class="source-page">Page ${esc(s.page ?? 'N/A')}</span>
          </div>
          <div class="source-item-text">${esc(s.content)}</div>
          <button type="button" class="source-expand">Show more</button>
        </div>`).join('')}
    </div>
  `;
  return panel;
}

function buildMessageRow(msg) {
  const row = document.createElement('div');
  row.className = `message-row ${msg.role}${msg.error ? ' error' : ''}`;
  row.dataset.id = msg.id;

  const avatar = document.createElement('div');
  avatar.className = `avatar ${msg.role === 'user' ? 'user-avatar' : 'bot-avatar'}`;
  avatar.textContent = msg.role === 'user' ? 'You' : 'AI';
  avatar.setAttribute('aria-hidden', 'true');
  if (msg.role === 'user') avatar.style.fontSize = '0.6rem';

  const body = document.createElement('div');
  body.className = 'message-body';

  const meta = document.createElement('div');
  meta.className = 'message-meta';
  if (msg.role === 'user') {
    meta.innerHTML = `<span>${formatTime(msg.ts)}</span>`;
  } else {
    const details = [formatTime(msg.ts)];
    if (msg.ms) details.push(`${(msg.ms / 1000).toFixed(1)}s`);
    meta.innerHTML = `<strong>AeroRAG</strong><span>${details.map(esc).join(' · ')}</span>`;
  }

  const content = document.createElement('div');
  content.className = 'message-content';
  if (msg.role === 'user') {
    content.textContent = msg.content;
  } else {
    const rich = document.createElement('div');
    rich.className = 'rich';
    rich.innerHTML = renderMarkdown(msg.content);
    content.appendChild(rich);
    if (msg.sources && msg.sources.length) content.appendChild(buildSources(msg.sources));
  }

  const actions = document.createElement('div');
  actions.className = 'message-actions';
  if (msg.role === 'user') {
    actions.innerHTML = `
      <button type="button" class="msg-action" data-action="edit" title="Edit and resend">${ICONS.edit}<span>Edit</span></button>
      <button type="button" class="msg-action" data-action="copy" title="Copy question">${ICONS.copy}<span>Copy</span></button>`;
  } else if (msg.error) {
    actions.innerHTML = `
      <button type="button" class="msg-action" data-action="regenerate" title="Try again">${ICONS.retry}<span>Retry</span></button>`;
  } else {
    actions.innerHTML = `
      <button type="button" class="msg-action" data-action="copy" title="Copy answer">${ICONS.copy}<span>Copy</span></button>
      <button type="button" class="msg-action" data-action="regenerate" title="Generate a new answer">${ICONS.retry}<span>Regenerate</span></button>`;
  }

  body.append(meta, content, actions);
  row.append(avatar, body);
  return row;
}

function appendMessage(msg, { typewriter = false } = {}) {
  const welcome = messagesContainer.querySelector('.welcome-screen');
  if (welcome) welcome.remove();

  state.messages.push(msg);
  saveHistory();

  const row = buildMessageRow(msg);
  const stick = isNearBottom(200);
  messagesContainer.appendChild(row);
  fx('message', row, msg.role);
  markLatest();
  updateChatChrome();

  if (typewriter && !prefersReducedMotion && msg.role === 'assistant' && !msg.error) {
    typewrite(row, msg.content, stick);
  } else if (stick || msg.role === 'user') {
    scrollToBottom();
  }
  return row;
}

/** Progressive reveal of an answer that has already arrived. */
function typewrite(row, text, stick) {
  const rich = row.querySelector('.rich');
  const sources = row.querySelector('.sources-panel');
  const actions = row.querySelector('.message-actions');
  if (sources) sources.style.display = 'none';
  actions.style.visibility = 'hidden';
  rich.classList.add('is-typing');

  const duration = Math.min(Math.max(text.length * 4, 400), 1800);
  const start = performance.now();

  const finish = () => {
    rich.innerHTML = renderMarkdown(text);
    rich.classList.remove('is-typing');
    actions.style.visibility = '';
    if (sources) {
      sources.style.display = '';
      fx('enter', sources, 'fade-up');
    }
    if (stick) scrollToBottom();
  };

  const step = (now) => {
    if (!row.isConnected) return;
    const t = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - t, 2);
    if (t >= 1) { finish(); return; }
    rich.innerHTML = renderMarkdown(text.slice(0, Math.floor(text.length * eased)));
    if (stick) messagesContainer.scrollTop = messagesContainer.scrollHeight;
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function buildThinkingRow() {
  const row = document.createElement('div');
  row.className = 'message-row assistant thinking';
  row.innerHTML = `
    <div class="avatar bot-avatar" aria-hidden="true">AI</div>
    <div class="message-body">
      <div class="message-meta"><strong>AeroRAG</strong><span>thinking…</span></div>
      <div class="message-content">
        <div class="thinking-card">
          <div class="thinking-head">
            <span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span>
            <span data-stage-label>Searching the document…</span>
          </div>
          <div class="thinking-steps">
            <span class="thinking-step is-active">Retrieve</span>
            <span class="thinking-step">Read context</span>
            <span class="thinking-step">Compose answer</span>
          </div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line"></div>
          <div class="skeleton-line"></div>
        </div>
      </div>
    </div>`;
  return row;
}

function startThinking() {
  const row = buildThinkingRow();
  messagesContainer.appendChild(row);
  fx('message', row, 'assistant');
  const stopDots = fx('thinking', row) || (() => {});
  scrollToBottom();

  const steps = row.querySelectorAll('.thinking-step');
  const label = row.querySelector('[data-stage-label]');
  const stages = ['Searching the document…', 'Reading the most relevant passages…', 'Composing a cited answer…'];
  const timers = [1200, 3200].map((delay, idx) => setTimeout(() => {
    steps[idx].classList.remove('is-active');
    steps[idx].classList.add('is-done');
    steps[idx + 1].classList.add('is-active');
    label.textContent = stages[idx + 1];
  }, delay));

  return () => {
    timers.forEach(clearTimeout);
    stopDots();
    row.remove();
  };
}

// ==========================================================================
// Chat: querying
// ==========================================================================
async function sendQuestion(question, { appendUser = true } = {}) {
  question = (question || '').trim();
  if (!question || state.isQuerying) return;

  if (!state.activeCollection) {
    showToast('Upload a document or select an indexed collection first.', 'error');
    fx('shake', chatInputBox);
    return;
  }

  if (appendUser) {
    appendMessage({ id: uid(), role: 'user', content: question, ts: Date.now() });
    chatInput.value = '';
    autoGrow();
  }

  state.isQuerying = true;
  state.abortController = new AbortController();
  updateChatChrome();

  const stopThinking = startThinking();
  const started = performance.now();

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.abortController.signal,
      body: JSON.stringify({
        question,
        collection_name: state.activeCollection,
        api_key: state.apiKey || undefined,
        top_k: state.topK,
      }),
    });

    let data = {};
    try { data = await res.json(); } catch { /* non-JSON error page */ }
    if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);

    stopThinking();
    appendMessage({
      id: uid(),
      role: 'assistant',
      content: data.answer || '',
      sources: data.sources || [],
      question,
      ts: Date.now(),
      ms: Math.round(performance.now() - started),
    }, { typewriter: true });
  } catch (err) {
    stopThinking();
    if (err.name === 'AbortError') {
      showToast('Stopped. Edit your question or ask another.', 'info');
    } else {
      appendMessage({
        id: uid(),
        role: 'assistant',
        error: true,
        content: `**Couldn't get an answer.** ${err.message}`,
        question,
        ts: Date.now(),
      });
    }
  } finally {
    state.isQuerying = false;
    state.abortController = null;
    updateChatChrome();
    markLatest();
    if (window.matchMedia('(hover: hover)').matches) chatInput.focus();
  }
}

function regenerate(msgId) {
  const idx = state.messages.findIndex((m) => m.id === msgId);
  if (idx === -1 || state.isQuerying) return;
  const msg = state.messages[idx];
  const question = msg.question || [...state.messages.slice(0, idx)].reverse().find((m) => m.role === 'user')?.content;
  if (!question) return;

  state.messages.splice(idx, 1);
  saveHistory();
  const row = messagesContainer.querySelector(`.message-row[data-id="${msgId}"]`);
  Promise.resolve(fx('exit', row, 'fade')).finally(() => {
    row?.remove();
    sendQuestion(question, { appendUser: false });
  });
}

function openSources(row, page) {
  const panel = row?.querySelector('.sources-panel');
  if (!panel) return;
  const wasOpen = panel.classList.contains('is-open');
  panel.classList.add('is-open');
  panel.querySelector('.sources-toggle').setAttribute('aria-expanded', 'true');
  if (!wasOpen) fx('stagger', panel.querySelectorAll('.source-item'), 'fade-up');

  if (page !== undefined) {
    const target = panel.querySelector(`.source-item[data-page="${CSS.escape(String(page))}"]`);
    if (target) {
      panel.querySelectorAll('.is-highlighted').forEach((el) => el.classList.remove('is-highlighted'));
      target.classList.add('is-highlighted');
      target.scrollIntoView({ block: 'nearest', behavior: prefersReducedMotion ? 'auto' : 'smooth' });
      fx('pop', target);
      setTimeout(() => target.classList.remove('is-highlighted'), 2200);
    } else {
      showToast(`Page ${page} isn't among the retrieved passages.`, 'info');
    }
  }
}

function clearChat() {
  if (!state.messages.length || state.isQuerying) return;
  const backup = state.messages.slice();
  state.messages = [];
  saveHistory();
  renderChat({ animate: true });
  showToast('Conversation cleared', 'info', {
    label: 'Undo',
    onClick: () => {
      state.messages = backup;
      saveHistory();
      renderChat();
    },
  });
}

function exportChat() {
  if (!state.messages.length) return;
  const doc = state.activeDocumentName || state.activeCollection || 'document';
  const lines = [`# AeroRAG conversation — ${doc}`, '', `_Exported ${new Date().toLocaleString()}_`, ''];
  state.messages.forEach((m) => {
    lines.push(m.role === 'user' ? `## 🧑 ${m.content}` : m.content, '');
    if (m.sources && m.sources.length) {
      const pages = [...new Set(m.sources.map((s) => s.page).filter(Boolean))].sort((a, b) => a - b);
      lines.push(`> Sources: ${m.sources[0].source}${pages.length ? ` — pages ${pages.join(', ')}` : ''}`, '');
    }
    if (m.role === 'assistant') lines.push('---', '');
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `aerorag-${doc.replace(/[^\w.-]+/g, '_').replace(/\.pdf$/i, '')}-${new Date().toISOString().slice(0, 10)}.md`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  showToast('Conversation exported as Markdown', 'success');
}

async function flashCopied(btn, text) {
  const ok = await copyText(text);
  if (!ok) { showToast('Copy failed — select the text manually.', 'error'); return; }
  const original = btn.innerHTML;
  btn.classList.add('is-done');
  btn.innerHTML = btn.classList.contains('code-copy') ? 'Copied' : `${ICONS.check}<span>Copied</span>`;
  setTimeout(() => { btn.classList.remove('is-done'); btn.innerHTML = original; }, 1600);
}

// ---------- Chat event wiring ----------
messagesContainer.addEventListener('click', (e) => {
  const target = e.target;
  const row = target.closest('.message-row');

  const suggestion = target.closest('.suggestion-card');
  if (suggestion) {
    if (!state.activeCollection) {
      chatInput.value = suggestion.dataset.prompt;
      autoGrow();
      updateComposer();
      showToast('Upload or select a document first — your question is ready to send.', 'info');
      fx('shake', chatInputBox);
      return;
    }
    sendQuestion(suggestion.dataset.prompt);
    return;
  }

  const action = target.closest('[data-action]');
  if (action && row) {
    const msg = state.messages.find((m) => m.id === row.dataset.id);
    if (!msg) return;
    if (action.dataset.action === 'copy') flashCopied(action, msg.content);
    if (action.dataset.action === 'regenerate') regenerate(msg.id);
    if (action.dataset.action === 'edit') {
      chatInput.value = msg.content;
      autoGrow();
      updateComposer();
      chatInput.focus();
      chatInput.setSelectionRange(chatInput.value.length, chatInput.value.length);
    }
    return;
  }

  const codeCopy = target.closest('.code-copy');
  if (codeCopy) {
    flashCopied(codeCopy, codeCopy.closest('.code-block').querySelector('pre').textContent);
    return;
  }

  const cite = target.closest('.cite-chip, .cite-ref');
  if (cite) { openSources(row, cite.dataset.page); return; }

  const toggle = target.closest('.sources-toggle');
  if (toggle) {
    const panel = toggle.closest('.sources-panel');
    if (panel.classList.contains('is-open')) {
      panel.classList.remove('is-open');
      toggle.setAttribute('aria-expanded', 'false');
    } else {
      openSources(row);
    }
    return;
  }

  const expand = target.closest('.source-expand');
  if (expand) {
    const item = expand.closest('.source-item');
    const open = item.classList.toggle('is-expanded');
    expand.textContent = open ? 'Show less' : 'Show more';
  }
});

messagesContainer.addEventListener('scroll', updateScrollButton, { passive: true });
scrollBottomBtn.addEventListener('click', () => scrollToBottom());

chatInput.addEventListener('input', () => {
  autoGrow();
  updateComposer();
});

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    sendQuestion(chatInput.value);
  } else if (e.key === 'ArrowUp' && !chatInput.value) {
    // Recall the last question
    const last = [...state.messages].reverse().find((m) => m.role === 'user');
    if (last) {
      e.preventDefault();
      chatInput.value = last.content;
      autoGrow();
      updateComposer();
    }
  } else if (e.key === 'Escape') {
    chatInput.blur();
  }
});

// Press "/" anywhere to jump into the composer
document.addEventListener('keydown', (e) => {
  if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey) return;
  const el = document.activeElement;
  if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable)) return;
  e.preventDefault();
  chatInput.focus();
});

sendBtn.addEventListener('click', () => {
  if (state.isQuerying) state.abortController?.abort();
  else sendQuestion(chatInput.value);
});

micBtn.addEventListener('click', toggleRecording);
clearChatBtn.addEventListener('click', clearChat);
exportChatBtn.addEventListener('click', exportChat);

topKSelect.addEventListener('change', () => {
  state.topK = parseInt(topKSelect.value, 10) || 5;
  store.set('rag_top_k', String(state.topK));
  updateChatChrome();
});

// ==========================================================================
// Boot
// ==========================================================================
async function initApp() {
  if (state.apiKey) apiKeyInput.value = state.apiKey;
  apiKeyInput.addEventListener('input', (e) => {
    state.apiKey = e.target.value.trim();
    store.set('rag_google_api_key', state.apiKey);
  });

  if (![...topKSelect.options].some((o) => o.value === String(state.topK))) state.topK = 5;
  topKSelect.value = String(state.topK);

  setupSpeechRecognition();
  loadHistory();
  renderSession();
  renderChat();
  await loadCollections();
}

window.addEventListener('DOMContentLoaded', initApp);

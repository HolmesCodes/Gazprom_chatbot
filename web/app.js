const state = {
  config: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
  chatHistory: JSON.parse(localStorage.getItem("chatHistory") || "[]"),
  historyIndex: -1,
  sourcesMode: "sources",
  currentSources: [],
  token: localStorage.getItem("token") || "",
  user: JSON.parse(localStorage.getItem("user") || "null"),
};

const CATEGORY_LABELS = {
  tech_cards: "Технические карты",
  instructions: "Инструкции",
  regulations: "Регламенты",
  sop: "Справочники (SOP)",
  normative: "Нормативные документы",
  safety: "Охрана труда",
  onboarding: "Адаптация",
  other: "Прочее",
};

// ── Auth helpers ──

function authHeaders() {
  return state.token ? { "auth": state.token } : {};
}

async function api(path, options = {}) {
  const headers = { "Content-Type": "application/json", ...authHeaders(), ...(options.headers || {}) };
  const response = await fetch(path, { headers, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || response.statusText || "API error");
  }
  return data;
}

function showScreen(id) {
  document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
  const el = document.getElementById(id);
  if (el) el.classList.remove("hidden");
}

async function checkAuth() {
  if (!state.token) {
    showScreen("screen-login");
    return false;
  }
  try {
    const data = await api("/api/auth/me");
    state.user = data.user;
    localStorage.setItem("user", JSON.stringify(data.user));
    showScreen("screen-app");
    return true;
  } catch {
    state.token = "";
    state.user = null;
    localStorage.removeItem("token");
    localStorage.removeItem("user");
    showScreen("screen-login");
    return false;
  }
}

async function doLogin(username, password) {
  try {
    const data = await api("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    state.token = data.token;
    state.user = data.user;
    localStorage.setItem("token", data.token);
    localStorage.setItem("user", JSON.stringify(data.user));
    showScreen("screen-app");
    loadConfig();
    return true;
  } catch (err) {
    document.getElementById("login-error").textContent = err.message;
    return false;
  }
}

async function doLogout() {
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch {}
  state.token = "";
  state.user = null;
  localStorage.removeItem("token");
  localStorage.removeItem("user");
  showScreen("screen-login");
}

async function doRegister(username, password) {
  try {
    await api("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ username, password }),
    });
    document.getElementById("register-error").textContent = "Аккаунт создан! Войдите.";
    document.getElementById("register-form").classList.add("hidden");
  } catch (err) {
    document.getElementById("register-error").textContent = err.message;
  }
}

// ── Chat ──

function toggleWelcome() {
  const chatLog = document.getElementById("chat-log");
  const welcome = chatLog.querySelector(".welcome-section");
  if (!welcome) return;
  const hasMessages = chatLog.querySelectorAll(".msg").length > 0;
  welcome.style.display = hasMessages ? "none" : "flex";
}

function appendMessage(role, text, images, stats, sources) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "bot") {
    const avatarWrap = document.createElement("div");
    avatarWrap.className = "msg-avatar";
    avatarWrap.innerHTML = `<svg width="48" height="48" viewBox="0 0 80 80" fill="none">
      <ellipse cx="39" cy="40.5" rx="39" ry="40.5" fill="#1A76B9"/>
      <ellipse cx="38.5" cy="41" rx="30.5" ry="31" fill="white"/>
      <ellipse cx="26.5" cy="43" rx="4.5" ry="5" fill="#D25332"/>
      <ellipse cx="50.5" cy="43" rx="4.5" ry="5" fill="#D25332"/>
      <path d="M30.696 0.5L28.9313 8.35455L25.4019 13.5909L17.9019 19.2636L11.2843 22.7545L5.54899 24.5H0.696045L2.46075 19.2636L5.54899 13.5909L9.51957 9.22727L13.9313 5.3L19.2255 2.68182L25.4019 0.5H30.696Z" fill="#D25332" transform="translate(7, 11)"/>
      <path d="M0.614258 0.5L2.37896 9.00909L5.90838 14.6818L13.4084 20.8273L20.026 24.6091L25.7613 26.5H30.6143L28.8496 20.8273L25.7613 14.6818L21.7907 9.95455L17.379 5.7L12.0848 2.86364L5.90838 0.5H0.614258Z" fill="#D25332" transform="translate(38, 11)"/>
      <path d="M1.28198 0.882324C1.28198 0.882324 8.10016 12.1323 16.282 0.882324" stroke="#D25332" stroke-width="3" fill="none" transform="translate(31, 59)"/>
    </svg>`;
    el.appendChild(avatarWrap);
    if (typeof marked !== "undefined") {
      const content = document.createElement("div");
      content.className = "msg-content";
      content.innerHTML = marked.parse(text);
      if (images && images.length) {
        const gallery = document.createElement("div");
        gallery.className = "msg-gallery";
        images.forEach((img, idx) => {
          const item = document.createElement("div");
          item.className = "msg-gallery-item";
          const thumb = document.createElement("img");
          thumb.className = "inline-image";
          thumb.src = img.src;
          thumb.alt = img.alt || "";
          thumb.loading = "lazy";
          thumb.addEventListener("click", () => openImagePopup(images, idx));
          item.appendChild(thumb);
          const meta = document.createElement("div");
          meta.className = "msg-gallery-meta";
          if (img.page) {
            const label = document.createElement("div");
            label.className = "msg-gallery-label";
            label.textContent = `стр. ${img.page}`;
            meta.appendChild(label);
          }
          if (img.description) {
            const desc = document.createElement("div");
            desc.className = "msg-gallery-description";
            desc.textContent = img.description;
            meta.appendChild(desc);
          } else if (img.excerpt) {
            const excerpt = document.createElement("div");
            excerpt.className = "msg-gallery-excerpt";
            excerpt.textContent = img.excerpt.substring(0, 120);
            meta.appendChild(excerpt);
          }
          item.appendChild(meta);
          gallery.appendChild(item);
        });
        content.appendChild(gallery);
      }
      el.appendChild(content);
    } else {
      const content = document.createElement("div");
      content.className = "msg-content";
      content.textContent = text;
      el.appendChild(content);
    }
  } else {
    el.textContent = text;
  }
  if (sources && sources.length) {
    const srcWrap = document.createElement("div");
    srcWrap.className = "msg-sources";
    const head = document.createElement("div");
    head.className = "msg-sources-head";
    head.textContent = "Источники";
    srcWrap.appendChild(head);
    sources.forEach((s) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "msg-source-chip";
      chip.textContent = `${s.source_file}${s.page_number ? ` · стр. ${s.page_number}` : ""}`;
      chip.title = s.excerpt || "";
      chip.addEventListener("click", () => openSourcePreview(s.source_file, s.chunk_id, s.page_number));
      srcWrap.appendChild(chip);
    });
    el.appendChild(srcWrap);
  }
  if (stats && (stats.retrieval_ms != null || stats.generation_ms != null)) {
    const statEl = document.createElement("div");
    statEl.className = "msg-stats";
    const r = stats.retrieval_ms != null ? (stats.retrieval_ms / 1000).toFixed(2) : "—";
    const g = stats.generation_ms != null ? (stats.generation_ms / 1000).toFixed(2) : "—";
    const t = stats.total_ms != null ? (stats.total_ms / 1000).toFixed(2) : "—";
    statEl.textContent = `⏱ RAG-поиск: ${r}с · генерация ИИ: ${g}с · всего: ${t}с`;
    el.appendChild(statEl);
  }
  const chatLog = document.getElementById("chat-log");
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  toggleWelcome();
}

function showTyping() {
  const chatLog = document.getElementById("chat-log");
  const typing = document.createElement("div");
  typing.className = "typing-indicator";
  typing.id = "typing-indicator";
  typing.innerHTML = `
    <span>София печатает</span>
    <span class="typing-dots"><span></span><span></span><span></span></span>
  `;
  chatLog.appendChild(typing);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function showStatus(text) {
  const bar = document.getElementById("status-bar");
  const txt = document.getElementById("status-text");
  if (!bar || !txt) return;
  txt.textContent = text;
  bar.classList.remove("hidden", "done");
}

function updateStatus(text) {
  const txt = document.getElementById("status-text");
  if (txt) txt.textContent = text;
}

function hideStatus(delay = 2000) {
  const bar = document.getElementById("status-bar");
  if (!bar) return;
  bar.classList.add("done");
  setTimeout(() => {
    bar.style.animation = "statusFadeOut 0.3s ease forwards";
    setTimeout(() => {
      bar.classList.add("hidden");
      bar.style.animation = "";
    }, 300);
  }, delay);
}

function clearChat() {
  document.getElementById("chat-log").querySelectorAll(".msg, .typing-indicator").forEach((el) => el.remove());
  state.currentSources = [];
  restoreSourcesView();
  toggleWelcome();
}

function saveToHistory(question) {
  state.chatHistory.push(question);
  if (state.chatHistory.length > 50) state.chatHistory.shift();
  localStorage.setItem("chatHistory", JSON.stringify(state.chatHistory));
  state.historyIndex = state.chatHistory.length;
}

async function submitQuestion(question) {
  appendMessage("user", question);
  saveToHistory(question);
  showTyping();
  const t0 = Date.now();
  showStatus("🔍 Ищу релевантные документы...");
  try {
    await new Promise(r => setTimeout(r, 200));
    updateStatus("📡 Отправляю запрос в API...");
    const result = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
    hideTyping();
    state.currentSources = result.sources;
    appendMessage("bot", result.answer, extractImages(result.sources, result.image_descriptions), {
      retrieval_ms: result.retrieval_ms,
      generation_ms: result.generation_ms,
      total_ms: result.total_ms,
    }, result.sources);
    renderSources(result.sources);
    const srcCount = result.sources ? result.sources.length : 0;
    const imgCount = result.image_descriptions ? Object.keys(result.image_descriptions).length : 0;
    const parts = [];
    parts.push(`${srcCount} ${declension(srcCount, "источник", "источника", "источников")}`);
    if (imgCount) parts.push(`${imgCount} ${declension(imgCount, "изображение", "изображения", "изображений")}`);
    updateStatus(`✅ Найдено: ${parts.join(" · ")} — ${elapsed}с`);
    hideStatus(4000);
    if (result.llm_model) {
      document.getElementById("chat-model-badge").textContent = `модель: ${result.llm_model}`;
    }
  } catch (err) {
    hideTyping();
    hideStatus(1000);
    appendMessage("system", `Ошибка: ${err.message}`);
  }
}

function declension(n, one, two, five) {
  const abs = Math.abs(n) % 100;
  const last = abs % 10;
  if (abs > 10 && abs < 20) return five;
  if (last > 1 && last < 5) return two;
  if (last === 1) return one;
  return five;
}

function extractImages(sources, imageDescriptions) {
  if (!sources) return [];
  const exts = ["jpg", "jpeg", "png", "gif", "webp"];
  const images = [];
  const seen = new Set();
  const descs = imageDescriptions || {};
  sources.forEach((s) => {
    // Картинки из PDF (извлечённые при индексации)
    if (s.image_paths && s.image_paths.length) {
      s.image_paths.forEach((img) => {
        const key = `${s.source_file}::${img}`;
        if (!seen.has(key)) {
          seen.add(key);
          const desc = s.image_descriptions && s.image_descriptions[img]
            ? s.image_descriptions[img]
            : descs[img] || "";
          images.push({
            src: `/api/media/images/${encodeURIComponent(s.source_file.replace(/\.[^.]+$/, ""))}/${encodeURIComponent(img)}`,
            alt: desc || img,
            description: desc,
            source: s.source_file,
            page: s.page_number,
            excerpt: s.excerpt || "",
          });
        }
      });
    }
    // Standalone image files
    if (exts.includes(s.source_file.split(".").pop().toLowerCase())) {
      const key = s.source_file;
      if (!seen.has(key)) {
        seen.add(key);
        images.push({
          src: `/api/media/documents/${encodeURIComponent(s.source_file)}`,
          alt: s.source_file,
          source: s.source_file,
          page: null,
        });
      }
    }
  });
  return images;
}

// ── Sources ──

function renderSources(sources) {
  state.sourcesMode = "sources";
  const list = document.getElementById("sources-list");
  list.innerHTML = "";
  if (!sources || !sources.length) {
    list.innerHTML = '<div class="sources-empty">Источники не найдены</div>';
    return;
  }
  sources.forEach((src) => {
    const block = document.createElement("div");
    block.className = "source-item";
    block.dataset.filename = src.source_file;
    block.dataset.chunkId = src.chunk_id != null ? src.chunk_id : "";
    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = `${src.source_file}${src.page_number ? ` · стр. ${src.page_number}` : ""}`;
    const excerpt = document.createElement("div");
    excerpt.className = "source-excerpt is-interactive";
    excerpt.textContent = src.excerpt || "";
    excerpt.dataset.filename = src.source_file;
    excerpt.dataset.chunkId = src.chunk_id != null ? src.chunk_id : "";
    excerpt.addEventListener("mouseenter", showOriginalTooltip);
    excerpt.addEventListener("mouseleave", hideOriginalTooltip);
    block.appendChild(title);
    block.appendChild(excerpt);
    block.addEventListener("click", () => openSourcePreview(src.source_file, src.chunk_id, src.page_number));
    list.appendChild(block);
  });
}

function restoreSourcesView() {
  state.sourcesMode = "sources";
  renderSources(state.currentSources);
}

function renderDocResults(results) {
  state.sourcesMode = "docs";
  const list = document.getElementById("sources-list");
  list.innerHTML = "";
  const backBtn = document.createElement("button");
  backBtn.className = "back-to-sources";
  backBtn.textContent = "← Назад к источникам";
  backBtn.addEventListener("click", restoreSourcesView);
  list.appendChild(backBtn);
  if (!results || !results.length) {
    list.innerHTML += '<div class="sources-empty">Ничего не найдено</div>';
    return;
  }
  results.forEach((doc) => {
    const div = document.createElement("div");
    div.className = "doc-result";
    div.innerHTML = `<div class="doc-name">${doc.name}</div><div class="doc-meta">${(CATEGORY_LABELS[doc.category] || doc.category)}, ${doc.size_kb} KB</div>`;
    div.addEventListener("click", () => openSourcePreview(doc.name, null));
    list.appendChild(div);
  });
}

function renderGroupedResults(categories, labels) {
  const list = document.getElementById("sources-list");
  list.innerHTML = "";
  const backBtn = document.createElement("button");
  backBtn.className = "back-to-sources";
  backBtn.textContent = "← Назад к источникам";
  backBtn.addEventListener("click", restoreSourcesView);
  list.appendChild(backBtn);
  let hasAny = false;
  Object.entries(categories).forEach(([cat, files]) => {
    if (!files || !files.length) return;
    hasAny = true;
    const title = document.createElement("div");
    title.className = "cat-group-title";
    title.textContent = labels[cat] || cat;
    list.appendChild(title);
    files.forEach((doc) => {
      const div = document.createElement("div");
      div.className = "doc-result";
      div.innerHTML = `<div class="doc-name">${doc.name}</div><div class="doc-meta">${doc.size_kb} KB</div>`;
      div.addEventListener("click", () => openSourcePreview(doc.name, null));
      list.appendChild(div);
    });
  });
  if (!hasAny) {
    list.innerHTML += '<div class="sources-empty">Нет документов в этой категории</div>';
  }
}

// ── Preview modal ──

async function openSourcePreview(filename, chunkId, pageNumber) {
  const modal = document.getElementById("preview-modal");
  const title = document.getElementById("preview-title");
  const body = document.getElementById("preview-body");
  const download = document.getElementById("preview-download");
  title.textContent = filename;
  download.href = `/api/media/documents/${encodeURIComponent(filename)}`;
  body.innerHTML = '<div class="sources-empty">Загрузка...</div>';
  modal.classList.remove("hidden");
  const ext = filename.split(".").pop().toLowerCase();
  const isImage = ["jpg", "jpeg", "png", "gif", "webp"].includes(ext);
  const enc = encodeURIComponent(filename);
  if (ext === "pdf") {
    let page = pageNumber;
    if ((page == null) && chunkId != null) {
      try {
        const d = await api(`/api/documents/${enc}/preview?chunk_id=${chunkId}`);
        page = d.metadata && d.metadata.page_number;
      } catch (_) { /* ignore */ }
    }
    await renderPdfViewer(body, `/api/media/documents/${enc}`, page, filename);
    return;
  }
  if (isImage) {
    body.innerHTML = `<img src="/api/media/documents/${enc}" class="preview-body-image" alt="${filename}">`;
    return;
  }
  try {
    const url = chunkId != null
      ? `/api/documents/${encodeURIComponent(filename)}/preview?chunk_id=${chunkId}`
      : `/api/documents/${encodeURIComponent(filename)}/preview`;
    const data = await api(url);
    if (data.type === "image") {
      body.innerHTML = `<img src="${data.media_url}" class="preview-body-image" alt="${filename}">`;
    } else if (data.type === "chunk") {
      body.innerHTML = `<div class="preview-body-content">${escapeHtml(data.content)}</div>`;
    } else {
      body.innerHTML = `<div class="preview-body-content">${escapeHtml(data.preview || "Нет содержимого")}</div>`;
    }
  } catch (err) {
    body.innerHTML = `<div class="sources-empty">Ошибка загрузки: ${err.message}</div>`;
  }
}

async function renderPdfViewer(container, url, targetPage, filename) {
  if (!window.pdfjsLib) {
    container.innerHTML = `<div class="sources-empty">PDF.js не загружен</div>`;
    return;
  }
  container.innerHTML = `
    <div class="pdf-viewer">
      <div class="pdf-toolbar">
        <button type="button" id="pdf-prev" class="pdf-nav">‹</button>
        <span id="pdf-pageinfo" class="pdf-pageinfo">Загрузка…</span>
        <button type="button" id="pdf-next" class="pdf-nav">›</button>
        <span class="pdf-filename" title="${escapeHtml(filename || "")}">${escapeHtml(filename || "")}</span>
      </div>
      <div class="pdf-canvas-wrap"><canvas id="pdf-canvas"></canvas></div>
    </div>`;

  const pdfjsLib = window.pdfjsLib;
  pdfjsLib.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.js";

  let pdf = null;
  let pageNum = targetPage && targetPage > 0 ? targetPage : 1;

  const canvas = document.getElementById("pdf-canvas");
  const ctx = canvas.getContext("2d");
  const pageinfo = document.getElementById("pdf-pageinfo");

  async function renderPage(n) {
    const page = await pdf.getPage(n);
    const viewport = page.getViewport({ scale: 1.6 });
    canvas.height = viewport.height;
    canvas.width = viewport.width;
    await page.render({ canvasContext: ctx, viewport }).promise;
    pageinfo.textContent = `стр. ${n} / ${pdf.numPages}`;
    pageNum = n;
  }

  try {
    const resp = await fetch(url, { headers: authHeaders() });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const buf = await resp.arrayBuffer();
    pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    if (pageNum > pdf.numPages) pageNum = pdf.numPages;
    await renderPage(pageNum);

    document.getElementById("pdf-prev").addEventListener("click", () => {
      if (pageNum > 1) renderPage(pageNum - 1);
    });
    document.getElementById("pdf-next").addEventListener("click", () => {
      if (pageNum < pdf.numPages) renderPage(pageNum + 1);
    });
  } catch (err) {
    container.innerHTML = `<div class="sources-empty">Ошибка отображения PDF: ${err.message}</div>`;
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

let originalTooltipEl = null;

function showOriginalTooltip(e) {
  const el = e.currentTarget;
  let text = el.dataset.full;
  if (!text) {
    const enc = encodeURIComponent(el.dataset.filename);
    const cid = el.dataset.chunkId;
    const url = cid
      ? `/api/documents/${enc}/preview?chunk_id=${cid}`
      : `/api/documents/${enc}/preview`;
    api(url)
      .then((data) => {
        text = data.content || data.preview || "(нет текста)";
        el.dataset.full = text;
        renderOriginalTooltip(el, text);
      })
      .catch(() => renderOriginalTooltip(el, "(не удалось загрузить оригинальный текст)"));
  } else {
    renderOriginalTooltip(el, text);
  }
}

function renderOriginalTooltip(el, text) {
  if (!originalTooltipEl) {
    originalTooltipEl = document.createElement("div");
    originalTooltipEl.id = "original-tooltip";
    originalTooltipEl.className = "original-tooltip hidden";
    document.body.appendChild(originalTooltipEl);
  }
  originalTooltipEl.textContent = text;
  originalTooltipEl.classList.remove("hidden");
  const rect = el.getBoundingClientRect();
  const tipRect = originalTooltipEl.getBoundingClientRect();
  let top = rect.top - tipRect.height - 8;
  if (top < 8) top = rect.bottom + 8;
  let left = rect.left;
  if (left + tipRect.width > window.innerWidth - 8) {
    left = window.innerWidth - tipRect.width - 8;
  }
  originalTooltipEl.style.top = `${top}px`;
  originalTooltipEl.style.left = `${Math.max(8, left)}px`;
}

function hideOriginalTooltip() {
  if (originalTooltipEl) originalTooltipEl.classList.add("hidden");
}

function openLightbox(src) {
  openImagePopup([{ src, alt: "" }], 0);
}

let _popupImages = [];
let _popupIndex = 0;

function openImagePopup(images, startIndex) {
  _popupImages = images;
  _popupIndex = startIndex || 0;
  _renderPopup();
  document.getElementById("image-lightbox").classList.remove("hidden");
}

function _renderPopup() {
  const img = _popupImages[_popupIndex];
  if (!img) return;
  document.getElementById("lightbox-img").src = img.src;
  document.getElementById("lightbox-img").alt = img.alt || "";
  const counter = document.getElementById("lightbox-counter");
  if (counter) {
    counter.textContent = _popupImages.length > 1
      ? `${_popupIndex + 1} / ${_popupImages.length}`
      : "";
  }
  const desc = document.getElementById("lightbox-excerpt");
  if (desc) {
    desc.textContent = img.excerpt || "";
    desc.style.display = img.excerpt ? "" : "none";
  }
  const prev = document.getElementById("lightbox-prev");
  const next = document.getElementById("lightbox-next");
  if (prev) prev.style.display = _popupImages.length > 1 ? "" : "none";
  if (next) next.style.display = _popupImages.length > 1 ? "" : "none";
}

function popupPrev() {
  if (_popupImages.length === 0) return;
  _popupIndex = (_popupIndex - 1 + _popupImages.length) % _popupImages.length;
  _renderPopup();
}

function popupNext() {
  if (_popupImages.length === 0) return;
  _popupIndex = (_popupIndex + 1) % _popupImages.length;
  _renderPopup();
}

// ── Category / Search / Commands ──

async function filterByCategory(cat) {
  try {
    const data = await api(`/api/documents?type=${encodeURIComponent(cat)}`);
    renderDocResults(data.files);
  } catch (err) {
    document.getElementById("sources-list").innerHTML = `<div class="sources-empty">${err.message}</div>`;
  }
}

async function searchDocuments(q) {
  try {
    const data = await api(`/api/documents?search=${encodeURIComponent(q)}`);
    renderDocResults(data.files);
  } catch (err) {
    document.getElementById("sources-list").innerHTML = `<div class="sources-empty">${err.message}</div>`;
  }
}

async function showRecent() {
  try {
    const data = await api("/api/documents?recent=true");
    renderDocResults(data.files);
  } catch (err) {
    document.getElementById("sources-list").innerHTML = `<div class="sources-empty">${err.message}</div>`;
  }
}

async function showByType() {
  try {
    const data = await api("/api/documents/categories");
    renderGroupedResults(data.categories, data.labels);
  } catch (err) {
    document.getElementById("sources-list").innerHTML = `<div class="sources-empty">${err.message}</div>`;
  }
}

// ── Theme ──

function toggleTheme() {
  const current = document.documentElement.getAttribute("data-theme");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next === "dark" ? "dark" : "");
  localStorage.setItem("theme", next);
}

function loadTheme() {
  const saved = localStorage.getItem("theme");
  if (saved === "dark") {
    document.documentElement.setAttribute("data-theme", "dark");
  }
}

// ── Admin / User Management ──

function openAdminModal() {
  document.getElementById("admin-modal").classList.remove("hidden");
  loadAdmin();
}

function closeAdminModal() {
  document.getElementById("admin-modal").classList.add("hidden");
}

function fillConfigForm(config) {
  state.config = config;
  document.getElementById("chat-model-badge").textContent = `модель: ${config.llm_model}`;
  document.getElementById("chat-embed-badge").textContent = `embed: ${config.embedding_model}`;
  document.getElementById("cfg-top-k").value = config.top_k;
  document.getElementById("cfg-fetch-k").value = config.fetch_k;
  document.getElementById("cfg-chunk-size").value = config.chunk_size;
  document.getElementById("cfg-chunk-overlap").value = config.chunk_overlap;
  document.getElementById("cfg-threshold").value = config.relevance_threshold;
  document.getElementById("llm-provider").value = config.llm_provider || "ollama";
  document.getElementById("llm-api-base").value = config.llm_api_base_url || "";
  document.getElementById("whisper-model").value = config.whisper_model || "tiny";
  document.getElementById("llm-model-input").value = config.llm_model || "";
  toggleProviderConfig(config.llm_provider);
}

function toggleProviderConfig(provider) {
  const oa = provider === "openai";
  document.getElementById("openai-config").style.display = oa ? "block" : "none";
  document.getElementById("llm-model-group").style.display = oa ? "none" : "block";
  document.getElementById("llm-model-custom").style.display = oa ? "block" : "none";
}

function fillModelSelects(models, current) {
  ["llm-model", "embed-model"].forEach((id) => {
    const sel = document.getElementById(id);
    sel.innerHTML = "";
  });
  models.forEach((model) => {
    const addOpt = (id, selectedName) => {
      const sel = document.getElementById(id);
      const opt = document.createElement("option");
      opt.value = model.name;
      opt.textContent = `${model.name}${model.size_gb ? ` (${model.size_gb} GB)` : ""}`;
      if (model.name === selectedName) opt.selected = true;
      sel.appendChild(opt);
    };
    addOpt("llm-model", current.llm_model);
    addOpt("embed-model", current.embedding_model);
  });
}

function renderAdminUsers() {
  const area = document.getElementById("admin-user-area");
  const list = document.getElementById("users-list");
  if (!state.user || state.user.role !== "admin") {
    area.innerHTML = "";
    list.innerHTML = '<div class="sources-empty">Управление пользователями доступно только администратору</div>';
    return;
  }
  area.innerHTML = `
    <div class="create-user-form">
      <input id="new-user-name" type="text" placeholder="Новый логин">
      <input id="new-user-pass" type="password" placeholder="Пароль">
      <button id="create-user-btn">Создать</button>
    </div>
  `;
  document.getElementById("create-user-btn").addEventListener("click", createUser);
  loadUsers();
}

async function loadUsers() {
  if (!state.user || state.user.role !== "admin") return;
  try {
    const data = await api("/api/admin/users");
    const list = document.getElementById("users-list");
    list.innerHTML = "";
    data.users.forEach((u) => {
      const row = document.createElement("div");
      row.className = "user-row";
      const isSelf = u.username === state.user.username;
      row.innerHTML = `
        <span class="user-name">${u.username}</span>
        <span class="user-role">${u.role}</span>
        ${u.username === "admin" || isSelf
          ? '<span class="cannot-delete">—</span>'
          : `<button data-del="${u.username}">удалить</button>`
        }
      `;
      list.appendChild(row);
    });
    list.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", async () => {
        if (!confirm(`Удалить ${btn.dataset.del}?`)) return;
        try {
          await api(`/api/admin/users/${encodeURIComponent(btn.dataset.del)}`, { method: "DELETE" });
          await loadUsers();
        } catch (err) { alert(err.message); }
      });
    });
  } catch (err) {
    document.getElementById("users-list").innerHTML = `<div class="sources-empty">${err.message}</div>`;
  }
}

async function createUser() {
  const name = document.getElementById("new-user-name").value.trim();
  const pass = document.getElementById("new-user-pass").value;
  if (!name || !pass) { alert("Введите логин и пароль"); return; }
  try {
    await api("/api/admin/users", {
      method: "POST",
      body: JSON.stringify({ username: name, password: pass, role: "user" }),
    });
    document.getElementById("new-user-name").value = "";
    document.getElementById("new-user-pass").value = "";
    await loadUsers();
  } catch (err) { alert(err.message); }
}

function renderIndexStats(status) {
  document.getElementById("index-stats").innerHTML = `
    <div class="stat"><div class="hint">Документов</div><div class="value">${status.documents_count}</div></div>
    <div class="stat"><div class="hint">Чанков в индексе</div><div class="value">${status.indexed_chunks}</div></div>
    <div class="stat"><div class="hint">План чанков</div><div class="value">${status.planned_chunks}</div></div>
    <div class="stat"><div class="hint">Embedding</div><div class="value" style="font-size:12px">${status.embedding_model}</div></div>
    <div class="stat"><div class="hint">Chroma</div><div class="value" style="font-size:14px">${status.chroma_ready ? "готово" : "пусто"}</div></div>
    <div class="stat"><div class="hint">Папка</div><div class="value" style="font-size:12px">${status.documents_dir}</div></div>
  `;
  const tbody = document.getElementById("docs-table");
  tbody.innerHTML = "";
  status.files.forEach((file) => {
    const ext = file.name.split(".").pop().toLowerCase();
    const isImage = ["jpg", "jpeg", "png", "gif", "webp"].includes(ext);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${file.name}</td>
      <td>${file.size_kb} KB</td>
      <td>${file.planned_chunks}</td>
      <td>${file.indexed_chunks}</td>
      <td>
        ${isImage ? `<a href="/api/media/documents/${encodeURIComponent(file.name)}" target="_blank" class="preview-link">👁️</a>` : ""}
        <button type="button" data-delete="${file.name}">удалить</button>
      </td>
    `;
    tbody.appendChild(tr);
  });
  tbody.querySelectorAll("[data-delete]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (!confirm(`Удалить ${btn.dataset.delete}?`)) return;
      try {
        await api(`/api/admin/documents/${encodeURIComponent(btn.dataset.delete)}?auto_reindex=true`, { method: "DELETE" });
        await loadAdmin();
      } catch (err) { alert(err.message); }
    });
  });
}

async function loadConfig() {
  try {
    const config = await api("/api/config");
    fillConfigForm(config);
  } catch (err) {
    appendMessage("system", `Не удалось загрузить конфиг: ${err.message}`);
  }
}

async function loadAdmin() {
  try {
    const [modelsPayload, status] = await Promise.all([
      api("/api/admin/models"),
      api("/api/admin/index/status"),
    ]);
    fillModelSelects(modelsPayload.models, modelsPayload.current);
    fillConfigForm(modelsPayload.current);
    renderIndexStats(status);
    renderAdminUsers();
    document.getElementById("models-status").textContent = `Доступно моделей Ollama: ${modelsPayload.models.length}`;
  } catch (err) {
    document.getElementById("models-status").textContent = err.message;
  }
}

// ── Microphone ──

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioChunks = [];
    state.mediaRecorder = new MediaRecorder(stream);
    state.mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) state.audioChunks.push(e.data); };
    state.mediaRecorder.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(state.audioChunks, { type: "audio/webm" });
      await sendAudio(blob);
    };
    state.mediaRecorder.start();
    state.isRecording = true;
    document.getElementById("mic-btn").textContent = "⏹";
    document.getElementById("mic-btn").classList.add("recording");
  } catch (err) {
    alert("Микрофон не доступен: " + err.message);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.isRecording) {
    state.mediaRecorder.stop();
    state.isRecording = false;
    document.getElementById("mic-btn").textContent = "🎤";
    document.getElementById("mic-btn").classList.remove("recording");
  }
}

async function sendAudio(blob) {
  const formData = new FormData();
  formData.append("file", blob, "audio.webm");
  try {
    const response = await fetch("/api/stt", { method: "POST", body: formData });
    let data;
    try { data = await response.json(); } catch { data = {}; }
    if (!response.ok) throw new Error(data.detail || "STT error");
    const text = (data.text || "").trim();
    if (text) {
      document.getElementById("ask-input").value = text;
      await submitQuestion(text);
    }
  } catch (err) {
    appendMessage("system", `Ошибка распознавания: ${err.message}`);
  }
}

// ── Init ──

document.addEventListener("DOMContentLoaded", async () => {
  loadTheme();
  const authed = await checkAuth();
  if (authed) {
    loadConfig();
  }
});

// ── Auth UI Events ──

document.getElementById("login-btn").addEventListener("click", async () => {
  const username = document.getElementById("login-username").value.trim();
  const password = document.getElementById("login-password").value;
  if (!username || !password) {
    document.getElementById("login-error").textContent = "Введите логин и пароль";
    return;
  }
  await doLogin(username, password);
});

document.getElementById("login-username").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("login-btn").click();
});
document.getElementById("login-password").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("login-btn").click();
});

document.getElementById("login-to-welcome-btn").addEventListener("click", () => {
  document.getElementById("login-error").textContent = "";
  showScreen("screen-welcome");
});

document.getElementById("welcome-to-login-btn").addEventListener("click", () => {
  document.getElementById("register-form").classList.add("hidden");
  showScreen("screen-login");
});

document.getElementById("welcome-to-register-btn").addEventListener("click", () => {
  document.getElementById("register-form").classList.remove("hidden");
});

document.getElementById("register-btn").addEventListener("click", async () => {
  const username = document.getElementById("reg-username").value.trim();
  const password = document.getElementById("reg-password").value;
  if (!username || !password) {
    document.getElementById("register-error").textContent = "Введите логин и пароль";
    return;
  }
  await doRegister(username, password);
});

// ── Send ──

document.getElementById("send-btn").addEventListener("click", () => {
  const input = document.getElementById("ask-input");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  submitQuestion(question);
});

document.getElementById("ask-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    e.preventDefault();
    document.getElementById("send-btn").click();
  }
});

document.getElementById("ask-input").addEventListener("keydown", (e) => {
  if (e.key === "ArrowUp" && state.chatHistory.length) {
    if (state.historyIndex > 0) {
      state.historyIndex--;
      document.getElementById("ask-input").value = state.chatHistory[state.historyIndex];
    }
  } else if (e.key === "ArrowDown") {
    if (state.historyIndex < state.chatHistory.length - 1) {
      state.historyIndex++;
      document.getElementById("ask-input").value = state.chatHistory[state.historyIndex];
    } else {
      state.historyIndex = state.chatHistory.length;
      document.getElementById("ask-input").value = "";
    }
  }
});

// Microphone
document.getElementById("mic-btn").addEventListener("click", () => {
  if (state.isRecording) stopRecording(); else startRecording();
});

// Example question
document.getElementById("example-question").addEventListener("click", () => {
  document.getElementById("ask-input").value = document.getElementById("example-question").textContent.trim();
  document.getElementById("send-btn").click();
});

// Command buttons
document.querySelectorAll(".cmd-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const cmd = btn.dataset.cmd;
    const input = document.getElementById("ask-input");
    switch (cmd) {
      case "/search":
        input.value = "";
        document.getElementById("search-bar").classList.remove("hidden");
        document.getElementById("search-input").focus();
        break;
      case "/recent":
        showRecent();
        break;
      case "/by_type":
        showByType();
        break;
      case "/by_date":
        showRecent();
        break;
      case "/back":
        if (state.chatHistory.length > 1) {
          state.historyIndex = state.chatHistory.length - 2;
          input.value = state.chatHistory[state.historyIndex] || "";
        }
        break;
      default:
        input.value = cmd;
        document.getElementById("send-btn").click();
    }
  });
});

// Category buttons
document.querySelectorAll(".src-cat").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".src-cat").forEach((b) => b.classList.remove("src-cat-active"));
    btn.classList.add("src-cat-active");
    filterByCategory(btn.dataset.cat);
  });
});

// Search bar
document.getElementById("search-go").addEventListener("click", () => {
  const q = document.getElementById("search-input").value.trim();
  if (q) searchDocuments(q);
});
document.getElementById("search-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("search-go").click();
});

// New chat
document.getElementById("new-chat-btn").addEventListener("click", clearChat);

// Three dots dropdown
document.getElementById("dots-menu").addEventListener("click", (e) => {
  e.stopPropagation();
  document.getElementById("dots-dropdown").classList.toggle("hidden");
});
document.addEventListener("click", () => {
  document.getElementById("dots-dropdown").classList.add("hidden");
});
document.querySelectorAll("#dots-dropdown button").forEach((btn) => {
  btn.addEventListener("click", (e) => {
    e.stopPropagation();
    document.getElementById("dots-dropdown").classList.add("hidden");
    const action = btn.dataset.action;
    if (action === "help") {
      document.getElementById("help-modal").classList.remove("hidden");
    } else if (action === "theme") {
      toggleTheme();
    } else if (action === "logout") {
      doLogout();
    }
  });
});

// Share
document.getElementById("btn-share").addEventListener("click", () => {
  if (navigator.share) {
    navigator.share({ title: "София — ИИ-помощник", url: location.href }).catch(() => {});
  } else {
    navigator.clipboard.writeText(location.href).then(() => alert("Ссылка скопирована")).catch(() => {});
  }
});

// New account (in topbar)
document.getElementById("btn-new-account").addEventListener("click", () => {
  document.getElementById("register-form").classList.remove("hidden");
  showScreen("screen-welcome");
});

// Admin modal
document.getElementById("btn-settings").addEventListener("click", openAdminModal);
document.getElementById("admin-modal-close").addEventListener("click", closeAdminModal);
document.getElementById("admin-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) closeAdminModal();
});

// Preview modal
document.getElementById("preview-modal-close").addEventListener("click", () => {
  document.getElementById("preview-modal").classList.add("hidden");
});
document.getElementById("preview-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) document.getElementById("preview-modal").classList.add("hidden");
});

// Help modal
document.getElementById("help-modal-close").addEventListener("click", () => {
  document.getElementById("help-modal").classList.add("hidden");
});
document.getElementById("help-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) document.getElementById("help-modal").classList.add("hidden");
});

// Lightbox
document.getElementById("lightbox-close").addEventListener("click", () => {
  document.getElementById("image-lightbox").classList.add("hidden");
});
document.getElementById("image-lightbox").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) document.getElementById("image-lightbox").classList.add("hidden");
});
document.getElementById("lightbox-prev").addEventListener("click", (e) => {
  e.stopPropagation();
  popupPrev();
});
document.getElementById("lightbox-next").addEventListener("click", (e) => {
  e.stopPropagation();
  popupNext();
});
document.addEventListener("keydown", (e) => {
  const lb = document.getElementById("image-lightbox");
  if (lb.classList.contains("hidden")) return;
  if (e.key === "Escape") lb.classList.add("hidden");
  if (e.key === "ArrowLeft") popupPrev();
  if (e.key === "ArrowRight") popupNext();
});

// Provider toggle
document.getElementById("llm-provider").addEventListener("change", (e) => toggleProviderConfig(e.target.value));

// Apply provider
document.getElementById("apply-provider").addEventListener("click", async () => {
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({
        llm_provider: document.getElementById("llm-provider").value,
        llm_api_base_url: document.getElementById("llm-api-base").value,
        llm_api_key: document.getElementById("llm-api-key").value,
      }),
    });
    fillConfigForm(config);
    document.getElementById("provider-status").textContent = "Провайдер применён.";
  } catch (err) { alert(err.message); }
});

// Apply whisper
document.getElementById("apply-whisper").addEventListener("click", async () => {
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({ whisper_model: document.getElementById("whisper-model").value }),
    });
    fillConfigForm(config);
    document.getElementById("whisper-status").textContent = "Модель STT сохранена.";
  } catch (err) { alert(err.message); }
});

// Apply models
document.getElementById("apply-models").addEventListener("click", async () => {
  const provider = document.getElementById("llm-provider").value;
  const llmModel = provider === "openai"
    ? document.getElementById("llm-model-input").value
    : document.getElementById("llm-model").value;
  if (!llmModel) { alert("Укажите модель LLM"); return; }
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({
        llm_model: llmModel,
        embedding_model: document.getElementById("embed-model").value,
      }),
    });
    fillConfigForm(config);
    if (config.reindex_triggered) {
      document.getElementById("models-status").textContent =
        "Модели применены. Запущена переиндексация (смена эмбеддинга) — может занять время, следите за статусом индекса.";
    } else {
      document.getElementById("models-status").textContent = "Модели применены.";
    }
  } catch (err) { alert(err.message); }
});

// Apply RAG
document.getElementById("apply-rag").addEventListener("click", async () => {
  const status = document.getElementById("rag-status");
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({
        top_k: Number(document.getElementById("cfg-top-k").value),
        fetch_k: Number(document.getElementById("cfg-fetch-k").value),
        chunk_size: Number(document.getElementById("cfg-chunk-size").value),
        chunk_overlap: Number(document.getElementById("cfg-chunk-overlap").value),
        relevance_threshold: Number(document.getElementById("cfg-threshold").value),
      }),
    });
    fillConfigForm(config);
    if (config.reindex_triggered) {
      status.textContent =
        "Параметры сохранены. Запущена переиндексация (смена размера чанка) — может занять время.";
    } else {
      status.textContent = "Параметры RAG сохранены (применяются сразу).";
    }
  } catch (err) { alert(err.message); }
});

// Reindex
document.getElementById("reindex-btn").addEventListener("click", async () => {
  const btn = document.getElementById("reindex-btn");
  btn.disabled = true;
  btn.textContent = "Индексация…";
  try {
    await api("/api/admin/reindex", { method: "POST", body: JSON.stringify({ recreate: true }) });
    await loadAdmin();
  } catch (err) { alert(err.message); } finally { btn.disabled = false; btn.textContent = "Переиндексировать"; }
});

// Upload
document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = document.getElementById("upload-input").files[0];
  if (!file) return;
  const formData = new FormData();
  formData.append("file", file);
  const autoReindex = document.getElementById("auto-reindex").checked;
  try {
    const response = await fetch(`/api/admin/documents/upload?auto_reindex=${autoReindex}`, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed");
    document.getElementById("upload-status").textContent = `Загружен: ${data.upload.name}`;
    document.getElementById("upload-input").value = "";
    await loadAdmin();
  } catch (err) { document.getElementById("upload-status").textContent = err.message; }
});

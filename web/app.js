const state = {
  config: null,
  mediaRecorder: null,
  audioChunks: [],
  isRecording: false,
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || response.statusText || "API error");
  }
  return data;
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `panel-${name}`);
  });
  if (name === "admin") {
    loadAdmin();
  }
}

function appendMessage(containerId, role, text) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "bot" && typeof marked !== "undefined") {
    el.innerHTML = marked.parse(text);
  } else {
    el.textContent = text;
  }
  const container = document.getElementById(containerId);
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
}

function renderSources(sources) {
  const panel = document.getElementById("sources-panel");
  panel.innerHTML = "";
  if (!sources || !sources.length) {
    panel.classList.add("muted");
    panel.textContent = "Источники не найдены";
    return;
  }
  panel.classList.remove("muted");
  sources.forEach((src) => {
    const block = document.createElement("div");
    block.className = "source";
    const title = document.createElement("div");
    title.className = "source-title";
    title.textContent = `${src.source_file}${src.page_number ? ` · стр. ${src.page_number}` : ""}${src.chunk_id != null ? ` · chunk #${src.chunk_id}` : ""}`;
    const ext = src.source_file.split(".").pop().toLowerCase();
    if (ext === "pdf") {
      const btn = document.createElement("button");
      btn.className = "pdf-btn";
      btn.textContent = "📄 Открыть PDF";
      btn.addEventListener("click", () => openPdfPreview(src.source_file, src.page_number));
      title.appendChild(btn);
    }
    if (["jpg", "jpeg", "png", "gif", "webp"].includes(ext)) {
      const link = document.createElement("a");
      link.href = `/api/media/documents/${encodeURIComponent(src.source_file)}`;
      link.target = "_blank";
      link.textContent = "🖼️ Открыть";
      link.className = "pdf-btn";
      title.appendChild(link);
    }
    const excerpt = document.createElement("div");
    excerpt.className = "muted small";
    excerpt.textContent = src.excerpt || "";
    block.appendChild(title);
    block.appendChild(excerpt);
    panel.appendChild(block);
  });
}

function openPdfPreview(filename, pageNumber) {
  const title = `PDF: ${filename}${pageNumber ? ` — стр. ${pageNumber}` : ""}`;
  document.getElementById("modal-title").textContent = title;
  const body = document.getElementById("modal-body");
  body.innerHTML = "";
  const iframe = document.createElement("iframe");
  iframe.src = `/api/media/documents/${encodeURIComponent(filename)}${pageNumber ? `#page=${pageNumber}` : ""}`;
  iframe.className = "pdf-frame";
  body.appendChild(iframe);
  document.getElementById("pdf-modal").classList.remove("hidden");
}

function fillConfigForm(config) {
  state.config = config;
  document.getElementById("user-model-badge").textContent = `модель: ${config.llm_model}`;
  const providerBadge = document.getElementById("user-provider-badge");
  if (config.llm_provider === "openai") {
    providerBadge.textContent = "Polza.ai / OpenAI";
    providerBadge.classList.add("provider-openai");
  } else {
    providerBadge.textContent = "Ollama (локально)";
    providerBadge.classList.remove("provider-openai");
  }
  document.getElementById("cfg-top-k").value = config.top_k;
  document.getElementById("cfg-fetch-k").value = config.fetch_k;
  document.getElementById("cfg-chunk-size").value = config.chunk_size;
  document.getElementById("cfg-chunk-overlap").value = config.chunk_overlap;
  document.getElementById("cfg-threshold").value = config.relevance_threshold;
  document.getElementById("llm-provider").value = config.llm_provider || "ollama";
  document.getElementById("llm-api-base").value = config.llm_api_base_url || "";
  document.getElementById("whisper-model").value = config.whisper_model || "karanchopda333/whisper";
  toggleProviderConfig(config.llm_provider);
}

function toggleProviderConfig(provider) {
  const cfg = document.getElementById("openai-config");
  cfg.style.display = provider === "openai" ? "block" : "none";
}

function fillModelSelects(models, current) {
  const llmSelect = document.getElementById("llm-model");
  const embedSelect = document.getElementById("embed-model");
  llmSelect.innerHTML = "";
  embedSelect.innerHTML = "";

  models.forEach((model) => {
    const addOpt = (sel, selectedName) => {
      const opt = document.createElement("option");
      opt.value = model.name;
      opt.textContent = `${model.name}${model.size_gb ? ` (${model.size_gb} GB)` : ""}`;
      if (model.name === selectedName) opt.selected = true;
      sel.appendChild(opt);
    };
    addOpt(llmSelect, current.llm_model);
    addOpt(embedSelect, current.embedding_model);
  });
}

function renderIndexStats(status) {
  const stats = document.getElementById("index-stats");
  stats.innerHTML = `
    <div class="stat"><div class="muted small">Документов</div><div class="value">${status.documents_count}</div></div>
    <div class="stat"><div class="muted small">Чанков в индексе</div><div class="value">${status.indexed_chunks}</div></div>
    <div class="stat"><div class="muted small">План чанков</div><div class="value">${status.planned_chunks}</div></div>
    <div class="stat"><div class="muted small">Embedding</div><div class="value small">${status.embedding_model}</div></div>
    <div class="stat"><div class="muted small">Chroma</div><div class="value small">${status.chroma_ready ? "готово" : "пусто"}</div></div>
    <div class="stat"><div class="muted small">Папка</div><div class="value small">${status.documents_dir}</div></div>
  `;

  const tbody = document.getElementById("docs-table");
  tbody.innerHTML = "";
  status.files.forEach((file) => {
    const tr = document.createElement("tr");
    const ext = file.name.split(".").pop().toLowerCase();
    const isImage = ["jpg", "jpeg", "png", "gif", "webp"].includes(ext);
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
        await api(`/api/admin/documents/${encodeURIComponent(btn.dataset.delete)}?auto_reindex=true`, {
          method: "DELETE",
        });
        await loadAdmin();
      } catch (err) {
        alert(err.message);
      }
    });
  });
}

async function loadConfig() {
  const config = await api("/api/config");
  fillConfigForm(config);
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
    document.getElementById("models-status").textContent =
      `Доступно моделей Ollama: ${modelsPayload.models.length}`;
  } catch (err) {
    document.getElementById("models-status").textContent = err.message;
  }
}

// Microphone
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.audioChunks = [];
    state.mediaRecorder = new MediaRecorder(stream);
    state.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.audioChunks.push(e.data);
    };
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
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "STT error");
    document.getElementById("ask-input").value = data.text;
    document.getElementById("ask-form").dispatchEvent(new Event("submit"));
  } catch (err) {
    appendMessage("chat-log", "system", `Ошибка распознавания: ${err.message}`);
  }
}

// Tab switching
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

// Ask question
document.getElementById("ask-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = document.getElementById("ask-input");
  const question = input.value.trim();
  if (!question) return;

  appendMessage("chat-log", "user", question);
  input.value = "";

  try {
    const result = await api("/api/ask", {
      method: "POST",
      body: JSON.stringify({ question }),
    });
    appendMessage("chat-log", "bot", result.answer);
    renderSources(result.sources);
    if (result.llm_model) {
      document.getElementById("user-model-badge").textContent = `модель: ${result.llm_model}`;
    }
  } catch (err) {
    appendMessage("chat-log", "system", `Ошибка: ${err.message}`);
  }
});

// Microphone button
document.getElementById("mic-btn").addEventListener("click", () => {
  if (state.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
});

// Provider
document.getElementById("llm-provider").addEventListener("change", (e) => {
  toggleProviderConfig(e.target.value);
});

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
  } catch (err) {
    alert(err.message);
  }
});

// Whisper model
document.getElementById("apply-whisper").addEventListener("click", async () => {
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({
        whisper_model: document.getElementById("whisper-model").value,
      }),
    });
    fillConfigForm(config);
    document.getElementById("whisper-status").textContent = "Модель STT сохранена.";
  } catch (err) {
    alert(err.message);
  }
});

// Ollama models
document.getElementById("apply-models").addEventListener("click", async () => {
  try {
    const config = await api("/api/admin/config", {
      method: "PATCH",
      body: JSON.stringify({
        llm_model: document.getElementById("llm-model").value,
        embedding_model: document.getElementById("embed-model").value,
      }),
    });
    fillConfigForm(config);
    document.getElementById("models-status").textContent = "Модели применены. Если меняли embedding — переиндексируйте.";
  } catch (err) {
    alert(err.message);
  }
});

// RAG params
document.getElementById("apply-rag").addEventListener("click", async () => {
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
    alert("Параметры RAG сохранены");
  } catch (err) {
    alert(err.message);
  }
});

// Reindex
document.getElementById("reindex-btn").addEventListener("click", async () => {
  const btn = document.getElementById("reindex-btn");
  btn.disabled = true;
  btn.textContent = "Индексация…";
  try {
    await api("/api/admin/reindex", {
      method: "POST",
      body: JSON.stringify({ recreate: true }),
    });
    await loadAdmin();
  } catch (err) {
    alert(err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Переиндексировать";
  }
});

// Upload
document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("upload-input");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  const autoReindex = document.getElementById("auto-reindex").checked;

  try {
    const response = await fetch(`/api/admin/documents/upload?auto_reindex=${autoReindex}`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed");
    document.getElementById("upload-status").textContent =
      `Загружен: ${data.upload.name}${data.reindex ? ` · чанков: ${data.reindex.chunks}` : ""}`;
    fileInput.value = "";
    await loadAdmin();
  } catch (err) {
    document.getElementById("upload-status").textContent = err.message;
  }
});

// PDF modal
document.getElementById("modal-close").addEventListener("click", () => {
  document.getElementById("pdf-modal").classList.add("hidden");
  document.getElementById("modal-body").innerHTML = "";
});

document.getElementById("pdf-modal").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) {
    document.getElementById("pdf-modal").classList.add("hidden");
    document.getElementById("modal-body").innerHTML = "";
  }
});

loadConfig().catch((err) => {
  appendMessage("chat-log", "system", `Не удалось загрузить конфиг: ${err.message}`);
});

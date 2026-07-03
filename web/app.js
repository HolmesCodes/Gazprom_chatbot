const state = {
  onboardingSessionId: null,
  config: null,
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
  el.textContent = text;
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
    const excerpt = document.createElement("div");
    excerpt.className = "muted small";
    excerpt.textContent = src.excerpt || "";
    block.appendChild(title);
    block.appendChild(excerpt);
    panel.appendChild(block);
  });
}

function fillConfigForm(config) {
  state.config = config;
  document.getElementById("user-model-badge").textContent = `модель: ${config.llm_model}`;
  document.getElementById("cfg-top-k").value = config.top_k;
  document.getElementById("cfg-fetch-k").value = config.fetch_k;
  document.getElementById("cfg-chunk-size").value = config.chunk_size;
  document.getElementById("cfg-chunk-overlap").value = config.chunk_overlap;
  document.getElementById("cfg-threshold").value = config.relevance_threshold;
}

function fillModelSelects(models, current) {
  const llmSelect = document.getElementById("llm-model");
  const embedSelect = document.getElementById("embed-model");
  llmSelect.innerHTML = "";
  embedSelect.innerHTML = "";

  models.forEach((model) => {
    const llmOption = document.createElement("option");
    llmOption.value = model.name;
    llmOption.textContent = `${model.name}${model.size_gb ? ` (${model.size_gb} GB)` : ""}`;
    if (model.name === current.llm_model) llmOption.selected = true;
    llmSelect.appendChild(llmOption);

    const embedOption = document.createElement("option");
    embedOption.value = model.name;
    embedOption.textContent = `${model.name}${model.size_gb ? ` (${model.size_gb} GB)` : ""}`;
    if (model.name === current.embedding_model) embedOption.selected = true;
    embedSelect.appendChild(embedOption);
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
    tr.innerHTML = `
      <td>${file.name}</td>
      <td>${file.size_kb} KB</td>
      <td>${file.planned_chunks}</td>
      <td>${file.indexed_chunks}</td>
      <td><button type="button" data-delete="${file.name}">удалить</button></td>
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

document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => setTab(btn.dataset.tab));
});

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

document.getElementById("onboarding-start").addEventListener("click", async () => {
  try {
    const result = await api("/api/onboarding/start", {
      method: "POST",
      body: JSON.stringify({ session_id: state.onboardingSessionId }),
    });
    state.onboardingSessionId = result.session_id;
    document.getElementById("onboarding-log").innerHTML = "";
    appendMessage("onboarding-log", "bot", result.reply);
  } catch (err) {
    appendMessage("onboarding-log", "system", `Ошибка: ${err.message}`);
  }
});

document.getElementById("onboarding-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.onboardingSessionId) {
    appendMessage("onboarding-log", "system", "Сначала нажмите «Старт»");
    return;
  }
  const input = document.getElementById("onboarding-input");
  const message = input.value.trim();
  if (!message) return;

  appendMessage("onboarding-log", "user", message);
  input.value = "";

  try {
    const result = await api("/api/onboarding/message", {
      method: "POST",
      body: JSON.stringify({ session_id: state.onboardingSessionId, message }),
    });
    appendMessage("onboarding-log", "bot", result.reply);
  } catch (err) {
    appendMessage("onboarding-log", "system", `Ошибка: ${err.message}`);
  }
});

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

loadConfig().catch((err) => {
  appendMessage("chat-log", "system", `Не удалось загрузить конфиг: ${err.message}`);
});

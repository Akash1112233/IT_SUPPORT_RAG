const chatBox   = document.getElementById("chat-box");
const chatForm  = document.getElementById("chat-form");
const input     = document.getElementById("question");
const sendBtn   = document.getElementById("send-btn");
const traceWrap = document.getElementById("trace-drawer");
const traceList = document.getElementById("trace-list");

function addMessage(text, role) {
    const div = document.createElement("div");
    div.className = `message ${role}`;
    div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
    return div;
}

function addBotBubble(html) {
    const div = document.createElement("div");
    div.className = "message bot";
    div.innerHTML = `<div class="bubble">${html}</div>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

function showTyping() {
    const div = document.createElement("div");
    div.className = "message bot";
    div.id = "typing";
    div.innerHTML = `<div class="bubble typing-dots"><span></span><span></span><span></span></div>`;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}
function hideTyping() {
    const el = document.getElementById("typing");
    if (el) el.remove();
}

function escapeHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
}

function renderTrace(trace) {
    if (!trace || !trace.length) { traceWrap.style.display = "none"; return; }
    traceWrap.style.display = "";
    traceList.innerHTML = trace.map(t => `<li>${escapeHtml(t)}</li>`).join("");
}

function renderCitations(citations) {
    if (!citations || !citations.length) return "";
    const items = citations.map(c => {
        if (c.url) return `<a href="${escapeHtml(c.url)}" target="_blank">${escapeHtml(c.title)}</a>`;
        return escapeHtml(c.title);
    });
    return `<div class="citations">📎 ${items.join(" · ")}</div>`;
}

function sourceBadge(source) {
    const labels = {
        private_kb: "Private KB",
        web_search: "Web Search",
        direct: "Direct",
        insufficient_evidence: "No Evidence",
    };
    return `<span class="source-badge">${labels[source] || source}</span>`;
}

chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = input.value.trim();
    if (!q) return;

    addMessage(q, "user");
    input.value = "";
    sendBtn.disabled = true;
    showTyping();

    try {
        const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question: q }),
        });
        hideTyping();

        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            addBotBubble(`⚠️ ${escapeHtml(err.detail || "Something went wrong.")}`);
            return;
        }

        const data = await res.json();
        const html = escapeHtml(data.answer)
            + sourceBadge(data.source_used)
            + renderCitations(data.citations);
        addBotBubble(html);
        renderTrace(data.trace);
    } catch (err) {
        hideTyping();
        addBotBubble("⚠️ Network error — is the server running?");
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const chatInput = document.getElementById("chat-input");
    const sendBtn = document.getElementById("send-btn");
    const chatHistory = document.getElementById("chat-history");
    const pdfUpload = document.getElementById("pdf-upload");
    const uploadStatus = document.getElementById("upload-status");
    const sidebar = document.getElementById("sidebar");
    const toggleSidebarBtn = document.getElementById("toggle-sidebar");
    const newChatBtn = document.getElementById("new-chat-btn");

    // Marked.js config for markdown rendering
    marked.setOptions({ breaks: true, gfm: true });

    // === PERSISTENCE LOGIC ===
    // Using sessionStorage instead of localStorage:
    // - Data is per-tab and automatically cleared when the tab/browser is closed
    // - Reopening the page starts a completely fresh session
    const saveChat = () => {
        // Do not save the typing indicator or active pipeline trackers
        let htmlToSave = chatHistory.innerHTML;
        htmlToSave = htmlToSave.replace(/<div class="typing-indicator">.*?<\/div>/g, '');
        sessionStorage.setItem("ragChatHistory", htmlToSave);
    };

    // Define the new, generic welcome message
    const welcomeMessageHTML = `
        <div class="message assistant-message welcome-message">
            <div class="avatar"><i class="fas fa-robot"></i></div>
            <div class="message-container">
                <div class="message-content">Hello! I am ready to help. Please upload a PDF and ask me anything about its contents.</div>
            </div>
        </div>`;

    const loadChat = () => {
        const savedHTML = sessionStorage.getItem("ragChatHistory");
        if (savedHTML && savedHTML.trim() !== "") {
            chatHistory.innerHTML = savedHTML;
            scrollToBottom();
        } else {
            chatHistory.innerHTML = welcomeMessageHTML;
        }
    };

    // One-time cleanup: remove old localStorage data from previous versions
    localStorage.removeItem("ragChatHistory");

    // Load history on startup (now uses sessionStorage — fresh per tab)
    loadChat();

    // New Chat Button - Clear both client-side and server-side state
    newChatBtn.addEventListener("click", async () => {
        sessionStorage.removeItem("ragChatHistory");
        chatHistory.innerHTML = welcomeMessageHTML;
        // Also clear server-side Weaviate data
        try {
            await fetch("/api/reset", { method: "POST" });
        } catch (e) {
            console.warn("Could not reset server session:", e);
        }
    });

    // Toggle Sidebar
    toggleSidebarBtn.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
    });

    // Auto-resize textarea
    chatInput.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
        sendBtn.disabled = this.value.trim() === "";
    });

    chatInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (!sendBtn.disabled) sendMessage();
        }
    });

    // Helpers
    const formatCitations = (text) => {
        return text.replace(/\[Page\s+(\d+(?:-\d+)?)\]/gi, '<span class="citation">Page $1</span>');
    };

    const scrollToBottom = () => {
        chatHistory.scrollTo({ top: chatHistory.scrollHeight, behavior: "smooth" });
    };

    // ── Pipeline Tracker ──────────────────────────────────────────────────
    const STEP_ICONS = {
        1: "fa-magnifying-glass",   // Embedding
        2: "fa-database",           // Searching
        3: "fa-ranking-star",       // Re-ranking
        4: "fa-robot",              // Generating
    };

    const createPipelineTracker = () => {
        const tracker = document.createElement("div");
        tracker.className = "pipeline-tracker";
        tracker.innerHTML = `
            <div class="pipeline-header">
                <i class="fas fa-cogs"></i>
                <span>Processing Pipeline</span>
            </div>
            <div class="pipeline-steps"></div>
        `;
        return tracker;
    };

    const updatePipelineStep = (tracker, data) => {
        const stepsContainer = tracker.querySelector(".pipeline-steps");
        const stepId = `step-${data.step}`;
        let stepEl = stepsContainer.querySelector(`#${stepId}`);

        if (!stepEl) {
            stepEl = document.createElement("div");
            stepEl.id = stepId;
            stepEl.className = "pipeline-step pending";
            stepsContainer.appendChild(stepEl);
        }

        const iconClass = STEP_ICONS[data.step] || "fa-cog";
        const stateIcon = data.state === "active"
            ? '<i class="fas fa-spinner fa-spin step-state-icon"></i>'
            : data.state === "done"
                ? '<i class="fas fa-check step-state-icon"></i>'
                : '<i class="fas fa-circle step-state-icon"></i>';

        const timeStr = data.time ? `<span class="step-time">${data.time}</span>` : '';

        stepEl.className = `pipeline-step ${data.state}`;
        stepEl.innerHTML = `
            <span class="step-icon"><i class="fas ${iconClass}"></i></span>
            <span class="step-label">${data.label}</span>
            <span class="step-status">${stateIcon}${timeStr}</span>
        `;
    };

    const collapsePipelineTracker = (tracker) => {
        // Convert the tracker into a compact <details> element after completion
        const steps = tracker.querySelectorAll(".pipeline-step");
        let summaryParts = [];
        steps.forEach(s => {
            const time = s.querySelector(".step-time");
            if (time) summaryParts.push(time.textContent);
        });

        const details = document.createElement("details");
        details.className = "pipeline-summary";
        const summary = document.createElement("summary");
        summary.innerHTML = `<i class="fas fa-cogs"></i> Pipeline completed (${summaryParts.join(" + ")})`;
        details.appendChild(summary);

        // Move steps into details
        const stepsClone = tracker.querySelector(".pipeline-steps").cloneNode(true);
        details.appendChild(stepsClone);

        tracker.replaceWith(details);
        return details;
    };

    // ── Message Element Creation ──────────────────────────────────────────
    const createMessageElement = (role, content = "") => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const contentContainer = document.createElement("div");
        contentContainer.className = "message-container";

        // Pipeline tracker (only for assistant messages, populated later)
        const pipelineSlot = document.createElement("div");
        pipelineSlot.className = "pipeline-slot";

        const thinkDiv = document.createElement("details");
        thinkDiv.className = "thought-process hidden";
        const thinkSummary = document.createElement("summary");
        thinkSummary.innerHTML = '<i class="fas fa-brain"></i> Thinking Process';
        const thinkContent = document.createElement("div");
        thinkContent.className = "thought-content";
        thinkDiv.appendChild(thinkSummary);
        thinkDiv.appendChild(thinkContent);

        const contentDiv = document.createElement("div");
        contentDiv.className = "message-content";
        contentDiv.innerHTML = content;

        contentContainer.appendChild(pipelineSlot);
        contentContainer.appendChild(thinkDiv);
        contentContainer.appendChild(contentDiv);

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentContainer);

        return { msgDiv, contentDiv, thinkContent, thinkDiv, pipelineSlot };
    };

    // ── PDF Upload handling ───────────────────────────────────────────────
    pdfUpload.addEventListener("change", async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        uploadStatus.textContent = `Uploading ${file.name}...`;
        uploadStatus.classList.remove("hidden");
        uploadStatus.style.color = "var(--text-primary)";

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/api/upload", { method: "POST", body: formData });
            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n\n");

                for (let line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.replace("data: ", "").trim();
                        try {
                            const data = JSON.parse(dataStr);
                            if (data.status) {
                                uploadStatus.textContent = data.status;
                                if (data.done) uploadStatus.style.color = "#10b981"; // success green
                            }
                        } catch (e) { console.error("Stream parse error", e); }
                    } else if (line.startsWith("event: error")) {
                        uploadStatus.textContent = "Upload failed.";
                        uploadStatus.style.color = "#ef4444"; // error red
                    }
                }
            }
        } catch (err) {
            uploadStatus.textContent = "Error connecting to server.";
            uploadStatus.style.color = "#ef4444";
        }
    });

    // ── Chat handling ─────────────────────────────────────────────────────
    const sendMessage = async () => {
        const query = chatInput.value.trim();
        if (!query) return;

        // Add user message & save immediately
        const { msgDiv: userMsg } = createMessageElement('user', query);
        chatHistory.appendChild(userMsg);
        saveChat();

        // Reset input
        chatInput.value = "";
        chatInput.style.height = "auto";
        sendBtn.disabled = true;
        scrollToBottom();

        // Create Assistant placeholder with pipeline tracker
        const {
            msgDiv: astMsg,
            contentDiv: astContent,
            thinkContent: astThink,
            thinkDiv: astThinkContainer,
            pipelineSlot
        } = createMessageElement('assistant');

        // Add pipeline tracker
        const pipelineTracker = createPipelineTracker();
        pipelineSlot.appendChild(pipelineTracker);

        // Set typing indicator
        astContent.innerHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
        chatHistory.appendChild(astMsg);
        scrollToBottom();

        // Connect to SSE for streaming
        try {
            let markdownAccumulator = "";
            let thinkAccumulator = "";
            let clearedLoader = false;

            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder("utf-8");

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const rawChunk = decoder.decode(value, { stream: true });
                const blocks = rawChunk.split("\n\n");

                for (const block of blocks) {
                    if (!block.trim()) continue;

                    // Parse SSE block: extract event type + data
                    const blockLines = block.split("\n");
                    let eventType = "";
                    let dataStr = "";

                    for (const bLine of blockLines) {
                        if (bLine.startsWith("event: ")) {
                            eventType = bLine.slice(7).trim();
                        } else if (bLine.startsWith("data: ")) {
                            dataStr = bLine.slice(6).trim();
                        }
                    }

                    // ── Handle status events (pipeline tracker) ──
                    if (eventType === "status") {
                        try {
                            const statusData = JSON.parse(dataStr);
                            updatePipelineStep(pipelineTracker, statusData);
                            scrollToBottom();
                        } catch (e) { console.error("Status parse error", e); }
                        continue;
                    }

                    // ── Handle sources event ──
                    if (eventType === "sources") {
                        continue; // Sources are handled implicitly by citations in the response
                    }

                    // ── Handle error event ──
                    if (eventType === "error") {
                        try {
                            const errData = JSON.parse(dataStr);
                            if (!clearedLoader) { astContent.innerHTML = ""; clearedLoader = true; }
                            astContent.innerHTML += `<p style="color: #ef4444;">Error: ${errData.error}</p>`;
                        } catch (e) {}
                        continue;
                    }

                    // ── Handle data events (tokens) ──
                    if (!dataStr) continue;

                    if (dataStr === "[DONE]") {
                        // Collapse the pipeline tracker into a compact summary
                        collapsePipelineTracker(pipelineTracker);
                        saveChat();
                        break;
                    }

                    try {
                        const data = JSON.parse(dataStr);
                        if (data.think) {
                            if (!clearedLoader) { astContent.innerHTML = ""; clearedLoader = true; }
                            astThinkContainer.classList.remove("hidden");
                            astThinkContainer.setAttribute("open", "");  // Auto-expand so users see thinking live
                            thinkAccumulator += data.think;
                            astThink.innerText = thinkAccumulator;
                            scrollToBottom();
                        }
                        else if (data.text) {
                            if (!clearedLoader) { astContent.innerHTML = ""; clearedLoader = true; }
                            // Collapse thinking once the real answer starts
                            if (astThinkContainer.hasAttribute("open")) {
                                astThinkContainer.removeAttribute("open");
                            }
                            markdownAccumulator += data.text;
                            const renderedHtml = marked.parse(markdownAccumulator);
                            astContent.innerHTML = formatCitations(renderedHtml);
                            scrollToBottom();
                        }
                        else if (data.error) {
                            if (!clearedLoader) astContent.innerHTML = "";
                            astContent.innerHTML += `<p style="color: #ef4444;">Error: ${data.error}</p>`;
                        }
                    } catch (e) {
                        console.error("Error parsing JSON Stream", e);
                    }
                }
            }
        } catch (err) {
            astContent.innerHTML = `<p style="color: #ef4444;">Connection failed.</p>`;
            saveChat();
        }
    };

    sendBtn.addEventListener("click", sendMessage);
});
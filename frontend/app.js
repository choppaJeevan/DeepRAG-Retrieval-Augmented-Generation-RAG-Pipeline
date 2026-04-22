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
    const saveChat = () => {
        // Do not save the typing indicator
        const htmlToSave = chatHistory.innerHTML.replace(/<div class="typing-indicator">.*?<\/div>/g, '');
        localStorage.setItem("ragChatHistory", htmlToSave);
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
        const savedHTML = localStorage.getItem("ragChatHistory");
        if (savedHTML && savedHTML.trim() !== "") {
            chatHistory.innerHTML = savedHTML;
            scrollToBottom();
        } else {
            chatHistory.innerHTML = welcomeMessageHTML;
        }
    };

    // Load history on startup
    loadChat();

    // New Chat Button - Clear state and reload for clean reset
    newChatBtn.addEventListener("click", () => {
        localStorage.removeItem("ragChatHistory");
        window.location.reload();
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

    const createMessageElement = (role, content = "") => {
        const msgDiv = document.createElement("div");
        msgDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;

        const avatar = document.createElement("div");
        avatar.className = "avatar";
        avatar.innerHTML = role === 'user' ? '<i class="fas fa-user"></i>' : '<i class="fas fa-robot"></i>';

        const contentContainer = document.createElement("div");
        contentContainer.className = "message-container";

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

        contentContainer.appendChild(thinkDiv);
        contentContainer.appendChild(contentDiv);

        msgDiv.appendChild(avatar);
        msgDiv.appendChild(contentContainer);

        return { msgDiv, contentDiv, thinkContent, thinkDiv };
    };

    // PDF Upload handling
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

    // Chat handling
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

        // Create Assistant placeholder
        const { msgDiv: astMsg, contentDiv: astContent, thinkContent: astThink, thinkDiv: astThinkContainer } = createMessageElement('assistant');
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

                const chunk = decoder.decode(value, { stream: true });
                const lines = chunk.split("\n\n");

                for (const line of lines) {
                    if (line.startsWith("data: ")) {
                        const dataStr = line.replace("data: ", "").trim();

                        if (dataStr === "[DONE]") {
                            saveChat(); // Save when stream finishes
                            break;
                        }

                        try {
                            const data = JSON.parse(dataStr);
                            if (data.think) {
                                if (!clearedLoader) { astContent.innerHTML = ""; clearedLoader = true; }
                                astThinkContainer.classList.remove("hidden");
                                thinkAccumulator += data.think;
                                astThink.innerText = thinkAccumulator;
                                scrollToBottom();
                            }
                            else if (data.text) {
                                if (!clearedLoader) { astContent.innerHTML = ""; clearedLoader = true; }
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
            }
        } catch (err) {
            astContent.innerHTML = `<p style="color: #ef4444;">Connection failed.</p>`;
            saveChat();
        }
    };

    sendBtn.addEventListener("click", sendMessage);
});
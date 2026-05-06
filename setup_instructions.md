# Setup & Execution Instructions

This guide provides detailed, step-by-step instructions on how to set up and run this NLP RAG project on your local machine.

## Prerequisites
- **Python 3.10+** installed on your system.
- **Docker Desktop** installed (Mandatory for running the Weaviate vector database).

---

## Option A: Running Locally via Python (Standard Method)

### Step 1: Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.
Open your terminal inside the `NLP_project` folder and run:

```bash
python -m venv .venv
```

Activate the virtual environment:
- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```

- **Mac/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Step 2: Install the Requirements
With your virtual environment activated, install all the required Python packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 3: Install Ollama and Download Models
This project uses local AI models to embed text and generate answers. You must download them before running the app.

1. Download and install **Ollama** from https://ollama.com/
2. Open a new terminal and run the following commands:

```bash
ollama pull nomic-embed-text
ollama pull deepseek-r1:8b
```
*(Note: These models are several gigabytes in size. Please wait for the downloads to finish).*

### Step 4: Configure Your API Keys
This project requires API keys for both LlamaCloud (for parsing documents) and Ngrok (for creating a tunnel).

1. Go to https://cloud.llamaindex.ai/ to generate your LlamaParse API Key.
2. Go to https://dashboard.ngrok.com/get-started/your-authtoken to get your Ngrok Auth Token.
3. Open the existing `.env` file included in this project.
4. Paste your API keys into the file. It should look like this:

```env
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
NGROK_AUTHTOKEN=your_ngrok_auth_token_here
```

### Step 5: Start the Vector Database
Since this project uses a local Weaviate database to store the chunks, you must start it using Docker.
Ensure Docker Desktop is open, then open your terminal in the project folder and run:

```bash
docker compose up -d weaviate
```

### Step 6: Start the Application
You can now run the project! 

Simply run the following command to launch the full interactive Web UI:

```bash
python api.py
```

*(This will automatically open the application in your local web browser. If you provided an ngrok token, it will also print a public link in the terminal).*

*(Optional) To run the RAG evaluation metrics instead of the Web UI, run:*
```bash
python rag_evaluate.py
```

---

## Option B: Running via Docker (Easiest Method)

If you have Docker installed, you can skip installing Python packages locally and run everything inside a container.

### Step 1: Configure Your API Keys
Just like in Option A, open the existing `.env` file in the project folder and paste your API keys into it.

### Step 2: Build and Run with Docker Compose
Ensure Docker Desktop is open and running on your machine. Open your terminal in the project folder and execute:

```bash
docker compose up --build
```

Docker will handle all the installations and start the application automatically.

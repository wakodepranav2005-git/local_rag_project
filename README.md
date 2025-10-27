# Local RAG AI Assistant 🤖

This project is a complete, 100% local Retrieval-Augmented Generation (RAG) assistant.

It allows you to chat with your own documents without sending any data to external APIs. The assistant loads your text files, indexes them in a local vector database, and uses a local Large Language Model (LLM) to answer questions based *only* on the information found in your documents.

## Core Technologies

* **LLM (Text Generation):** `Qwen/Qwen2.5-3B-Instruct` (running locally via `transformers`)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (running locally)
* **Vector Database:** `ChromaDB` (stores embeddings locally in a `chroma_db` folder)
* **Orchestration:** `LangChain` (manages the RAG pipeline)
* **Hardware:** Runs on commodity hardware (NVIDIA GPU or Apple Silicon) using 4-bit quantization.

## 🚀 Setup Instructions (Ubuntu)

This guide assumes you have an NVIDIA GPU and are using Ubuntu.

### 1. Clone the Repository

```bash
git clone https://github.com/wakodepranav2005-git/local_rag_project.git
cd local_rag_project
```

### 2. Create and Activate Python 3.11 Environment
This project requires Python 3.11 to work correctly with all AI packages.

```Bash
# Ensure you have Python 3.11 and its 'venv' module
sudo apt update
sudo apt install python3.11 python3.11-venv

# Create the virtual environment
python3.11 -m venv venv

# Activate the environment
source venv/bin/activate
(Your terminal prompt should now start with (venv))
```

### 3. Install Dependencies

```Bash
# Install all required packages
pip install -r requirements.txt
```
(This will take a few minutes as it installs PyTorch and other large libraries.)

## 🏁 How to Run

### 1. Add Your Documents
Place all your knowledge files (as .txt files) into the data/ folder.

### 2. Run the Application
With your virtual environment active, start the assistant:

```Bash

python3 src/app.py
```

### 3. First-Time Run

The first time you run the app, it will:
- **Download the models** (Qwen and all-MiniLM-L6-v2). This is a one-time download and may take several gigabytes of space.
- **Process your documents:** It will read all files in data/, split them into chunks, and create embeddings.
- **Create the chroma_db folder:** This is where it saves the embeddings.

### 4. Ask Questions!
Once the setup is complete, you will see a prompt. You can now ask questions about your documents.

```Bash
...
Successfully added 21 chunks to the vector database.

Enter a question or 'quit' to exit: What is AI?
```
Subsequent runs will be much faster, as the app will re-use the models and the existing chroma_db database.
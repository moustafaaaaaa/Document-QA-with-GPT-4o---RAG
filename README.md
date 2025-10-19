# 🤖 Document Q&A with GPT-4o — Retrieval-Augmented Generation (RAG)

This project demonstrates how to build a **Document Question Answering (QA)** system using **GPT-4o** and **LangChain** with **Retrieval-Augmented Generation (RAG)** architecture.

It enables users to upload any **PDF document** 📄 and ask questions 💬 about its content.  
The model retrieves the most relevant chunks from the document and generates accurate, context-aware answers using **GPT-4o**.

---

## 🚀 Key Features
✅ Supports **PDF document ingestion**  
✅ **Semantic text splitting** and embedding  
✅ **FAISS** vector database for efficient similarity search  
✅ Uses **HuggingFace Embeddings** for document representation  
✅ Leverages **OpenAI GPT-4o** for high-quality natural language answers  
✅ Simple, modular, and extendable design  

---

## 🧠 Project Overview

This RAG pipeline integrates **retrieval** and **generation**:

1. **Document Loading** — The document is loaded using `UnstructuredFileLoader` (handles PDF, text, and other formats).
2. **Chunking** — The text is split into manageable chunks with overlap to preserve context.
3. **Embedding** — Each chunk is embedded using `HuggingFaceEmbeddings` and stored in **FAISS**.
4. **Retrieval** — Given a user query, the most similar document chunks are retrieved.
5. **Generation** — The retriever’s results are passed to `GPT-4o` (via LangChain’s `ChatOpenAI`), which formulates the final answer.

---

## 🧩 Tech Stack

| Component | Description |
|------------|--------------|
| 🧠 **GPT-4o** | Large Language Model from OpenAI for text generation |
| 🧾 **LangChain** | Framework for managing LLM pipelines and retrieval chains |
| 💬 **LangChain-Community / OpenAI / HuggingFace** | Modular integrations for embeddings and LLMs |
| 🔍 **FAISS** | Vector similarity search engine |
| 🧩 **Unstructured** | Document loader for parsing text & PDFs |
| 🧠 **HuggingFace Embeddings** | Generates dense vector representations of document text |

---

## ⚙️ Project Structure

document-qa-rag/
│
├── Document_QA_with_GPT4o_RAG.ipynb # Main Jupyter Notebook
├── attention_is_all_you_need.pdf # Example document
├── requirements.txt # Dependencies list
└── README.md # Project documentation

makefile

---

## 🧪 Example Workflow

```python
# Load and split document
loader = UnstructuredFileLoader("attention_is_all_you_need.pdf")
documents = loader.load()

# Split into chunks
text_splitter = CharacterTextSplitter(separator='/n', chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_documents(documents)

# Embed and store in FAISS
embeddings = HuggingFaceEmbeddings()
knowledge_base = FAISS.from_documents(text_chunks, embeddings)

# Build retrieval + generation chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o", temperature=0),
    retriever=knowledge_base.as_retriever()
)

# Ask questions
response = qa_chain.invoke({"query": "What is this document about?"})
print(response["result"])
```
🧰 Installation & Setup
1️⃣ Clone the repository

git clone https://github.com/<moustafaaaaaa>/document-qa-rag.git
cd document-qa-rag
2️⃣ Install dependencies

pip install -r requirements.txt
or manually:
pip install transformers sentence-transformers langchain langchain-community langchain-openai faiss-cpu unstructured unstructured[pdf]
3️⃣ Set your OpenAI API key
Before running, export your API key:

Windows (PowerShell):


setx OPENAI_API_KEY "your_openai_api_key"
macOS / Linux:

export OPENAI_API_KEY="your_openai_api_key"
4️⃣ Run the notebook
Open and run:
jupyter notebook "Document QA with GPT-4o - RAG.ipynb"

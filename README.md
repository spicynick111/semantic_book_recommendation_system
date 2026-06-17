# 📚 Semantic Book Recommendation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/%F0%9F%A6%9C%EF%B8%8F-LangChain-brightgreen)](https://github.com/langchain-ai/langchain)
[![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-red)](https://github.com/chroma-core/chroma)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Spaces-Deployed-yellow)](https://huggingface.co/spaces)

An enterprise-ready, intelligent book recommendation system that leverages Large Language Models (LLMs), deep semantic search, and a persistent vector database to deliver highly context-aware book suggestions based on user intent, themes, and emotional resonance rather than just raw keyword or popularity matching.

---

## 🚀 Key Features

* **True Semantic Search:** Moves beyond classic TF-IDF/BM25 keyword search to understand the deep contextual meaning of user queries using dense vector embeddings.
* **Flexible Dual-Input System:** Users can input either a 13-digit standard **ISBN** for direct lookups or free-form text descriptions (e.g., *"a gripping historical fiction set in ancient Egypt"*).
* **Persistent Vector Knowledge Base:** Powered by **ChromaDB**, allowing sub-millisecond similarity queries and persistent on-disk index reusability without cold-start re-embedding.
* **Rich UI Frontend:** An interactive, accessible web interface built with **Gradio** featuring smooth visual layouts and dynamic book cover displays.
* **Cloud Native:** Native structural support for headless deployment on **Hugging Face Spaces**.

---

## 🛠️ System Architecture

The pipeline consists of three main stages: Document Chunking, Embedding Generation, and Fast Vector Similarity Retrieval.

```mermaid
graph TD
    A[books_with_emotions.csv] --> B[CharacterTextSplitter Chunking]
    B --> C[all-MiniLM-L6-v2 Transformer]
    C --> D[Dense Numerical Embeddings]
    D --> E[Persistent ChromaDB Store]
    F[User Query: Text / ISBN] --> G[all-MiniLM-L6-v2 Transformer]
    G --> H[Query Vector]
    H --> I[ChromaDB Vector Matching]
    I --> J[Top Similar Book Matches]
    J --> K[Gradio UI Gallery Output]

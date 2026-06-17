# 📚 Semantic Book Recommendation System

An AI-powered book recommendation engine that leverages semantic search, transformer embeddings, sentiment analysis, and vector similarity retrieval to deliver context-aware book recommendations beyond traditional keyword matching.

## 🚀 Project Overview

Traditional recommendation systems rely heavily on keyword matching or collaborative filtering, which often fail to capture the true meaning behind user queries.

This project addresses that limitation by using Natural Language Processing (NLP) and transformer-based embeddings to understand the semantic context of books and user preferences.

Users can search using natural language queries such as:

> "Books similar to Harry Potter with dark magic themes"

and receive recommendations based on meaning rather than exact keyword matches.

## ✨ Key Features

* Semantic search using Sentence Transformers
* Vector similarity retrieval with FAISS
* Book sentiment analysis
* Genre-aware recommendations
* Interactive Gradio dashboard
* Real-time recommendation generation
* Content-based recommendation engine
* Scalable embedding pipeline

## 🏗️ System Architecture

User Query
↓
Sentence Transformer Embedding
↓
FAISS Vector Database
↓
Similarity Search
↓
Ranking & Filtering
↓
Recommended Books

## 🛠️ Tech Stack

### Programming

* Python

### Machine Learning & NLP

* Sentence Transformers
* Hugging Face Transformers
* Scikit-Learn
* Pandas
* NumPy

### Vector Search

* FAISS

### Visualization & UI

* Gradio
* Matplotlib
* Seaborn

### Development Tools

* Jupyter Notebook
* Git
* GitHub

## 📂 Project Structure

semantic_book_recommendation_system/
│
├── data/
├── notebooks/
├── src/
├── models/
├── app.py
├── requirements.txt
└── README.md

## 📊 Dataset

The project utilizes a curated collection of books containing:

* Title
* Author
* Genre
* Description
* Rating
* Reviews

Text descriptions are transformed into dense vector embeddings to enable semantic retrieval.

## 🔬 Methodology

### Data Preprocessing

* Missing value handling
* Text cleaning
* Feature extraction

### Embedding Generation

* Sentence Transformer encoding
* Dense vector creation

### Similarity Retrieval

* FAISS indexing
* Cosine similarity search

### Recommendation Pipeline

* Query embedding generation
* Top-K retrieval
* Ranking and recommendation

## 📈 Results

* Improved recommendation relevance compared to keyword-based search
* Faster retrieval using vector indexing
* Better handling of vocabulary mismatch problems
* Context-aware recommendations

## 💡 Example Query

Input:
"Fantasy books with magical schools and adventure"

Output:

* Harry Potter Series
* The Name of the Wind
* Percy Jackson
* Eragon
* The Magicians

## 🔮 Future Enhancements

* Hybrid recommendation system
* User personalization
* LLM-based recommendation explanations
* Multi-language support
* Cloud deployment

## 👨‍💻 Author

Aryan Kumar

B.Tech CSE
Birla Institute of Technology Mesra

GitHub: https://github.com/spicynick111

## ⭐ If you found this project useful, consider giving it a star.

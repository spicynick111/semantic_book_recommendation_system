📚 Semantic Book Recommendation System
✨ Overview
This project implements an intelligent Book Recommendation System that leverages semantic search to provide highly relevant book suggestions. Unlike traditional systems that rely solely on explicit ratings or purchase history, this system deeply understands the meaning and context within book descriptions. This allows for intuitive discovery of new reads based on content, not just popularity. It's deployed as a user-friendly web application using Gradio on Hugging Face Spaces.

🚀 Features
🔍 Semantic Search: Recommends books by understanding the underlying meaning of their descriptions, enabling more intuitive and context-aware suggestions.

✍️ Flexible Input: Users can input either a 13-digit ISBN for a specific book or a free-form text description (e.g., "a gripping historical fiction set in ancient Egypt").

🖼️ Image-Based Recommendations: Displays vibrant book cover images (if available) alongside recommendations for a richer visual experience.

⚡ Efficient Knowledge Base: Utilizes a persisted vector database (ChromaDB) for fast and scalable retrieval of book embeddings, ensuring quick responses.

🌐 User-Friendly Interface: Built with Gradio, providing a simple, interactive, and accessible web application.

☁️ Cloud Deployment: Hosted on Hugging Face Spaces, making it publicly accessible and easy to share with anyone, anywhere.

💡 How It Works (High-Level Flow)
The system operates through a series of intelligent steps:

📚 Data Preparation: Book descriptions from a CSV dataset are meticulously cleaned and transformed into structured documents, ready for processing.

🧠 Embedding Generation: Each cleaned book description is converted into a high-dimensional numerical vector (an "embedding"). This is achieved using a BERT-based Sentence Transformer model (all-MiniLM-L6-v2), which excels at capturing the semantic meaning of the text.

🗄️ Vector Database (ChromaDB): These powerful embeddings are then efficiently stored in a local, persistent vector database, ChromaDB. This setup allows for lightning-fast similarity searches.

❓ User Query: When a user provides an ISBN or a text description, this input is also transformed into an embedding using the same model.

🔎 Semantic Similarity Search: The user's query embedding is then used to perform a rapid similarity search within ChromaDB, identifying the most semantically similar book embeddings.

✨ Recommendation Display: Finally, the system retrieves the full details (title, author, image) of the top similar books and presents them in a clear, visual format through the Gradio web interface.

🛠️ Technologies Used
Python: The robust core programming language powering the entire system.

Pandas: Essential for efficient data manipulation and cleaning of the large book dataset.

LangChain: A powerful framework for building applications with language models, providing key components:

CharacterTextSplitter: For intelligent text chunking, optimizing embedding generation.

SentenceTransformerEmbeddings: For generating high-quality, context-aware text embeddings using all-MiniLM-L6-v2.

Chroma: Our reliable, persistent vector database for storing and querying embeddings.

Gradio: The intuitive framework used to create the interactive and user-friendly web interface.

Hugging Face Spaces: The versatile cloud platform chosen for seamless deployment and hosting of the live application.

📸 Screenshots / Demo GIF
(Replace these placeholders with actual images or a GIF of your running application to make this section truly shine!)

Application Interface

A visual representation of the main application interface, showcasing the input field and recommendation gallery.

Example Recommendation

An example of books recommended based on a user query, highlighting the relevance of semantic search.

⚙️ Setup and Run Locally
To set up and run this project on your local machine:

Clone the Repository:

git clone https://github.com/spicynick111/Book-Recommendation-System-Semantic-Search.git
cd Book-Recommendation-System-Semantic-Search

(Note: If you manually uploaded files and didn't include my_book_recommender_db on GitHub, the database will be built automatically on the first run, which may take some time.)

Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

(Ensure requirements.txt contains all necessary libraries like pandas, gradio, langchain-community, langchain-text-splitters, sentence-transformers, chromadb.)

Prepare Data:

Ensure books_with_emotions.csv is in the root directory of your project.

Run the Application:

python app.py

The application will launch locally, and you can access it via the URL provided in your terminal (usually http://127.0.0.1:7860).

☁️ Deployment
This project is seamlessly deployed on Hugging Face Spaces for easy accessibility and sharing.

Deployment Steps:

Create a new Hugging Face Space (select Gradio as the SDK).

Upload app.py, books_with_emotions.csv, and requirements.txt to the Space's file section.

Upload the my_book_recommender_db folder (containing the persisted ChromaDB) to the root of the Space. (Alternatively, the DB will be built automatically on the first run if not uploaded, but this takes longer).

The Space will automatically build and deploy the application, making it live for the world to see!

🤝 Contributing
Contributions are highly welcome! If you have suggestions for improvements, new features, or bug fixes, please feel free to open an issue or submit a pull request. Let's make this project even better together!

📄 License
This project is licensed under the MIT License.

Developed by: Aryan

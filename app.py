# app.py content for Hugging Face Spaces deployment - ORIGINAL UI WITH CACHE FIX

# 1. IMPORTS
import gradio as gr
import pandas as pd
import os
import torch
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

# 2. GLOBAL VARIABLES / PATHS
DATA_DIR = "." # Assuming data files are in the same directory as app.py

BOOKS_CSV_PATH = os.path.join(DATA_DIR, "books_with_emotions.csv")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "my_book_recommender_db") # Corrected path for Hugging Face Spaces

# 3. LOAD DATA & INITIALIZE MODELS (Run once when the app starts)
print("--- Initializing application components ---")

# Load the books DataFrame
books = None
try:
    books = pd.read_csv(BOOKS_CSV_PATH)
    print(f"Loaded books DataFrame with {len(books)} entries from {BOOKS_CSV_PATH}.")

    # Clean 'isbn13' column
    print("--- Cleaning 'isbn13' column ---")
    books['isbn13_str'] = books['isbn13'].astype(str)
    initial_rows = len(books)
    books['isbn13'] = pd.to_numeric(books['isbn13_str'], errors='coerce')
    books.dropna(subset=['isbn13'], inplace=True)
    books['isbn13'] = books['isbn13'].astype(int)
    rows_removed = initial_rows - len(books)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with invalid ISBNs from DataFrame during startup.")
    print(f"Final books DataFrame size after ISBN cleaning: {len(books)}")
    print("--- 'isbn13' column cleaned. ---")

except FileNotFoundError:
    print(f"Error: {BOOKS_CSV_PATH} not found. Ensure data files are uploaded to Hugging Face Space.")
    exit()
except Exception as e:
    print(f"Error loading books DataFrame: {e}")
    exit()

# Prepare Documents for ChromaDB
documents = []
print("--- Preparing Documents for ChromaDB ---")
try:
    for index, row in books.iterrows():
        isbn_val = str(row.get('isbn13'))
        desc = str(row.get('description', '')).strip()

        generic_phrases = ["reproduction of the original artefact", "no description available for this book"]

        if isbn_val and desc and not any(phrase in desc.lower() for phrase in generic_phrases):
            doc_content = f"{isbn_val}: {desc}"
            documents.append(Document(page_content=doc_content, metadata={"source": "book_description", "isbn13": isbn_val}))

    print(f"Generated {len(documents)} initial documents from DataFrame.")

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)

    split_documents = []
    for doc in documents:
        split_documents.extend(text_splitter.split_documents([doc]))

    documents = split_documents
    print(f"Split into {len(documents)} final processed documents for ChromaDB.")

except Exception as e:
    print(f"Error preparing documents for ChromaDB: {e}")
    import traceback
    traceback.print_exc()
    documents = []

# Initialize Embeddings
print("--- Initializing Embeddings ---")
embeddings = None
try:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings model loaded.")
except Exception as e:
    print(f"Error loading SentenceTransformer embeddings: {e}")

# Initialize ChromaDB
db_books = None
print("--- Initializing ChromaDB ---")
if embeddings and documents:
    try:
        if os.path.exists(CHROMA_DB_PATH) and len(os.listdir(CHROMA_DB_PATH)) > 0:
            db_books = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)
            print(f"Loaded existing ChromaDB from {CHROMA_DB_PATH}.")
        else:
            print(f"Creating new ChromaDB at {CHROMA_DB_PATH}...")
            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            db_books = Chroma.from_documents(documents, embeddings, persist_directory=CHROMA_DB_PATH)
            db_books.persist()
            print(f"New ChromaDB created and persisted at {CHROMA_DB_PATH}.")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        db_books = None
else:
    print("Skipping ChromaDB initialization due to missing embeddings or documents.")

print("--- Application components initialized ---")


# 4. RECOMMENDATION LOGIC FUNCTION
# ORIGINAL: Only book_input parameter
def recommend_books(book_input: str):
    print(f"Received input: '{book_input}'")

    query_text = ""
    book_title = ""

    if isinstance(book_input, str):
        try:
            isbn = int(book_input)
            print(f"Input treated as ISBN: {isbn}")

            book_data = books[books['isbn13'] == isbn]

            if book_data.empty:
                return [], "Book not found by ISBN. Please try another ISBN or a description."

            query_text = book_data.iloc[0]['description']
            book_title = book_data.iloc[0]['title']
            print(f"Found book '{book_title}' with description for ISBN: {query_text[:100]}...")
        except ValueError:
            query_text = book_input
            book_title = f"Query: '{book_input}'"
            print(f"Input treated as description: {query_text[:100]}...")
    else:
        return [], "Invalid input type. Please enter text or ISBN."

    if not query_text:
        return [], "No valid query text provided."

    if not db_books:
        return [], "Recommendation service not ready. ChromaDB could not be initialized. Check logs for details."

    try:
        # ORIGINAL: k is fixed to 5
        results = db_books.similarity_search_with_score(query_text, k=5)
        print(f"ChromaDB search returned {len(results)} results.")

        recommended_items = []
        debug_info = []

        for doc, score in results:
            try:
                doc_isbn_str = doc.page_content.split(":")[0].strip()
                doc_isbn = int(doc_isbn_str)
            except (ValueError, IndexError):
                debug_info.append(f"Warning: Could not parse ISBN from document content: {doc.page_content[:50]}...")
                continue

            recommended_book = books[books['isbn13'] == doc_isbn]

            if not recommended_book.empty:
                title = recommended_book.iloc[0]['title']
                authors = recommended_book.iloc[0]['authors']
                # description = recommended_book.iloc[0]['description'] # Not used here, but available

                image_url = recommended_book.iloc[0].get('thumbnail', '')

                caption = f"Title: {title}\nAuthors: {authors}\nSimilarity Score: {score:.4f}"

                if image_url:
                    recommended_items.append((image_url, caption))
                    debug_info.append(f"Added: {title} (Score: {score:.4f}, Image: {image_url[:50]}...)")
                else:
                    debug_info.append(f"Added (No Image): {title} (Score: {score:.4f}) - No thumbnail URL found.")

            else:
                debug_info.append(f"Warning: Book details for ISBN {doc_isbn} not found in DataFrame.")

        if recommended_items:
            return recommended_items, f"Recommendations for: {book_title}\n\n" + "\n".join(debug_info)
        else:
            return [], f"No relevant recommendations found for: {book_title}\n\n" + "\n".join(debug_info)

    except Exception as e:
        print(f"Error during recommendation process: {e}")
        import traceback
        traceback.print_exc()
        return [], f"An error occurred: {e}"


# 5. GRADIO INTERFACE (ORIGINAL, SIMPLE GR.INTERFACE)
print("--- Launching Gradio Interface ---")
iface = gr.Interface(
    fn=recommend_books,
    inputs=gr.Textbox(label="Enter Book ISBN (13-digit number) or Description", placeholder="e.g., a sci-fi novel about space exploration or 9780345451052"),
    outputs=[
        gr.Gallery(label="Book Recommendations", columns=3, height=400, preview=True, object_fit="contain"),
        gr.Textbox(label="Status/Debug Info", lines=2)
    ],
    title="Book Recommendation System with Images",
    description="Enter a book's ISBN or a brief description to get image-based recommendations. Note: Images displayed if available.",
    examples=[
        "9780345451052",
        "A tale of magic and adventure in a fantasy world.",
        "A detective story with a surprising twist."
    ],
    allow_flagging="never",
    cache_examples=False # THIS IS THE CRITICAL FIX
)

if __name__ == "__main__":
    iface.launch()

import os
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import ollama

# 1. Initialize ChromaDB (in-memory client) and collection
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="vs_code_summarizer_kb")

# 2. Load the Sentence Transformer embedding model
print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def add_text_to_db(text):
  """Adds raw text to ChromaDB."""
  vector = embedding_model.encode(text).tolist()
  collection.add(documents=[text], embeddings=[vector], ids=["text_doc_1"])


def add_pdf_to_db(pdf_path):
  """Extracts text from a PDF and adds pages to ChromaDB."""
  if not os.path.exists(pdf_path):
    print(f"Error: File '{pdf_path}' not found.")
    return False

  reader = PdfReader(pdf_path)
  print(f"Extracting text from {len(reader.pages)} pages...")
  for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
      vector = embedding_model.encode(text).tolist()
      collection.add(
          documents=[text], embeddings=[vector], ids=[f"pdf_page_{i}"]
      )
  return True


def generate_summary(query="Summarize this content."):
  """Retrieves context from ChromaDB and generates a summary via Ollama."""
  query_vector = embedding_model.encode(query).tolist()

  # Query top relevant chunks
  results = collection.query(
      query_embeddings=[query_vector],
      n_results=min(3, collection.count()),
  )

  if not results["documents"] or not results["documents"][0]:
    return "No content found to summarize."

  retrieved_context = "\n".join(results["documents"][0])

  print("Generating summary with Ollama...")
  response = ollama.chat(
      model="llama3",
      messages=[
          {
              "role": "system",
              "content": (
                  "You are a helpful assistant. Provide a concise summary based"
                  " strictly on the provided context."
              ),
          },
          {"role": "user", "content": f"Context:\n{retrieved_context}"},
      ],
  )
  return response["message"]["content"]


# ==========================================
# VS CODE INTERACTIVE TERMINAL INTERFACE
# ==========================================
if __name__ == "__main__":
  print("\n=== AI Text & PDF Summarizer ===")
  print("1. Summarize Raw Text")
  print("2. Summarize PDF File")

  choice = input("Select an option (1 or 2): ").strip()

  if choice == "1":
    print("\nEnter or paste your text below (press Enter twice when done):")
    user_text = input("Text: ")
    if user_text.strip():
      add_text_to_db(user_text)
      summary = generate_summary()
      print("\n--- Summary Result ---")
      print(summary)
    else:
      print("Error: Text cannot be empty.")

  elif choice == "2":
    pdf_file = input(
        "\nEnter the PDF file path (e.g., sample.pdf): "
    ).strip()
    if add_pdf_to_db(pdf_file):
      summary = generate_summary()
      print("\n--- Summary Result ---")
      print(summary)

  else:
    print("Invalid option selected. Please choose 1 or 2.")
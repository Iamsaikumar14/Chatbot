import sqlite3
import numpy as np
import os
import sqlite_vec
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# We store the database file in the project workspace
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_documents.db")

class RAGService:
    def __init__(self):
        # Using gemini-embedding-001 which is fully supported and tested
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.init_db()

    def get_connection(self):
        """Establish a connection to SQLite database and try to load the sqlite-vec extension."""
        conn = sqlite3.connect(DB_PATH)
        has_vec = False
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            has_vec = True
        except (AttributeError, sqlite3.OperationalError):
            # Fallback when extension loading is disabled in Python standard library (e.g. standard macOS Python)
            pass
        return conn, has_vec

    def init_db(self):
        """Initialize the SQLite tables for documents, chunks, and virtual vector table if supported."""
        conn, has_vec = self.get_connection()
        cursor = conn.cursor()
        
        # Table to track documents
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                image_data TEXT
            )
        """)
        
        # Check if we need to migrate existing table to add image_data column
        cursor.execute("PRAGMA table_info(documents)")
        columns = [row[1] for row in cursor.fetchall()]
        if "image_data" not in columns:
            cursor.execute("ALTER TABLE documents ADD COLUMN image_data TEXT")
        
        # Table to store chunks with embedding vectors
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER,
                chunk_index INTEGER,
                content TEXT,
                embedding BLOB,
                FOREIGN KEY (doc_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        """)

        # Table for virtual vector search if sqlite-vec is supported
        if has_vec:
            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                    chunk_id INTEGER PRIMARY KEY,
                    embedding float[3072]
                )
            """)
            
        # Table to store user thumbs up/down feedback for chunks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rag_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                filename TEXT,
                chunk_content TEXT,
                feedback_value INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
            
        conn.commit()
        conn.close()

    def add_document(self, filename: str, text: str):
        """Split text, embed chunks, and persist them in SQLite and vec_chunks table if supported."""
        if not text.strip():
            return
            
        conn, has_vec = self.get_connection()
        cursor = conn.cursor()
        
        # If document already exists, clean up its old records
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
            if has_vec:
                cursor.execute("""
                    DELETE FROM vec_chunks 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                """, (doc_id,))
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            
        # Insert new document metadata
        cursor.execute("INSERT INTO documents (filename, image_data) VALUES (?, NULL)", (filename,))
        doc_id = cursor.lastrowid
        
        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(text)
        
        if chunks:
            # Generate embeddings for all chunks in batch
            embeddings_list = self.embeddings.embed_documents(chunks)
            
            # Save chunks to SQLite as floats32 blob
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings_list)):
                emb_array = np.array(embedding, dtype=np.float32)
                emb_blob = emb_array.tobytes()
                cursor.execute("""
                    INSERT INTO chunks (doc_id, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, i, chunk_text, emb_blob))
                
                chunk_id = cursor.lastrowid
                
                # If using sqlite-vec, insert vector embedding
                if has_vec:
                    cursor.execute("""
                        INSERT INTO vec_chunks (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, emb_blob))
                
        conn.commit()
        conn.close()

    def add_image(self, filename: str, image_base64: str, content_type: str):
        """Analyze image with Gemini, transcribe tables/text, generate embeddings, and save the base64 source."""
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        # Multimodal prompt to extract detailed contents and tabular data
        message = HumanMessage(
            content=[
                {
                    "type": "text", 
                    "text": (
                        "Analyze this image in detail for a Retrieval-Augmented Generation (RAG) system.\n"
                        "1. Transcribe any text visible in the image exactly.\n"
                        "2. If there are tables present, extract and transcribe them completely and accurately "
                        "into well-formatted Markdown tables.\n"
                        "3. Describe any graphs, charts, diagrams, or visual workflows in detail.\n"
                        "Provide a structured, comprehensive description so it can be searched and retrieved later."
                    )
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{content_type};base64,{image_base64}"}
                }
            ]
        )
        
        response = llm.invoke([message])
        description = response.content
        
        if not description.strip():
            return
            
        conn, has_vec = self.get_connection()
        cursor = conn.cursor()
        
        # If document already exists, clean up
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
            if has_vec:
                cursor.execute("""
                    DELETE FROM vec_chunks 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                """, (doc_id,))
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            
        # Store document with base64 data URL
        image_data_url = f"data:{content_type};base64,{image_base64}"
        cursor.execute("INSERT INTO documents (filename, image_data) VALUES (?, ?)", (filename, image_data_url))
        doc_id = cursor.lastrowid
        
        # Split description text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_text(description)
        
        if chunks:
            # Generate embeddings
            embeddings_list = self.embeddings.embed_documents(chunks)
            
            # Save chunks to SQLite
            for i, (chunk_text, embedding) in enumerate(zip(chunks, embeddings_list)):
                emb_array = np.array(embedding, dtype=np.float32)
                emb_blob = emb_array.tobytes()
                cursor.execute("""
                    INSERT INTO chunks (doc_id, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?)
                """, (doc_id, i, chunk_text, emb_blob))
                
                chunk_id = cursor.lastrowid
                
                # If using sqlite-vec, insert vector embedding
                if has_vec:
                    cursor.execute("""
                        INSERT INTO vec_chunks (chunk_id, embedding)
                        VALUES (?, ?)
                    """, (chunk_id, emb_blob))
                
        conn.commit()
        conn.close()

    def delete_document(self, filename: str):
        """Remove a document and all associated chunks from database."""
        conn, has_vec = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE filename = ?", (filename,))
        row = cursor.fetchone()
        if row:
            doc_id = row[0]
            if has_vec:
                cursor.execute("""
                    DELETE FROM vec_chunks 
                    WHERE chunk_id IN (SELECT id FROM chunks WHERE doc_id = ?)
                """, (doc_id,))
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
        conn.close()

    def list_documents(self):
        """List all indexed documents and their upload times."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT filename, upload_time FROM documents ORDER BY upload_time DESC")
            rows = cursor.fetchall()
            return [{"filename": row[0], "upload_time": row[1]} for row in rows]
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    def add_feedback(self, query: str, filename: str, chunk_content: str, feedback_value: int):
        """Save thumbs up/down user feedback for a given query and chunk."""
        conn, _ = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO rag_feedback (query, filename, chunk_content, feedback_value)
                VALUES (?, ?, ?, ?)
            """, (query.strip(), filename, chunk_content, feedback_value))
            conn.commit()
        except Exception as e:
            print(f"Error logging feedback: {e}")
        finally:
            conn.close()

    def get_chunk_feedback(self, cursor, filename: str, content: str) -> int:
        """Sum the feedback values for a given chunk to compute its weight adjustment."""
        try:
            cursor.execute("""
                SELECT SUM(feedback_value) FROM rag_feedback
                WHERE filename = ? AND chunk_content = ?
            """, (filename, content))
            row = cursor.fetchone()
            return row[0] if row[0] is not None else 0
        except Exception:
            return 0

    def query_documents(self, query: str, limit: int = 5):
        """Retrieve the top matching chunks based on cosine similarity, using sqlite-vec if supported."""
        if not query.strip():
            return []
            
        # Get query embedding
        query_emb = self.embeddings.embed_query(query)
        query_vector = np.array(query_emb, dtype=np.float32)
        
        conn, has_vec = self.get_connection()
        cursor = conn.cursor()
        
        results = []
        
        if has_vec:
            try:
                # Query using sqlite-vec MATCH (L2 distance query)
                # Cosine similarity = 1.0 - (distance^2)/2.0
                cursor.execute("""
                    SELECT chunks.content, documents.filename, documents.image_data, vec_chunks.distance
                    FROM vec_chunks
                    JOIN chunks ON vec_chunks.chunk_id = chunks.id
                    JOIN documents ON chunks.doc_id = documents.id
                    WHERE vec_chunks.embedding MATCH ?
                    ORDER BY distance
                    LIMIT ?
                """, (query_vector.tobytes(), limit * 3)) # Fetch more to allow filtering of bad documents
                rows = cursor.fetchall()
                
                for content, filename, image_data, distance in rows:
                    similarity = 1.0 - (distance * distance) / 2.0
                    feedback_sum = self.get_chunk_feedback(cursor, filename, content)
                    
                    # Flag bad documents/chunks if feedback is extremely negative (<= -3)
                    if feedback_sum <= -3:
                        continue
                        
                    # Reweight retrieval score based on feedback (+- 5% per vote)
                    similarity = float(similarity) + (feedback_sum * 0.05)
                    
                    results.append({
                        "content": content,
                        "filename": filename,
                        "image_data": image_data,
                        "similarity": float(similarity)
                    })
            except sqlite3.OperationalError:
                has_vec = False
                    
        if not has_vec:
            # Fallback to manual in-memory scan (e.g. on macOS without extension load support)
            try:
                cursor.execute("""
                    SELECT chunks.content, chunks.embedding, documents.filename, documents.image_data
                    FROM chunks
                    JOIN documents ON chunks.doc_id = documents.id
                """)
                rows = cursor.fetchall()
                
                for content, emb_blob, filename, image_data in rows:
                    emb_vector = np.frombuffer(emb_blob, dtype=np.float32)
                    
                    # Compute cosine similarity manually
                    dot_product = np.dot(query_vector, emb_vector)
                    norm_q = np.linalg.norm(query_vector)
                    norm_e = np.linalg.norm(emb_vector)
                    
                    similarity = 0.0
                    if norm_q > 0 and norm_e > 0:
                        similarity = dot_product / (norm_q * norm_e)
                        
                    feedback_sum = self.get_chunk_feedback(cursor, filename, content)
                    
                    # Flag bad documents/chunks if feedback is extremely negative (<= -3)
                    if feedback_sum <= -3:
                        continue
                        
                    # Reweight retrieval score based on feedback (+- 5% per vote)
                    similarity = float(similarity) + (feedback_sum * 0.05)
                    
                    results.append({
                        "content": content,
                        "filename": filename,
                        "image_data": image_data,
                        "similarity": float(similarity)
                    })
            except sqlite3.OperationalError:
                pass
                
        conn.close()
        
        # Filter results by similarity score to ensure relevance
        MIN_SIMILARITY_THRESHOLD = 0.60
        results = [res for res in results if res["similarity"] >= MIN_SIMILARITY_THRESHOLD]
        
        # Sort matches by similarity in descending order
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

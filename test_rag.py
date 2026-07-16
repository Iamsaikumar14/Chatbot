import os
import sys
from rag_service import RAGService

def run_test():
    print("Initializing RAG Service...")
    rag = RAGService()
    
    test_filename = "test_document_recipe.txt"
    test_content = """
    The Secret ingredients of Antigravity Coffee:
    1. 2 spoonfuls of celestial stardust.
    2. A splash of zero-gravity whipped cream.
    3. Exactly 3 drops of pure liquid sunshine.
    Mix all ingredients in a cup and stir counter-clockwise to float.
    """
    
    print(f"Adding test document: {test_filename}")
    rag.add_document(test_filename, test_content)
    
    print("Listing documents in RAG system:")
    docs = rag.list_documents()
    found = False
    for doc in docs:
        print(f" - {doc['filename']} (uploaded: {doc['upload_time']})")
        if doc['filename'] == test_filename:
            found = True
            
    if not found:
        print("FAIL: Test document not found in list_documents output!")
        sys.exit(1)
        
    print("Testing query retrieval...")
    query = "What are the secret ingredients of Antigravity Coffee?"
    results = rag.query_documents(query, limit=3)
    
    if not results:
        print("FAIL: No search results returned!")
        sys.exit(1)
        
    print(f"Query: '{query}'")
    print("Top Search Results:")
    for i, res in enumerate(results):
        print(f"  [{i+1}] {res['filename']} (similarity: {res['similarity']:.4f})")
        print(f"      Content snippet: {res['content'].strip()[:100]}...")
        
    # Check if similarity is high for the test file
    best_match = results[0]
    if best_match['filename'] != test_filename:
        print(f"FAIL: Expected top match to be {test_filename}, but got {best_match['filename']}!")
        sys.exit(1)
        
    if best_match['similarity'] < 0.3:
        print(f"FAIL: Similarity score too low! Got {best_match['similarity']:.4f}")
        sys.exit(1)
        
    print("Cleaning up test document...")
    rag.delete_document(test_filename)
    
    docs_after = rag.list_documents()
    for doc in docs_after:
        if doc['filename'] == test_filename:
            print("FAIL: Document was not deleted successfully!")
            sys.exit(1)
            
    # Test multimodal image ingestion
    print("\nTesting multimodal image ingestion...")
    test_image_filename = "test_image.png"
    # Small 1x1 black pixel PNG base64
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    
    print(f"Adding test image document: {test_image_filename}")
    rag.add_image(test_image_filename, test_image_base64, "image/png")
    
    print("Listing documents in RAG system after image upload:")
    docs = rag.list_documents()
    found_image = False
    for doc in docs:
        print(f" - {doc['filename']}")
        if doc['filename'] == test_image_filename:
            found_image = True
            
    if not found_image:
        print("FAIL: Test image document not found in list_documents output!")
        sys.exit(1)
        
    print("Testing image query retrieval...")
    image_query = "Describe the color or pixel"
    image_results = rag.query_documents(image_query, limit=3)
    
    if not image_results:
        print("FAIL: No search results returned for image query!")
        sys.exit(1)
        
    print(f"Query: '{image_query}'")
    print("Top Search Results for Image:")
    image_match = None
    for i, res in enumerate(image_results):
        print(f"  [{i+1}] {res['filename']} (similarity: {res['similarity']:.4f})")
        print(f"      Content snippet: {res['content'].strip()[:100]}...")
        if res['image_data']:
            print(f"      Image base64 data URL starts with: {res['image_data'][:40]}...")
        if res['filename'] == test_image_filename:
            image_match = res
            
    if not image_match:
        print("FAIL: Expected image document to match query results!")
        sys.exit(1)
        
    if not image_match['image_data'] or not image_match['image_data'].startswith("data:image/png;base64,"):
        print(f"FAIL: Expected valid image data url, got: {image_match['image_data']}")
        sys.exit(1)
        
    print("Cleaning up test image...")
    rag.delete_document(test_image_filename)
    
    docs_after_all = rag.list_documents()
    for doc in docs_after_all:
        if doc['filename'] == test_image_filename:
            print("FAIL: Image document was not deleted successfully!")
            sys.exit(1)
    # Test feedback loop reweighting and flagging
    print("\nTesting feedback loop reweighting and flagging...")
    fb_filename = "test_feedback_doc.txt"
    fb_content = "The capital of NIT Rourkela is Sector 1."
    rag.add_document(fb_filename, fb_content)
    
    # Query initially
    initial_results = rag.query_documents("Sector 1", limit=1)
    if not initial_results:
        print("FAIL: Feedback test document not returned on initial search!")
        sys.exit(1)
    initial_score = initial_results[0]["similarity"]
    print(f"Initial similarity score: {initial_score:.4f}")
    
    # Log 1 thumbs down (feedback_value = -1)
    print("Adding 1 negative feedback (thumbs down)...")
    rag.add_feedback("Sector 1", fb_filename, initial_results[0]["content"], -1)
    
    # Query and check score penalty
    penalized_results = rag.query_documents("Sector 1", limit=1)
    if not penalized_results:
        print("FAIL: Document not returned after single penalty!")
        sys.exit(1)
    penalized_score = penalized_results[0]["similarity"]
    print(f"Penalized similarity score: {penalized_score:.4f}")
    if penalized_score >= initial_score:
        print("FAIL: Similarity score did not decrease after negative feedback!")
        sys.exit(1)
        
    # Log 2 more thumbs down (total -3, should trigger exclusion)
    print("Adding 2 more negative feedbacks (total 3 thumbs down)...")
    rag.add_feedback("Sector 1", fb_filename, initial_results[0]["content"], -1)
    rag.add_feedback("Sector 1", fb_filename, initial_results[0]["content"], -1)
    
    # Query and verify exclusion
    excluded_results = rag.query_documents("Sector 1", limit=1)
    if excluded_results and any(r["filename"] == fb_filename for r in excluded_results):
        print("FAIL: Document was not excluded/flagged after 3 thumbs down!")
        sys.exit(1)
    print("Document successfully excluded after 3 thumbs down.")
    
    # Log positive feedback to override
    print("Adding 4 positive feedbacks (thumbs up)...")
    for _ in range(4):
        rag.add_feedback("Sector 1", fb_filename, initial_results[0]["content"], 1)
        
    # Query and verify boost
    boosted_results = rag.query_documents("Sector 1", limit=1)
    if not boosted_results:
        print("FAIL: Document not returned after positive feedback boost override!")
        sys.exit(1)
    boosted_score = boosted_results[0]["similarity"]
    print(f"Boosted similarity score: {boosted_score:.4f}")
    if boosted_score <= initial_score:
        print("FAIL: Similarity score did not increase after positive feedback!")
        sys.exit(1)
        
    # Clean up test feedback records
    conn, _ = rag.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM rag_feedback WHERE filename = ?", (fb_filename,))
    conn.commit()
    conn.close()
    
    rag.delete_document(fb_filename)
    
    print("\nSUCCESS: All RAG service tests passed successfully!")

if __name__ == "__main__":
    run_test()

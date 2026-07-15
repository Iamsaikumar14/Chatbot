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
            
    print("SUCCESS: All RAG service tests passed successfully!")

if __name__ == "__main__":
    run_test()

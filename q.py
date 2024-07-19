import faiss

# Load the FAISS index
index = faiss.read_index('faiss_index\index.faiss')

# Print index details
print("Index dimension:", index.d)
print("Number of vectors:", index.ntotal)

# Check and reconstruct the vectors
if index.ntotal > 0:
    # Reconstruct all available vectors (adjust the number as needed)
    num_vectors_to_reconstruct = min(10, index.ntotal)
    vectors = index.reconstruct_n(0, num_vectors_to_reconstruct)
    print("Sample vectors:", vectors)

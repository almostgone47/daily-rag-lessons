from diskcache import Cache
from helpers import model, load_resume_data
from lesson2_chunking import chunk_section
import hashlib
import numpy as np

# Initialize diskcache for embedding cache
embedding_cache = Cache('./embedding_cache')

# Load resume data and create chunks
resume_data = load_resume_data()

# Generate all chunks from resume
all_chunks = []
for section in resume_data.get("sections", []):
    chunks = chunk_section(section)
    all_chunks.extend(chunks)

def generate_embeddings(chunks, use_cache=True):
	"""
	Generate embeddings for chunks, using cache to avoid re-embedding unchanged chunks.
	"""
	embeddings = []
	chunks_to_embed = []
	chunk_indices = []
	
	for i, chunk in enumerate(chunks):
		chunk_text = chunk["chunk"]
		
		if use_cache:
			cached_embedding = get_cached_embedding(chunk_text)
			if cached_embedding is not None:
				# Use cached embedding
				embeddings.append((i, np.array(cached_embedding)))
				continue
		
		# Need to embed this chunk
		chunks_to_embed.append(chunk_text)
		chunk_indices.append(i)
	
	# Batch embed all chunks that weren't cached
	if chunks_to_embed:
		new_embeddings = model.encode(chunks_to_embed)
		for idx, embedding, chunk_text in zip(chunk_indices, new_embeddings, chunks_to_embed):
			# Save to cache
			if use_cache:
				save_embedding_to_cache(chunk_text, embedding)
			embeddings.append((idx, embedding))
	
	# Sort by original index and return just the embeddings
	embeddings.sort(key=lambda x: x[0])
	return [emb for _, emb in embeddings]

# In-memory vector store (replaces ChromaDB due to dependency issues)
vector_store = {
    "chunks": [],
    "embeddings": [],
    "ids": []
}

def setup_vector_store(chunks, embeddings):
	"""Store chunks and embeddings in memory."""
	vector_store["chunks"] = chunks
	vector_store["embeddings"] = embeddings
	vector_store["ids"] = [f'chunk_{i}' for i in range(len(chunks))]

def query_vector_store(query_text, top_k=3):
	"""
	Query the vector store and return top-k similar chunks.
	Uses cosine similarity (like Lesson 1).
	"""
	# 1. Embed the query
	query_embedding = model.encode([query_text])[0]
	
	# 2. Calculate cosine similarity with all stored embeddings
	embeddings_array = np.array(vector_store["embeddings"])
	query_array = np.array(query_embedding).reshape(1, -1)
	
	# Normalize vectors for cosine similarity
	embeddings_norm = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
	query_norm = query_array / np.linalg.norm(query_array, axis=1, keepdims=True)
	
	# Cosine similarity: dot product of normalized vectors
	similarities = np.dot(embeddings_norm, query_norm.T).flatten()
	
	# 3. Get top-k indices
	top_indices = np.argsort(similarities)[::-1][:top_k]
	
	# 4. Return results with metadata
	results = {
		"ids": [vector_store["ids"][i] for i in top_indices],
		"documents": [vector_store["chunks"][i]["chunk"] for i in top_indices],
		"metadatas": [vector_store["chunks"][i]["metadata"] for i in top_indices],
		"distances": [float(1 - similarities[i]) for i in top_indices]  # Convert similarity to distance
	}
	
	return results


def get_cached_embedding(chunk_text):
	"""Get embedding from cache if it exists."""
	chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
	return embedding_cache.get(chunk_hash)

def save_embedding_to_cache(chunk_text, embedding):
	"""Save embedding to cache."""
	chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()
	embedding_cache.set(chunk_hash, embedding.tolist() if hasattr(embedding, 'tolist') else embedding)

embeddings = generate_embeddings(all_chunks)
setup_vector_store(all_chunks, embeddings)
results = query_vector_store("How many years of experience do I have?")

print(results)
print(f"Total chunks to embed: {len(all_chunks)}")


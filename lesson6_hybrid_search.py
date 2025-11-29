import re
import string
import numpy as np
from rank_bm25 import BM25Okapi

from helpers import load_resume_data, parse_sections
from lesson2_chunking import chunk_section, get_all_chunks
from lesson3_embeddings import setup_vector_store, query_vector_store, generate_embeddings
from lesson4_retrieval import preprocess_query


"""
Day 6: Hybrid Search Implementation

Combines semantic search (embeddings) with keyword search (BM25) for better retrieval.
"""
def tokenize(text):
	text = text.translate(str.maketrans('', '', string.punctuation))
	tokens = [word.lower() for word in text.split() if word.strip()]
	return tokens

# Step 1: Build BM25 Index
# -------------------------
# Question: What data does BM25 need to work?
# Hint: BM25Okapi needs a list of tokenized documents (list of lists of words)
def build_bm25_index(vector_store):
#   1. Gets all chunks from vector_store["chunks"]
	chunks = vector_store["chunks"]
#   2. Extracts the text from each chunk (each chunk is a dict with "chunk" key)
	chunks_list = [chunk["chunk"] for chunk in chunks]
#   3. Tokenizes each chunk's text using the tokenize() function above
	token_lists = [tokenize(chunk) for chunk in chunks_list]
#   4. Creates a BM25Okapi index from the tokenized chunks
	bm25_index = BM25Okapi(token_lists)
#   5. Returns the BM25 index
	return bm25_index

# Step 2: Keyword Search Function
# ---------------------------------
# Question: How do you search with BM25 once you have an index?
# Hint: BM25Okapi has a get_scores() method that takes tokenized query words
def keyword_search(query, bm25_index, vector_store, top_k=3):
#   1. Tokenizes the query using tokenize()
	query_tokens = tokenize(query)
#   2. Gets BM25 scores for all chunks using bm25_index.get_scores()
	scores = bm25_index.get_scores(query_tokens)
#   3. Finds the top-k indices (highest scores first - use np.argsort)
	top_k_indices = np.argsort(scores)[::-1][:top_k]
#   4. Returns results in the same format as query_vector_store:
#      {"ids": [...], "documents": [...], "metadatas": [...], "scores": [...]}
	documents = [vector_store["chunks"][i]["chunk"] for i in top_k_indices]
	metadatas = [vector_store["chunks"][i]["metadata"] for i in top_k_indices]
	scores = [scores[i] for i in top_k_indices]
	ids = [vector_store["ids"][i] for i in top_k_indices]
	return {"ids": ids, "documents": documents, "metadatas": metadatas, "scores": scores}
#
# Important: BM25 returns scores (higher = better), not distances (lower = better)
#            So you'll use "scores" not "distances" in your return dict
def normalize_bm25_scores(keyword_results):
	# Get all BM25 scores from keyword_results
	bm25_scores = keyword_results["scores"]
	# Find min and max
	min_score = min(bm25_scores) if bm25_scores else 0
	max_score = max(bm25_scores) if bm25_scores else 1
	# Normalize: maps [min_score, max_score] → [0, 1]
	if max_score > min_score:
		normalized_bm25 = [(score - min_score) / (max_score - min_score) for score in bm25_scores]
	else:
		# All scores are the same (or empty)
		normalized_bm25 = [0.0] * len(bm25_scores) if bm25_scores else []
	return normalized_bm25

def merge_results(semantic_results, keyword_results, normalized_bm25_scores, normalized_semantic_scores):
	merged_results = {}

	# First, add all semantic results
	for i, chunk_id in enumerate(semantic_results["ids"]):
		merged_results[chunk_id] = {
			"semantic_score": normalized_semantic_scores[i],
			"keyword_score": 0.0,  # Default to 0 if not in keyword results
			"document": semantic_results["documents"][i],
			"metadata": semantic_results["metadatas"][i]
		}

	# Then, add/update with keyword results
	for i, chunk_id in enumerate(keyword_results["ids"]):
		if chunk_id in merged_results:
			# Chunk appears in both - update keyword_score
			merged_results[chunk_id]["keyword_score"] = normalized_bm25_scores[i]
		else:
			# Chunk only in keyword results - add it
			merged_results[chunk_id] = {
				"semantic_score": 0.0,  # Default to 0 if not in semantic results
				"keyword_score": normalized_bm25_scores[i],
				"document": keyword_results["documents"][i],
				"metadata": keyword_results["metadatas"][i]
			}
	# Step 4: Calculate combined scores
	semantic_weight = 0.7
	keyword_weight = 0.3

	for chunk_id, chunk_data in merged_results.items():
		combined_score = (
			semantic_weight * chunk_data["semantic_score"] + 
			keyword_weight * chunk_data["keyword_score"]
		)
		chunk_data["combined_score"] = combined_score

	return merged_results

def hybrid_search(query, bm25_index, vector_store, top_k=3, use_keyword_search=True):
	if not query or not isinstance(query, str) or not query.strip():
		raise ValueError("query must be a non-empty string")
	if bm25_index is None:
		raise ValueError("bm25_index cannot be None")
	if vector_store is None:
		raise ValueError("vector_store cannot be None")
	if top_k <= 0:
		raise ValueError("top_k must be greater than 0")

	semantic_results = query_vector_store(query, top_k=top_k*2)
	if not use_keyword_search:
		return semantic_results
	else:
		try:
			# Step 1: Run both searches
			keyword_results = keyword_search(query, bm25_index, vector_store, top_k)
			# Step 2: Normalize scores
			# Semantic: distances are 0-1, convert to similarity (1 - distance)
			# BM25: normalize to 0-1 (divide by max, or use min-max scaling)
			normalized_bm25_scores = normalize_bm25_scores(keyword_results)
			normalized_semantic_scores = [1 - distance for distance in semantic_results["distances"]]
			# Step 3: Merge results by chunk ID
			# Use a dict: {chunk_id: {"semantic_score": ..., "keyword_score": ..., "document": ..., "metadata": ...}}
			merged_results = merge_results(semantic_results, keyword_results, normalized_bm25_scores, normalized_semantic_scores)
			# Step 4: Calculate combined scores
			# combined_score = semantic_weight * semantic_score + keyword_weight * keyword_score
			sorted_chunks = sorted(
				merged_results.items(), 
				key=lambda x: x[1]["combined_score"], 
				reverse=True
			)[:top_k]
			# Step 5: Sort by combined_score (descending) and return top_k
			return {
				"ids": [chunk_id for chunk_id, _ in sorted_chunks],
				"documents": [data["document"] for _, data in sorted_chunks],
				"metadatas": [data["metadata"] for _, data in sorted_chunks],
				"scores": [data["combined_score"] for _, data in sorted_chunks]
			}
		except Exception:
			return semantic_results

# Step 3: Test It
# ---------------
# data = load_resume_data()
# all_chunks = get_all_chunks(data)
# embeddings = generate_embeddings(all_chunks)
# vector_store = setup_vector_store(all_chunks, embeddings)
# bm25_index = build_bm25_index(vector_store)
# results = hybrid_search("python", bm25_index, vector_store)
# print("Results: ", results)

# def test_hybrid_search(bm25_index, vector_store):
#     """
#     Test hybrid search vs pure semantic search on 10 diverse queries.
#     """
#     test_queries = [
#         "Python",  # Exact keyword
#         "leadership experience",  # Semantic/conceptual
#         "React",  # Exact keyword
#         "cloud deployment",  # Semantic
#         "machine learning",  # Could be both
#         "AWS",  # Exact keyword
#         "front-end architecture",  # Semantic
#         "Node.js",  # Exact keyword (with slash)
#         "software engineering experience",  # Semantic
#         "database",  # Could match MySQL, Postgres
#     ]
    
#     print("=" * 60)
#     print("HYBRID SEARCH TEST RESULTS")
#     print("=" * 60)
    
#     for query in test_queries:
#         print(f"\nQuery: '{query}'")
#         print("-" * 60)
        
#         # Pure semantic search
#         semantic_results = query_vector_store(query, top_k=3)
        
#         # Hybrid search
#         hybrid_results = hybrid_search(query, bm25_index, vector_store, top_k=3)
        
#         # Compare top result
#         print(f"Semantic top: {semantic_results['documents'][0][:80]}...")
#         print(f"Hybrid top:   {hybrid_results['documents'][0][:80]}...")
#         print(f"Hybrid score: {hybrid_results['scores'][0]:.3f}")

# test_hybrid_search(bm25_index, vector_store)

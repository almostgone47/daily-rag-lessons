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
	# Remove punctuation and convert to lowercase
	text = text.translate(str.maketrans('', '', string.punctuation))
	# Split by whitespace and filter empty strings
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

# Step 3: Test It
# ---------------
# Once you've written both functions, test them:
#   1. Build the BM25 index
#   2. Search for "Python" 
#   3. Print the results to see if it finds chunks containing "Python"
data = load_resume_data()
all_chunks = get_all_chunks(data)
embeddings = generate_embeddings(all_chunks)
vector_store = setup_vector_store(all_chunks, embeddings)
bm25_index = build_bm25_index(vector_store)
results = keyword_search("Python", bm25_index, vector_store)
print("Results: ", results)
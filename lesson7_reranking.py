from sentence_transformers import CrossEncoder

from helpers import load_resume_data
from lesson2_chunking import get_all_chunks
from lesson3_embeddings import setup_vector_store, generate_embeddings
from lesson6_hybrid_search import build_bm25_index, hybrid_search


def setup_ranker():
	model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
	return model

def get_modified_query(query, context):
	templates = {
		"ats": "ATS keywords and resume optimization: {query}",
		"skills": "Required skills matching: {query}",
		"experience": "Relevant work experience: {query}"
	}
	if context and context in templates:
		return templates[context].format(query=query)
	return query

def should_rerank(search_results, confidence_threshold=0.7):
	scores = search_results["scores"]
	if not scores:
		return True
	top_score = max(scores)
	if top_score <= confidence_threshold:
		return True
	return False

def rerank(query, search_results, reranker, context=None, top_k=5):
	if not should_rerank(search_results):
		return {
			"ids": search_results["ids"][:top_k],
			"documents": search_results["documents"][:top_k],
			"metadatas": search_results["metadatas"][:top_k],
			"scores": search_results["scores"][:top_k]
		}
	documents = search_results["documents"]
	modified_query = get_modified_query(query, context)
	pairs = [(modified_query, document) for document in documents]
	scores = reranker.predict(pairs)
	documents_with_scores = list(zip(
		search_results["ids"],
		search_results['metadatas'],
		documents, 
		scores
	))
	sorted_documents = sorted(documents_with_scores, key=lambda x: x[3], reverse=True)
	top_k_zipped = sorted_documents[:top_k]
	# Return in same format as hybrid_search
	return {
		"ids": [item[0] for item in top_k_zipped],
		"documents": [item[2] for item in top_k_zipped],
		"metadatas": [item[1] for item in top_k_zipped],
		"scores": [item[3] for item in top_k_zipped]
	}

# Test the reranking pipeline
# if __name__ == "__main__":
# 	# Load and prepare data
# 	print("Loading data...")
# 	data = load_resume_data()
# 	all_chunks = get_all_chunks(data)
# 	embeddings = generate_embeddings(all_chunks)
# 	vector_store = setup_vector_store(all_chunks, embeddings)
# 	bm25_index = build_bm25_index(vector_store)
	
# 	# Setup reranker
# 	print("Setting up reranker...")
# 	reranker = setup_ranker()
	
# 	# Test query
# 	query = "Python programming experience"
# 	print(f"\nQuery: '{query}'")
# 	print("=" * 60)
	
# 	# Stage 1: Hybrid search (retrieve top-20)
# 	print("\nStage 1: Hybrid Search (top-20)...")
# 	hybrid_results = hybrid_search(query, bm25_index, vector_store, top_k=20)
# 	print(f"Retrieved {len(hybrid_results['documents'])} candidates")
	
# 	# Stage 2: Rerank to top-5
# 	print("\nStage 2: Reranking (top-5)...")
# 	reranked_results = rerank(query, hybrid_results, reranker, context="experience", top_k=5)
	
# 	# Print results
# 	print("\nFinal Reranked Results:")
# 	print("=" * 60)
# 	for i, (doc, score, metadata) in enumerate(zip(
# 		reranked_results["documents"],
# 		reranked_results["scores"],
# 		reranked_results["metadatas"]
# 	), 1):
# 		print(f"\n{i}. Score: {score:.4f}")
# 		print(f"   Section: {metadata.get('section_type', 'N/A')}")
# 		print(f"   Document: {doc}")
# 		print("-" * 60)
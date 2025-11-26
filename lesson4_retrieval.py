import string

from lesson3_embeddings import query_vector_store


def preprocess_query(query):
	text = query.translate(str.maketrans('', '', string.punctuation))
	return text.lower().strip()

def filter_results(results, threshold=0.7):
	similarities = [1 - distance for distance in results["distances"]]
	valid_indices = [index for index, similarity in enumerate(similarities) if similarity >= threshold]
	filtered = {
		"ids": [results["ids"][index] for index in valid_indices],
		"documents": [results["documents"][index] for index in valid_indices],
		"distances": [results["distances"][index] for index in valid_indices],
		"metadatas": [results["metadatas"][index] for index in valid_indices],
	}
	return filtered

# Just for debugging - show what was retrieved
def format_results(results):
    for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"]), 1):
        print(f"Chunk {i}: {doc[:100]}...")
        print(f"Section: {metadata.get('section_type')}")

def retrieval_pipeline(query, top_k=3, threshold=0.7):
	clean_query = preprocess_query(query)
	results = query_vector_store(clean_query, top_k)
	filtered = filter_results(results, threshold)
	formatted = format_results(filtered)
	return formatted


# query = preprocess_query("What are the candidate's technical skills?")
# results = retrieval_pipeline(query)
# print(results)
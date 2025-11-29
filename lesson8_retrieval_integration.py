import time
from helpers import load_resume_data
from lesson2_chunking import get_all_chunks
from lesson3_embeddings import generate_embeddings, setup_vector_store
from lesson6_hybrid_search import hybrid_search, build_bm25_index
from lesson7_reranking import rerank, setup_ranker


def setup_data():
	try:
		data = load_resume_data()
		chunks = get_all_chunks(data)
		if not chunks:
			raise ValueError("No chunks generated")
		embeddings = generate_embeddings(chunks)
		if not embeddings:
			raise ValueError("No embeddings generated")
		vector_store = setup_vector_store(chunks, embeddings)
		return vector_store
	except Exception as e:
		raise ValueError(f"Failed to setup data: {e}")

def retrieval(query, vector_store, bm25_index, reranker, context=None):
	if not query or not query.strip():
		raise ValueError("Invalid query: must be a non-empty string")

	try:
		search_results = hybrid_search(query, bm25_index, vector_store)
		reranked_results = rerank(query, search_results, reranker, context=context)
		return reranked_results
	except Exception as e:
		# If rerank fails, try to return hybrid search results
		try:
			if 'search_results' in locals():
				return search_results
			search_results = hybrid_search(query, bm25_index, vector_store)
			return search_results  # Partial result - no reranking
		except Exception:
			# If hybrid search fails, return empty results with error flag
			return {
				"ids": [],
				"documents": [],
				"metadatas": [],
				"scores": [],
				"error": str(e)
			}

def evaluate_retrieval_system(vector_store, bm25_index, reranker):
	"""
	Evaluation script that tests the retrieval system on 20 diverse queries.
	Measures performance and displays results for manual relevance checking.
	"""
	test_queries = [
		# Skills-based queries
		"Python programming experience",
		"JavaScript and React skills",
		"Cloud deployment experience",
		"Database management",
		"Machine learning experience",
		
		# Experience-based queries
		"Software engineering roles",
		"Leadership and team management",
		"Project management experience",
		"Client-facing work experience",
		"Remote work experience",
		
		# Education queries
		"Educational background",
		"University degree",
		"Certifications and training",
		
		# Specific technologies
		"AWS cloud services",
		"Docker and containerization",
		"API development",
		"Frontend development",
		"Backend architecture",
		
		# General queries
		"What are your main skills?",
		"Years of professional experience",
		"Recent work projects"
	]
	
	print("=" * 80)
	print("RETRIEVAL SYSTEM EVALUATION")
	print("=" * 80)
	print(f"Testing {len(test_queries)} diverse queries...\n")
	
	results_summary = {
		"total_queries": len(test_queries),
		"successful_queries": 0,
		"failed_queries": 0,
		"total_time": 0,
		"query_times": [],
		"results_with_errors": 0
	}
	
	for i, query in enumerate(test_queries, 1):
		print(f"\n{'=' * 80}")
		print(f"Query {i}/{len(test_queries)}: '{query}'")
		print("=" * 80)
		
		start_time = time.time()
		try:
			results = retrieval(query, vector_store, bm25_index, reranker)
			elapsed_time = time.time() - start_time
			
			results_summary["query_times"].append(elapsed_time)
			results_summary["total_time"] += elapsed_time
			
			if "error" in results:
				results_summary["results_with_errors"] += 1
				print(f"⚠️  Error: {results['error']}")
			else:
				results_summary["successful_queries"] += 1
				print(f"✅ Retrieved {len(results.get('documents', []))} results in {elapsed_time:.3f}s")
				
				# Display top 3 results
				documents = results.get("documents", [])
				scores = results.get("scores", [])
				metadatas = results.get("metadatas", [])
				
				for j, (doc, score, metadata) in enumerate(zip(documents[:3], scores[:3], metadatas[:3]), 1):
					print(f"\n  Top {j} (Score: {score:.4f}):")
					print(f"    Section: {metadata.get('section_type', 'N/A')}")
					print(f"    Preview: {doc[:150]}...")
			
			# Check performance requirement (<3 seconds)
			if elapsed_time > 3.0:
				print(f"⚠️  Performance warning: Query took {elapsed_time:.3f}s (target: <3s)")
			else:
				print(f"✓ Performance: {elapsed_time:.3f}s (within <3s target)")
				
		except Exception as e:
			elapsed_time = time.time() - start_time
			results_summary["failed_queries"] += 1
			results_summary["query_times"].append(elapsed_time)
			results_summary["total_time"] += elapsed_time
			print(f"❌ Query failed: {e}")
			print(f"   Time: {elapsed_time:.3f}s")
	
	# Print summary statistics
	print("\n" + "=" * 80)
	print("EVALUATION SUMMARY")
	print("=" * 80)
	print(f"Total queries: {results_summary['total_queries']}")
	print(f"Successful: {results_summary['successful_queries']}")
	print(f"Failed: {results_summary['failed_queries']}")
	print(f"Results with errors: {results_summary['results_with_errors']}")
	
	if results_summary['query_times']:
		avg_time = results_summary['total_time'] / len(results_summary['query_times'])
		max_time = max(results_summary['query_times'])
		min_time = min(results_summary['query_times'])
		print(f"\nPerformance Metrics:")
		print(f"  Average time: {avg_time:.3f}s")
		print(f"  Min time: {min_time:.3f}s")
		print(f"  Max time: {max_time:.3f}s")
		print(f"  Total time: {results_summary['total_time']:.3f}s")
		
		# Check if all queries meet <3s requirement
		queries_within_target = sum(1 for t in results_summary['query_times'] if t < 3.0)
		print(f"  Queries within <3s target: {queries_within_target}/{len(results_summary['query_times'])}")
	
	print("\n" + "=" * 80)
	print("NOTE: Manual relevance checking required for >85% relevance metric")
	print("Review the top-3 results for each query above to assess relevance.")
	print("=" * 80)

if __name__ == "__main__":
	# Setup the retrieval system
	print("Setting up retrieval system...")
	vector_store = setup_data()
	bm25_index = build_bm25_index(vector_store)
	reranker = setup_ranker()
	print("Setup complete!\n")
	
	# Choose: Run single query test or full evaluation
	run_evaluation = True  # Set to False to run a single query test instead
	
	if run_evaluation:
		# Run full evaluation on 20 diverse queries
		evaluate_retrieval_system(vector_store, bm25_index, reranker)
	else:
		# Run a single query test
		query = "Summarize your experience with Python."
		print(f"Query: '{query}'")
		print("=" * 80)
		results = retrieval(query, vector_store, bm25_index, reranker)
		
		print(f"\nRetrieved {len(results.get('documents', []))} results:")
		for i, (doc, score, metadata) in enumerate(zip(
			results.get("documents", []),
			results.get("scores", []),
			results.get("metadatas", [])
		), 1):
			print(f"\n{i}. Score: {score:.4f}")
			print(f"   Section: {metadata.get('section_type', 'N/A')}")
			print(f"   Document: {doc}")
			print("-" * 80)

"""
Day 5: Advanced Chunking for Resumes

This lesson covers embedding-based semantic chunking, which is different from
the structure-based semantic chunking we use for resumes.

WHY WE'RE NOT USING EMBEDDING-BASED SEMANTIC CHUNKING FOR RESUMES:
====================================================================

1. Resumes have clear structure: bullets, sections, headings
   - Natural boundaries already exist (each bullet = semantic unit)
   - No need to discover boundaries automatically

2. Structure-based chunking is more accurate for resumes:
   - "Built React app" and "Led team of 5" are different bullets
   - Embedding similarity might incorrectly merge them if they're semantically similar
   - Structure preserves the author's intended organization

3. Performance: Structure-based is faster (no embedding computation needed)

WHEN TO USE EMBEDDING-BASED SEMANTIC CHUNKING:
==============================================

Use embedding-based semantic chunking when:
- Working with unstructured text (novels, articles without clear headings)
- No natural boundaries exist (long paragraphs, continuous text)
- Need to automatically discover topic boundaries
- Text doesn't follow a predictable structure

Examples:
- Long-form articles without section headings
- Books or novels
- Customer reviews or feedback
- Research papers with dense paragraphs
- Chat logs or transcripts
"""

import numpy as np
from sentence_transformers import SentenceTransformer


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Alternative to sklearn.metrics.pairwise.cosine_similarity
    """
    vec1 = np.array(vec1).flatten()
    vec2 = np.array(vec2).flatten()
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


# ============================================================================
# EXAMPLE 1: Pure Python Implementation
# ============================================================================

def semantic_chunk_by_embeddings_python(text, model, similarity_threshold=0.7, min_chunk_size=100):
    """
    Embedding-based semantic chunking using pure Python.
    
    Strategy:
    1. Split text into sentences
    2. Embed each sentence
    3. Calculate similarity between adjacent sentences
    4. Merge sentences if similarity > threshold (same topic)
    5. Split when similarity < threshold (different topic)
    
    Args:
        text: Long unstructured text to chunk
        model: SentenceTransformer model for embeddings
        similarity_threshold: Merge chunks if similarity > this (default 0.7)
        min_chunk_size: Minimum characters per chunk (default 100)
    
    Returns:
        List of semantically coherent chunks
    """
    # Step 1: Split into sentences (simple approach - in production, use NLTK/spaCy)
    sentences = text.split('. ')
    sentences = [s.strip() + '.' for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return [text]
    
    # Step 2: Generate embeddings for all sentences
    embeddings = model.encode(sentences)
    
    # Step 3: Calculate similarity between adjacent sentences
    chunks = []
    current_chunk = [sentences[0]]
    current_embedding = embeddings[0]  # Start with first sentence embedding
    
    for i in range(1, len(sentences)):
        # Calculate similarity between current chunk and next sentence
        next_embedding = embeddings[i]
        # Get the current chunk's embedding (average if multiple sentences)
        if len(current_embedding.shape) > 1:
            current_emb = current_embedding[0]
        else:
            current_emb = current_embedding
        similarity = cosine_similarity(current_emb, next_embedding)
        
        # Step 4: Decision logic
        if similarity >= similarity_threshold:
            # Similar topics - merge into current chunk
            current_chunk.append(sentences[i])
            # Update chunk embedding (average of all sentences in chunk)
            current_embedding = model.encode(' '.join(current_chunk))
        else:
            # Different topic - finalize current chunk and start new one
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= min_chunk_size:
                chunks.append(chunk_text)
            current_chunk = [sentences[i]]
            current_embedding = embeddings[i]
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text) >= min_chunk_size:
            chunks.append(chunk_text)
    
    return chunks


# ============================================================================
# EXAMPLE 2: LangChain Implementation (Much Simpler!)
# ============================================================================

def semantic_chunk_by_embeddings_langchain_example():
    """
    LangChain's SemanticChunker makes embedding-based semantic chunking very simple!
    
    How it works:
    1. Splits text into sentences
    2. Embeds each sentence
    3. Calculates similarity between adjacent sentences
    4. Combines similar sentences into chunks
    5. Splits when similarity drops (topic change)
    
    Note: Requires langchain-experimental and an embedding provider (OpenAI, etc.)
    """
    example_code = '''
# Load the documents
from langchain_community.document_loaders import TextLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

loader = TextLoader("langchain_intro.txt")
docs = loader.load()

# Initialize embedding model
embedding = OpenAIEmbeddings()

# Create the semantic chunker
chunker = SemanticChunker(embedding)

# Split the documents
chunks = chunker.split_documents(docs)

# Result
for i, chunk in enumerate(chunks):
    print(f"\\nChunk {i+1}:\\n{chunk.page_content}")
'''
    
    print("=" * 70)
    print("LANGCHAIN SEMANTIC CHUNKING (Simple Example)")
    print("=" * 70)
    print("\nLangChain makes this incredibly simple - just a few lines:")
    print(example_code)
    print("\nWhat happens under the hood:")
    print("  1. Splits text into sentences")
    print("  2. Embeds each sentence using the embedding model")
    print("  3. Calculates similarity between adjacent sentences")
    print("  4. Combines similar sentences (same topic)")
    print("  5. Splits when similarity drops (topic change)")
    print("\nInstallation:")
    print("  uv pip install langchain-experimental langchain-openai")
    print("\nNote: For resumes, we use structure-based chunking instead")
    print("      (chunking by bullets/sections), which is faster and more accurate.")


# ============================================================================
# COMPARISON: Structure-based vs Embedding-based
# ============================================================================

def compare_chunking_strategies():
    """
    Demonstrates the difference between structure-based and embedding-based chunking.
    """
    # Example: Long unstructured article
    unstructured_text = """
    Machine learning has revolutionized many industries. Deep learning models can process
    vast amounts of data. Neural networks are particularly effective for image recognition.
    However, training these models requires significant computational resources. GPUs have
    become essential for modern AI research. The cost of training large models can be
    prohibitive for smaller organizations. Cloud computing has made AI more accessible.
    Many companies now offer AI-as-a-Service platforms. This democratizes access to
    powerful AI capabilities. Startups can now leverage state-of-the-art models without
    massive infrastructure investments.
    """
    
    print("=" * 70)
    print("COMPARISON: Structure-based vs Embedding-based Chunking")
    print("=" * 70)
    
    # Structure-based (what we use for resumes)
    print("\n1. STRUCTURE-BASED (for resumes):")
    print("   - Chunks by bullets, sections, headings")
    print("   - Uses document structure as boundaries")
    print("   - Fast, accurate for structured documents")
    print("   - Example: Each bullet point = one chunk")
    
    # Embedding-based (for unstructured text)
    print("\n2. EMBEDDING-BASED (for unstructured text):")
    print("   - Chunks by semantic similarity")
    print("   - Discovers topic boundaries automatically")
    print("   - Slower (requires embedding computation)")
    print("   - Example: Merges similar sentences, splits on topic changes")
    
    # When to use each
    print("\n3. WHEN TO USE EACH:")
    print("   Structure-based: Resumes, structured documents, articles with headings")
    print("   Embedding-based: Novels, long paragraphs, unstructured text")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Initialize model (same one we use in helpers.py)
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # Example unstructured text
    sample_text = """
    Artificial intelligence is transforming healthcare. Machine learning algorithms
    can analyze medical images with remarkable accuracy. These systems help doctors
    diagnose diseases earlier. However, AI in healthcare raises ethical concerns.
    Patient privacy must be protected. Regulatory frameworks are evolving to address
    these challenges. The future of medicine will likely involve human-AI collaboration.
    Doctors will use AI as a tool to enhance their expertise. This partnership can
    improve patient outcomes significantly.
    """
    
    print("=" * 70)
    print("EMBEDDING-BASED SEMANTIC CHUNKING EXAMPLE")
    print("=" * 70)
    
    # Test the Python implementation
    print("\nChunking text with similarity threshold = 0.7...")
    chunks = semantic_chunk_by_embeddings_python(
        sample_text, 
        model, 
        similarity_threshold=0.7,
        min_chunk_size=50
    )
    
    print(f"\nCreated {len(chunks)} semantic chunks:\n")
    for i, chunk in enumerate(chunks, 1):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(f"  {chunk[:100]}...")
        print()
    
    # Show comparison
    compare_chunking_strategies()
    
    # Show LangChain example
    print("\n" + "=" * 70)
    semantic_chunk_by_embeddings_langchain_example()


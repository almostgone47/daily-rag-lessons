
from sentence_transformers import SentenceTransformer

from helpers import load_resume_data, parse_sections, get_most_similar, ask_llm


model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

data = load_resume_data()
resume_sections = parse_sections(data)
# create query string
query_string = "How many years of experience do you have as a software engineer?"
# embed query string
embed_query = model.encode(query_string)
# embed resume sections
embed_chunks = model.encode(resume_sections)
# get similarity tensor
similarity = model.similarity(embed_query, embed_chunks)

most_similar = get_most_similar(similarity, resume_sections)
print(f"Best match (index {most_similar['index']}, score {most_similar['score']:.4f}):")
print(most_similar['chunk'])
print("\n" + "="*50 + "\n")

# Final step: Pass query + retrieved chunk to LLM
answer = ask_llm(query_string, most_similar['chunk'])
print(f"Question: {query_string}")
print(f"\nAnswer: {answer}")
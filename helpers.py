import os
import json
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load environment variables
load_dotenv()
# Initialize Groq client with API key from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def load_resume_data():
	try:
		with open('resume.json', 'r') as f:
			data = json.load(f)
		if not data:
			raise ValueError("No data loaded")
		return data
	except Exception as e:
		raise ValueError(f"Failed to load resume data: {e}")

def parse_sections(data):
	chunks = []
	sections = data.get('sections', [])
	
	for section in sections:
		kind = section.get('kind', '')
		title = section.get('title', '')
		
		# Handle RICHTEXT sections (like Summary)
		if kind == 'RICHTEXT':
			content = section.get('content', '')
			if content:
				chunks.append(f"{title}: {content}")
		
		# Handle EXPERIENCE sections
		elif kind == 'EXPERIENCE':
			items = section.get('items', [])
			for item in items:
				headline = item.get('headline', '')
				subheadline = item.get('subheadline', '')
				start_date = item.get('startDate', '')
				end_date = item.get('endDate', '')
				location = item.get('location', '')
				bullets = item.get('bullets', [])
				
				# Format the experience chunk
				exp_text = f"Experience: {headline}"
				if subheadline:
					exp_text += f" at {subheadline}"
				if start_date or end_date:
					exp_text += f" ({start_date} to {end_date})"
				if location:
					exp_text += f" - {location}"
				if bullets:
					exp_text += ". " + ". ".join(bullets)
				
				chunks.append(exp_text)
		
		# Handle other section types (SKILLS, EDUCATION, etc.)
		else:
			content = section.get('content', '')
			if content:
				chunks.append(f"{title}: {content}")
	
	return chunks

def get_most_similar(similarity_tensor, chunks):
	# Convert tensor to numpy array and get the first row (since it's 2D)
	scores = similarity_tensor[0].numpy() if hasattr(similarity_tensor[0], 'numpy') else similarity_tensor[0]
	
	# Find index of highest similarity score
	best_idx = np.argmax(scores)
	best_score = scores[best_idx]
	
	return {
		'index': int(best_idx),
		'score': float(best_score),
		'chunk': chunks[best_idx]
	}


def ask_llm(query, context_chunk):
	"""
	Build a RAG prompt and get answer from LLM.
	This is the final step: query + retrieved context → LLM → answer
	"""
	# Build the RAG prompt
	prompt = f"""Based on the following resume information, answer the question.

	Resume Context:
	{context_chunk}

	Question: {query}

	Answer:"""
		
	# Call Groq API
	completion = client.chat.completions.create(
		model="llama-3.1-8b-instant",  # Free Groq model
		messages=[
			{
				"role": "user",
				"content": prompt
			}
		],
		temperature=0.7,
		max_tokens=512,
		stream=False  # Set to False for simpler handling
	)
	
	# Extract the answer
	answer = completion.choices[0].message.content
	return answer
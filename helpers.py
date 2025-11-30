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
		
		# Handle SKILLS sections
		elif kind == 'SKILLS':
			items = section.get('items', [])
			skill_names = []
			for item in items:
				headline = item.get('headline', '').strip()
				if headline:
					skill_names.append(headline)
			if skill_names:
				chunks.append(f"{title}: {', '.join(skill_names)}")
		
		# Handle EDUCATION sections
		elif kind == 'EDUCATION':
			items = section.get('items', [])
			for item in items:
				headline = item.get('headline', '')  # School name
				subheadline = item.get('subheadline', '')  # Degree
				start_date = item.get('startDate', '')
				end_date = item.get('endDate', '')
				location = item.get('location', '')
				
				# Format the education chunk
				edu_text = f"Education: {subheadline}" if subheadline else f"Education: {headline}"
				if headline and subheadline:
					edu_text += f" at {headline}"
				elif headline:
					edu_text = f"Education: {headline}"
				if start_date or end_date:
					edu_text += f" ({start_date} to {end_date})"
				if location:
					edu_text += f" - {location}"
				
				chunks.append(edu_text)
		
		# Handle other section types
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
	prompt = f"""Based on the following resume information, generate a professional resume summary.

	Resume Context:
	{context_chunk}

	Instructions: Write a concise 2-3 sentence professional summary in first person that highlights the candidate's experience, skills, and strengths. This text should be ready to copy and paste directly into a resume summary section. Write as if the candidate is describing themselves (use "I" or direct professional language). Do not use phrases like "The candidate is" or "This person has" - write it as the candidate would write it themselves.

	Professional Summary:"""
		
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
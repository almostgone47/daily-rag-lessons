from langchain_text_splitters import RecursiveCharacterTextSplitter

from helpers import load_resume_data, parse_sections


def chunk_structured_list(items, section_type):
    """
    Chunks structured list items (like Education) where each item is a separate chunk.
    Each item should be kept as its own semantic unit with metadata.
    """
    chunks = []
    
    for item in items:
        # Build chunk text from item fields
        # Education example: "Education: Bachelor's in Computer Science at University (2020-2024)"
        chunk_text_parts = []
        
        # Add section type prefix
        if section_type.upper() == "EDUCATION":
            if item.get("degree"):
                chunk_text_parts.append(f"Education: {item.get('degree', '')}")
            if item.get("school"):
                chunk_text_parts.append(f"at {item.get('school', '')}")
            if item.get("startDate") or item.get("endDate"):
                dates = f"({item.get('startDate', '')} to {item.get('endDate', '')})"
                chunk_text_parts.append(dates)
        # Skills handling
        elif section_type.upper() == "SKILLS":
            if item.get("headline"):
                chunk_text_parts.append(f"Skills: {item.get('headline', '')}")
        else:
            # Generic structured list handling
            # Include relevant fields based on what's available
            if item.get("title"):
                chunk_text_parts.append(f"{section_type}: {item.get('title', '')}")
            if item.get("subtitle"):
                chunk_text_parts.append(f"at {item.get('subtitle', '')}")
        
        chunk_text = " ".join(chunk_text_parts)
        
        # Create chunk with metadata
        chunk_data = {
            "chunk": chunk_text,
            "metadata": {
                "section_type": section_type,
                **{k: v for k, v in item.items() if k not in ["bullets", "bullets_rich", "meta"]}  # Include item fields as metadata
            }
        }
        chunks.append(chunk_data)
    
    return chunks


def chunk_by_size(content, section_type, chunk_size=500, chunk_overlap=50):
	"""
	Chunks content by character size using recursive splitting.
	Returns chunk objects with metadata for consistency with other chunking functions.
	"""
	chunks = []
	
	# Split the content using RecursiveCharacterTextSplitter
	text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
	split_chunks = text_splitter.split_text(content)
	
	# Convert each split chunk into a chunk object with metadata
	for chunk_text in split_chunks:
		chunk_data = {
			"chunk": chunk_text,
			"metadata": {
				"section_type": section_type,
			}
		}
		chunks.append(chunk_data)
	
	return chunks


# start with chunking experience but consider how to handle other sections with bullets after
def chunk_by_bullets(jobs, section_type):
	chunked_bullets = []

	for job in jobs:
		bullets = job.get("bullets", [])
		if not bullets:  # Skip items without bullets
			continue
		for bullet in bullets:
			# chunked_bullets.append(f"Experience: {job['headline']} at {job['subheadline']} from {job['startDate']} to {job['endDate']}. {bullet}")
			data = {
					"chunk": f"{section_type}: {job['headline']} at {job['subheadline']} from {job['startDate']} to {job['endDate']}. {bullet}",
					"metadata": {
						"section_type": section_type,
						"job_title": job["headline"],
						"company": job["subheadline"],
						"start_date": job["startDate"],
						"end_date": job["endDate"]
					}
				}
			chunked_bullets.append(data)

	return chunked_bullets

def chunk_section(section):
    """
    Smart chunking that adapts to section structure.
    """
    section_type = section.get("title", "")
    items = section.get("items", [])
    content = section.get("content", "")
    
    # Strategy 1: Has items with bullets? → Semantic chunking
    if items and any(item.get("bullets") for item in items):
        return chunk_by_bullets(items, section_type)
    
    # Strategy 2: Has content (paragraph)? → Recursive splitting if needed
    if content:
        if len(content) > 500:
            return chunk_by_size(content, section_type, chunk_size=500, chunk_overlap=50)
        else:
            return [{
				"chunk": content, 
				"metadata": {
					"section_type": section_type, 
				}
			}]
    
    # Strategy 3: Structured list (Education)? → Keep items separate
    if items and not any(item.get("bullets") for item in items):
        return chunk_structured_list(items, section_type)
    
    return []

def get_all_chunks(resume_data):
	all_chunks = []
	for section in resume_data.get("sections", []):
		chunks = chunk_section(section)
		all_chunks.extend(chunks)
	return all_chunks

# # # load resume data
# resume_data = load_resume_data()

# print(f"Total chunks created: {len(all_chunks)}")
# print("\nFirst few chunks:")
# for i, chunk in enumerate(all_chunks[:3], 1):
#     print(f"\nChunk {i}:")
#     print(f"  Text: {chunk['chunk'][:100]}...")
#     print(f"  Metadata: {chunk['metadata']}")
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helpers import load_resume_data, parse_sections

# load resume data
resume_data = load_resume_data()

# start with chunking experience but consider how to handle other sections with bullets after
def semantic_chunk_by_bullets(jobs):
	chunked_bullets = []

	for job in jobs:
		bullets = job["bullets"]
		for bullet in bullets:
			# chunked_bullets.append(f"Experience: {job['headline']} at {job['subheadline']} from {job['startDate']} to {job['endDate']}. {bullet}")
			data = {
					"chunk": f"Experience: {job['headline']} at {job['subheadline']} from {job['startDate']} to {job['endDate']}. {bullet}",
					"metadata": {
						"section_type": "EXPERIENCE",
						"job_title": job["headline"],
						"company": job["subheadline"],
						"start_date": job["startDate"],
						"end_date": job["endDate"]
					}
				}
			chunked_bullets.append(data)

	return chunked_bullets

all_jobs = []
for section in resume_data.get("sections", []):
    if section.get("kind") == "EXPERIENCE":
        all_jobs.extend(section.get("items", []))

# Now chunk by bullets
chunked_bullets = semantic_chunk_by_bullets(all_jobs)
print(chunked_bullets)
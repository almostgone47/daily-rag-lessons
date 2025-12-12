import os
from dotenv import load_dotenv
from langsmith import traceable

from lesson15_api_wrapper import generate_section_suggestions, get_section_text
from helpers import load_resume_data, load_job_description

load_dotenv()  # Load your .env file

@traceable
def generate_section_suggestions_with_tracing(
    resume_data,
    section_id: str,
    section_text: str,
    job_description: str
):
    """Wrapper function with tracing enabled."""
    return generate_section_suggestions(
        resume_data=resume_data,
        section_id=section_id,
        section_text=section_text,
        job_description=job_description,
        context={}
    )

# Load test data
resume_data = load_resume_data()
job_description = load_job_description()
section_text = get_section_text(resume_data, "experience")

# Call with tracing
print("Generating suggestions with LangSmith tracing...")
result = generate_section_suggestions_with_tracing(
    resume_data=resume_data,
    section_id="experience",
    section_text=section_text,
    job_description=job_description
)

print(f"\n✅ Generated {len(result)} suggestions")
for i, suggestion in enumerate(result[:3], 1):
    print(f"{i}. {suggestion.get('text', '')[:80]}...")
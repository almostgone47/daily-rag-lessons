from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add

from helpers import load_resume_data, load_job_description, parse_sections, ask_llm, extract_keywords_from_job_description

	
def merge_dicts(left: Dict[str, float], right: Dict[str, float]) -> Dict[str, float]:
    """Merge two dictionaries, combining their values."""
    result = left.copy() if left else {}
    if right:
        result.update(right)
    return result
	
ResumeState = TypedDict("ResumeState", {
	"resume_data": Dict[str, Any], 
	"extracted_sections": List[str], 
	"extracted_skills": List[str], 
	"summary": str, 
	"quality_score": int,
	"job_description": str,
	"ats_score": int,
	"skill_gap_score": int,
	"relevant_experience": str,
	"user_decision": str,
	"user_modifications": List[str],
	"analysis_times": Annotated[Dict[str, float], merge_dicts],
	"errors": Annotated[List[str], add]
	})

def prepare_documents_for_job(resume_id: str, job_id: str, graph):
    """
    Simulates API endpoint: POST /api/resumes/:id/customize
    In production, this would:
    - Load resume from database using resume_id
    - Load job from database using job_id
    - Invoke LangGraph workflow
    """
    # Simulate loading from database
    resume_data = load_resume_data()  # In production: db.get_resume(resume_id)
    job_description = load_job_description()  # In production: db.get_job(job_id).descriptionText
    
    # Invoke the workflow
    initial_state = {
        "resume_data": resume_data,
        "job_description": job_description
    }
    
    result = graph.invoke(initial_state)
    return result

def parse_resume(state: ResumeState):
	try:
		sections = parse_sections(state["resume_data"])
		
		# Extract skills from resume
		skills = []
		for section in state.get("resume_data", {}).get("sections", []):
			if section.get("kind", "").upper() == "SKILLS":
				skill_items = section.get("items", [])
				skills = [item.get("headline", "").strip() for item in skill_items if item.get("headline")]
		
		return {
			"extracted_sections": sections,
			"extracted_skills": skills,
		}
	except Exception as e:
		return {"errors": [str(e)]}

def check_ats_score(state: ResumeState):
    """
    Calculates ATS compatibility score based on keyword matching.
    
    PRODUCTION IMPROVEMENTS:
    - Add keyword weighting (required vs nice-to-have skills from job description)
    - Use context-aware matching (e.g., "React" in "React.js" should match)
    - Handle synonyms and variations (e.g., "Node.js" = "NodeJS" = "node")
    - Normalize keywords using a skills taxonomy/database
    - Consider keyword frequency/importance in job description
    - Add industry-specific keyword dictionaries
    - Use embeddings for semantic similarity (not just exact matches)
    """
    import time
    start_time = time.time()
    
    try:
        job_desc = state.get("job_description", "")
        if not job_desc:
            return {"errors": ["No job description provided"]}
        
        # Extract keywords from job description
        required_keywords = extract_keywords_from_job_description(job_desc)
        
        if not required_keywords:
            return {"ats_score": 0, "analysis_times": {"ats_score": time.time() - start_time}}
        
        resume_text = " ".join(state.get("extracted_sections", [])).lower()
        
        # Count matches
        # TODO: Improve matching with weighted scoring, synonyms, and context-aware matching
        matched = [kw for kw in required_keywords if kw in resume_text]
        
        score = int((len(matched) / len(required_keywords)) * 100) if required_keywords else 0
        
        elapsed = time.time() - start_time
        
        return {
            "ats_score": score,
            "analysis_times": {"ats_score": elapsed}
        }
    except Exception as e:
        return {"errors": [f"ATS check failed: {str(e)}"]}

def check_skill_gap_score(state: ResumeState):
    """
    Identifies missing skills by comparing job requirements to resume skills.
    
    PRODUCTION IMPROVEMENTS:
    - Use a skills taxonomy/database (e.g., LinkedIn Skills API, O*NET) for normalization
    - Filter out non-skills from extracted keywords (e.g., "fullstack" might not be a skill)
    - Handle skill variations and synonyms (e.g., "React" = "React.js" = "ReactJS")
    - Categorize skills (technical, soft skills, tools, frameworks) for better analysis
    - Use embeddings for fuzzy matching (semantic similarity, not just substring matching)
    - Weight skills by importance (required vs nice-to-have from job description)
    - Return structured data: missing_skills list, categorized by type, with suggestions
    """
    import time
    start_time = time.time()
    
    try:
        job_desc = state.get("job_description", "")
        if not job_desc:
            return {"errors": ["No job description provided"]}
        
        # Extract required skills from job description
        # TODO: Filter and normalize using skills taxonomy
        required_skills = extract_keywords_from_job_description(job_desc)
        
        if not required_skills:
            return {
                "skill_gap_score": 0, 
                "analysis_times": {"skill_gap_score": time.time() - start_time}
            }
        
        # Get resume skills (normalize to lowercase for comparison)
        resume_skills = state.get("extracted_skills", [])
        resume_skills_lower = [s.lower().strip() for s in resume_skills if s]
        
        # Find missing skills (required but not in resume)
        # TODO: Improve matching with embeddings and skills taxonomy
        missing_skills = []
        for skill in required_skills:
            # Check if skill is in resume (exact match or substring)
            found = False
            for resume_skill in resume_skills_lower:
                if skill in resume_skill or resume_skill in skill:
                    found = True
                    break
            if not found:
                missing_skills.append(skill)
        
        # Gap score: number of missing skills (or could be percentage)
        # TODO: Return structured data with categorized missing skills
        gap_score = len(missing_skills)
        
        elapsed = time.time() - start_time
        
        return {
            "skill_gap_score": gap_score,
            "analysis_times": {"skill_gap_score": elapsed}
        }
    except Exception as e:
        return {"errors": [f"Skill gap check failed: {str(e)}"]}

def check_relevant_experience(state: ResumeState):
    """
    Identifies and summarizes the most relevant work experiences for the job.
    
    PRODUCTION IMPROVEMENTS:
    - Return structured data: list of relevant experiences with relevance scores
    - Include specific bullet points that match job requirements (not just summary)
    - Rank experiences by relevance (most relevant first)
    - Quantify impact: extract metrics and achievements from matching experiences
    - Use embeddings for semantic matching (beyond LLM summary)
    - Return actionable suggestions: "Highlight this experience because..."
    - Add confidence scores for each experience match
    - Consider recency: weight recent experiences higher
    """
    import time
    start_time = time.time()
    
    try:
        job_desc = state.get("job_description", "")
        if not job_desc:
            return {"errors": ["No job description provided"]}
        
        # Get all experience sections from resume
        experience_text = "\n".join([
            section for section in state.get("extracted_sections", [])
            if "experience:" in section.lower()
        ])
        
        if not experience_text:
            return {
                "relevant_experience": "No experience found in resume",
                "analysis_times": {"relevant_experience": time.time() - start_time}
            }
        
        # Use LLM to find most relevant experiences
        # TODO: Return structured data with ranked experiences and specific bullet points
        prompt = f"""Based on this job description, identify which experiences from the resume are most relevant.

		Job Description:
		{job_desc}

		Resume Experiences:
		{experience_text}

		Return a brief summary (2-3 sentences) of the most relevant experiences that match the job requirements."""
        
        relevant_summary = ask_llm(prompt, experience_text)
        
        elapsed = time.time() - start_time
        
        return {
            "relevant_experience": relevant_summary,
            "analysis_times": {"relevant_experience": elapsed}
        }
    except Exception as e:
        return {"errors": [f"Relevant experience check failed: {str(e)}"]}

def aggregate_analyses(state: ResumeState):
    """
    Aggregates results from all parallel analyses.
    This node waits for all three analyses to complete before running.
    """
	# TODO: Add section-specific scoring
    # When optimizing a specific section, calculate:
    # - Section-specific ATS match score
    # - Section improvement percentage
    # - Impact on overall resume score
    # This will be used in guided mode for per-section optimization
    
    try:
        # Combine all analysis times
        all_times = state.get("analysis_times", {})
        
        # Create summary
        summary = {
            "ats_score": state.get("ats_score", 0),
            "skill_gap_score": state.get("skill_gap_score", 0),
            "relevant_experience": state.get("relevant_experience", ""),
            "total_time": sum(all_times.values()),
            "individual_times": all_times
        }
        
        print("\n=== Analysis Results ===")
        print(f"ATS Score: {summary['ats_score']}/100")
        print(f"Skill Gap: {summary['skill_gap_score']} missing skills")
        print(f"Relevant Experience: {summary['relevant_experience'][:100]}...")
        print(f"\nTiming:")
        for analysis, time_taken in all_times.items():
            print(f"  {analysis}: {time_taken:.3f}s")
        print(f"Total: {summary['total_time']:.3f}s")
        
        # Parallel execution performance:
        # Total time (1.696s) ≈ max(0.561, 0.575, 0.561) + overhead
        # Sequential would be: 0.561 + 0.575 + 0.561 = 1.697s
        # Parallel is faster because analyses run simultaneously, not one after another
        
        return {
            "summary": str(summary)  # Store as string for now
        }
    except Exception as e:
        return {"errors": [f"Aggregation failed: {str(e)}"]}

graph = StateGraph(ResumeState)
graph.add_node("parse_resume", parse_resume)
graph.add_node("check_ats_score", check_ats_score)
graph.add_node("check_skill_gap_score", check_skill_gap_score)
graph.add_node("check_relevant_experience", check_relevant_experience)
graph.add_node("aggregate_analyses", aggregate_analyses)
# Sequential: START -> parse_resume
graph.add_edge(START, "parse_resume")
# Parallel: All three analyses start after parse_resume
graph.add_edge("parse_resume", "check_ats_score")
graph.add_edge("parse_resume", "check_skill_gap_score")
graph.add_edge("parse_resume", "check_relevant_experience")
# All three feed into aggregation
graph.add_edge("check_ats_score", "aggregate_analyses")
graph.add_edge("check_skill_gap_score", "aggregate_analyses")
graph.add_edge("check_relevant_experience", "aggregate_analyses")

# End after aggregation
graph.add_edge("aggregate_analyses", END)
compiled_graph = graph.compile()
graph_return = prepare_documents_for_job("123", "456", compiled_graph)
# print("graph_return???: ", graph_return)
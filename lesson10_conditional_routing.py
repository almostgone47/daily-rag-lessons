from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add

from helpers import load_resume_data, parse_sections, ask_llm

ResumeState = TypedDict("ResumeState", {
	"resume_data": Dict[str, Any], 
	"extracted_sections": List[str], 
	"extracted_skills": List[str], 
	"summary": str, 
	"quality_score": int,
	"errors": Annotated[List[str], add]
	})

def required_sections_present(sections):
	section_names = [section["title"].upper() for section in sections]
	# Check for required sections (40 points)
	required_sections = ["EXPERIENCE", "EDUCATION", "SKILLS"]
	# ... how would you check if these exist?
	required_sections_present = True
	for section in required_sections:
		if section not in section_names:
			required_sections_present = False
			break
	return required_sections_present

def sections_have_content(sections):
	for section in sections:
		if section.get("kind", "").upper() == "RICHTEXT":
			if not bool(section.get("content", "").strip()):
				return False
		else:
			if not len(section.get("items", [])) > 0:
				return False
	return True

def validate_experience_dates(sections):
	for section in sections:
		current_section = section.get("title").upper()
		if current_section == "EXPERIENCE":
			experience_sections = section.get("items", [])
			break

	for experience in experience_sections:
		start_date = experience.get("startDate", "")
		end_date = experience.get("endDate", "")
		if start_date and end_date and start_date < end_date:
			return True
	return False

def calculate_quality_score(state: ResumeState):
	score = 0
	sections = state.get("resume_data", {}).get("sections", [])

	if required_sections_present(sections):
		score += 40
	# Check if sections have content (30 points)
	if sections_have_content(sections):
		score += 30
	# Data quality (20 points)
	# ... how would you validate dates, structure?
	if validate_experience_dates(sections):
		score += 20
	print("score:", score)
	# Bonus sections (10 points)
	# ... what extra sections would be a bonus?
	return {"quality_score": score}

def parse_resume(state: ResumeState):
	try:
		# print(f"Parsing resume...{state}")
		resume_data = load_resume_data()
		sections = parse_sections(resume_data)
		return {
			"resume_data": resume_data, 
			"extracted_sections": sections,
		}
	except Exception as e:
		return {"errors": [str(e)]}

def extract_skills(state: ResumeState):
	# print(f"Extracting skills...{state}")
	try:
		skills = []
		for section in state.get("resume_data", {}).get("sections", []):
			if section["kind"].upper() == "SKILLS":
				skill_items = section.get("items", [])
				skills = [item.get("headline", "") for item in skill_items]
				# print(f"Extracted skills: {skills}")
		return {"extracted_skills": skills}
	except Exception as e:
		return {"errors": [str(e)]}

def generate_summary(state: ResumeState):
	try:
		sections_text = "".join(state.get("extracted_sections", []))
		response = ask_llm("Generate a summary of the resume", sections_text)
		# print(f"Generated summary: {response}")
		return {"summary": response}
	except Exception as e:
		return {"errors": [str(e)]}

def error_handler(state: ResumeState):
    print("Errors: ", state.get("errors"))

def route_based_on_errors(state: ResumeState):
	if state.get("errors"):
		return "error_handler"
	else:
		return "extract_skills"

graph = StateGraph(ResumeState)
graph.add_node("parse_resume", parse_resume)
graph.add_node("extract_skills", extract_skills)
graph.add_node("generate_summary", generate_summary)
graph.add_node("error_handler", error_handler)
graph.add_edge(START, "parse_resume")
graph.add_conditional_edges("parse_resume", route_based_on_errors)
graph.add_edge("extract_skills", "generate_summary")
graph.add_edge("generate_summary", END)
graph.add_edge("error_handler", END)
graph = graph.compile()
result = graph.invoke({})
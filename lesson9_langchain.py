
from langgraph.graph import StateGraph, START, END 
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add

from helpers import load_resume_data, parse_sections, ask_llm

ResumeState = TypedDict("ResumeState", {
	"resume_data": Dict[str, Any], 
	"extracted_sections": List[str], 
	"extracted_skills": List[str], 
	"summary": str, 
	"errors": Annotated[List[str], add]
	})

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


graph = StateGraph(ResumeState)
graph.add_node("parse_resume", parse_resume)
graph.add_node("extract_skills", extract_skills)
graph.add_node("generate_summary", generate_summary)
graph.add_edge(START, "parse_resume")
graph.add_edge("parse_resume", "extract_skills")
graph.add_edge("extract_skills", "generate_summary")
graph.add_edge("generate_summary", END)
graph = graph.compile()
result = graph.invoke({})
print("Summary:", result.get("summary", ""))
print("\nFinal state errors:", result.get("errors", []))
print("Number of errors:", len(result.get("errors", [])))

graph_structure = graph.get_graph()
mermaid_diagram = graph_structure.draw_mermaid()
print(mermaid_diagram)
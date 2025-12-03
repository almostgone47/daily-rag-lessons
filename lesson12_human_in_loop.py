from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add

from lesson11_parallel_nodes import ResumeState, parse_resume, check_ats_score, check_skill_gap_score, check_relevant_experience, aggregate_analyses
from helpers import load_resume_data, load_job_description


def human_in_loop(state: ResumeState):
	"""
	Pauses workflow for user decision.
	"""
	print("Do you want to proceed with optimizations?")
	print("1. Yes")
	print("2. No")
	print("3. Modify")
	print("4. Exit")
	interrupt("Please enter your choice")
	return {
		"user_decision": "pending",
	}
	

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
# QUESTION WHY DO WE HARD CODE THREAD ID 1 HERE????????
	# Create config with thread_id for checkpointing
	config = {"configurable": {"thread_id": "1"}}
	result = graph.invoke(initial_state, config=config)
	current_state = graph.get_state(config=config)
	# print("current_state.next???: ", current_state.next)
	if current_state.next:
		choice = input("Please enter your choice: ")
		result = graph.invoke(
            Command(resume={"user_decision": choice, "should_optimize": choice == "1"}),
            config=config
        )
		print("Workflow resumed!", "user choice: ", choice)

	return result

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

# Human-in-the-loop: Pause for user decision
graph.add_node("human_in_loop", human_in_loop)
graph.add_edge("aggregate_analyses", "human_in_loop")
graph.add_edge("human_in_loop", END)
checkpointer = MemorySaver()
# End after aggregation
compiled_graph = graph.compile(checkpointer=checkpointer)
graph_return = prepare_documents_for_job("123", "456", compiled_graph)
# print("graph_return???: ", graph_return)
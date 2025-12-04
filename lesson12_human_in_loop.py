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
	interrupt("Please enter your choice") # Question: 
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
	# Actual LOG: current_state.next???:  ('human_in_loop',) why is this a tuple and should we use it to
	# check if the workflow is paused at the human_in_loop node in order use conditionals
	# to detirmine the next course of action?
	if current_state.next:
		import threading
		
		choice = None
		timeout_occurred = threading.Event()
		
		def get_input():
			nonlocal choice
			try:
				choice = input("Please enter your choice: ")
			except:
				pass
			timeout_occurred.set()
		
		# Start input thread
		input_thread = threading.Thread(target=get_input)
		input_thread.daemon = True
		input_thread.start()
		
		# Wait for input or timeout (5 minutes = 300 seconds)
		# For testing, use 30 seconds instead
		input_thread.join(timeout=30)  # Change to 300 for production
		
		if choice is None:
			print("\nTimeout: No response received. Defaulting to 'No' (2).")
			choice = "2"
		
		result = graph.invoke(
			Command(resume={"user_decision": choice}),
			config=config
		)
		# print("Workflow resumed! result:", result.get("user_decision"))

	return result

def generate_suggestions(state: ResumeState):
	"""
	Generates suggestions for the user.
	"""
	return {
		"user_modifications": "test suggestion, will be updated later",
	}

def apply_suggestions(state: ResumeState):
	"""
	Applies suggestions to the resume data.
	For now, this is a placeholder - will implement actual modifications later.
	"""
	# TODO: Apply the suggestions from user_modifications to resume_data
	# This should modify the resume_data structure with the optimized content
	# For now, just pass through the resume_data
	return {
		"resume_data": state.get("resume_data"),
	}

def error_handler(state: ResumeState):
    print("Errors: ", state.get("errors"))

def route_based_on_choice(state: ResumeState):
	choice = state.get("user_decision")
	if choice == "1":
		return "generate_suggestions"
	elif choice == "2":
		return END
	elif state.get("errors"):
		return "error_handler"
	else:
		# Default: if no valid choice, end the workflow
		return END

# score → review → optimize again OR exit → optimize → score → review → optimize again OR exit
graph = StateGraph(ResumeState)
graph.add_node("parse_resume", parse_resume)
graph.add_node("check_ats_score", check_ats_score)
graph.add_node("check_skill_gap_score", check_skill_gap_score)
graph.add_node("check_relevant_experience", check_relevant_experience)
graph.add_node("aggregate_analyses", aggregate_analyses)
# Human-in-the-loop: Pause for user decision
graph.add_node("human_in_loop", human_in_loop)
graph.add_node("generate_suggestions", generate_suggestions)
graph.add_node("apply_suggestions", apply_suggestions)

graph.add_node("error_handler", error_handler)
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
# User decision about optimizing the resume
graph.add_edge("aggregate_analyses", "human_in_loop")
graph.add_conditional_edges("human_in_loop", route_based_on_choice)
graph.add_edge("generate_suggestions", "apply_suggestions")
graph.add_edge("apply_suggestions", "parse_resume")  # Re-parse with new content

graph.add_edge("aggregate_analyses", "human_in_loop")  # This creates the feedback loop
graph.add_edge("error_handler", END)

checkpointer = MemorySaver()
# End after aggregation
compiled_graph = graph.compile(checkpointer=checkpointer)
graph_return = prepare_documents_for_job("123", "456", compiled_graph)
# print("graph_return???: ", graph_return)
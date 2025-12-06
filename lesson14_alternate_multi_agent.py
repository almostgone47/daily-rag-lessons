"""
OPTIMIZED Multi-Agent Workflow with Prompt Stuffing & Section-by-Section Optimization

Key Optimizations:
1. Single comprehensive analysis (prompt stuffing) instead of 3 parallel LLM calls
   - Reduces analysis time from 8-16s to 2-3s
2. Section-by-section optimization instead of entire resume at once
   - Faster per-step, better UX, more focused suggestions
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add
import re
import json
import os
import time
from groq import Groq
from dotenv import load_dotenv

from lesson11_parallel_nodes import parse_resume, merge_dicts
from lesson12_human_in_loop import apply_suggestions, error_handler
from helpers import load_resume_data, load_job_description, parse_sections

# Load environment variables and initialize Groq client
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ResumeState with additional fields for section-by-section optimization
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
    "errors": Annotated[List[str], add],
    # Additional fields for multi-agent workflow
    "formatting_state": str,
    "formatting_issues": List[str],
    "is_properly_formatted": bool,
    "conflicts_resolved": bool,
    "conflict_messages": List[str],
    "resolved_suggestions": List[str],
    "workflow_complete": bool,
    # Section-by-section optimization
    "current_section": str,  # Which section we're currently optimizing (e.g., "Experience", "Summary", "Skills")
    "optimized_sections": List[str],  # List of sections that have been optimized
})

# ============================================================================
# OPTIMIZED ANALYSIS: Single LLM call with prompt stuffing
# ============================================================================

def analyze_resume_comprehensive(state: ResumeState):
    """
    OPTIMIZED: Single LLM call that does ATS score, skill gap, AND relevant experience.
    This replaces 3 separate parallel calls, reducing time from 8-16s to 2-3s.
    """
    start_time = time.time()
    
    try:
        job_desc = state.get("job_description", "")
        if not job_desc:
            return {"errors": ["No job description provided"]}
        
        resume_sections = state.get("extracted_sections", [])
        resume_skills = state.get("extracted_skills", [])
        resume_text = " ".join(resume_sections).lower()
        
        # Build comprehensive prompt that asks for all three analyses at once
        prompt = f"""Analyze this resume against the job description and return a JSON object with three metrics:

1. ATS_SCORE (0-100): Calculate the percentage of job keywords found in the resume.
   - Extract keywords from the job description (technical skills, tools, frameworks)
   - Count how many of those keywords appear in the resume
   - Score = (matched_keywords / total_keywords) * 100

2. MISSING_SKILLS: List all skills/tools/technologies mentioned in the job description that are NOT found in the resume.
   - Compare job requirements to resume skills and content
   - Return as a list of strings

3. RELEVANT_EXPERIENCE: Write a 2-3 sentence summary of the most relevant work experiences that match the job requirements.
   - Focus on experiences that align with job responsibilities
   - Highlight specific achievements or technologies mentioned

Job Description:
{job_desc[:2000]}

Resume Sections:
{chr(10).join(resume_sections[:10])}

Resume Skills:
{', '.join(resume_skills) if resume_skills else 'None listed'}

Return ONLY valid JSON in this exact format (no markdown, no explanations):
{{
  "ats_score": 75,
  "missing_skills": ["React", "TypeScript", "Docker"],
  "relevant_experience": "Summary of most relevant experiences here..."
}}"""

        # Single LLM call instead of 3!
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Deterministic
            max_tokens=512,
            stream=False
        )
        
        response_text = completion.choices[0].message.content.strip()
        
        # Parse JSON from response
        # Try to extract JSON if wrapped in markdown
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group()
        
        result = json.loads(response_text)
        
        ats_score = result.get("ats_score", 0)
        missing_skills = result.get("missing_skills", [])
        relevant_experience = result.get("relevant_experience", "No relevant experience found.")
        
        elapsed = time.time() - start_time
        
        print("\n=== Analysis Results ===")
        print(f"ATS Score: {ats_score}/100")
        print(f"Skill Gap: {len(missing_skills)} missing skills")
        print(f"Relevant Experience: {relevant_experience[:100]}...")
        print(f"\nTiming:")
        print(f"  comprehensive_analysis: {elapsed:.3f}s")
        print(f"Total: {elapsed:.3f}s")
        print(f"\n⚡ Optimized: Single analysis call (vs 3 parallel calls)")
        
        return {
            "ats_score": ats_score,
            "skill_gap_score": len(missing_skills),
            "relevant_experience": relevant_experience,
            "analysis_times": {"comprehensive_analysis": elapsed}
        }
        
    except Exception as e:
        print(f"   ❌ Comprehensive analysis failed: {e}")
        return {"errors": [f"Comprehensive analysis failed: {str(e)}"]}

# ============================================================================
# SECTION-BY-SECTION OPTIMIZATION
# ============================================================================

def get_next_section_to_optimize(state: ResumeState) -> str:
    """
    Determine which section to optimize next.
    Priority: Experience → Summary → Skills → Education
    """
    optimized = state.get("optimized_sections", [])
    resume_data = state.get("resume_data", {})
    sections = resume_data.get("sections", [])
    
    # Priority order
    priority_sections = [
        ("EXPERIENCE", "Experience"),
        ("RICHTEXT", "Summary"),  # Summary is usually RICHTEXT
        ("SKILLS", "Skills"),
        ("EDUCATION", "Education"),
    ]
    
    for kind, name in priority_sections:
        if name not in optimized:
            # Check if this section type exists in resume
            if any(s.get("kind") == kind for s in sections):
                return name
    
    return None  # All sections optimized

def generate_suggestions_for_section(state: ResumeState) -> Dict[str, Any]:
    """
    OPTIMIZED: Generate suggestions for ONE section at a time.
    This is faster and more focused than optimizing the entire resume.
    """
    print("\n💡 Optimization Agent: Generating suggestions for current section...")
    
    resume_data = state.get("resume_data", {})
    job_description = state.get("job_description", "")
    formatting_state = state.get("formatting_state", "")
    current_section = state.get("current_section", "Experience")
    
    if not resume_data:
        return {
            "user_modifications": [],
            "errors": ["No resume data available for suggestions"]
        }
    
    # Find the target section in resume data
    sections = resume_data.get("sections", [])
    target_section = None
    section_kind = None
    
    # Map section name to kind
    section_map = {
        "Experience": "EXPERIENCE",
        "Summary": "RICHTEXT",
        "Skills": "SKILLS",
        "Education": "EDUCATION",
    }
    
    section_kind = section_map.get(current_section, "EXPERIENCE")
    
    # Find the section
    for section in sections:
        if section.get("kind") == section_kind:
            target_section = section
            break
    
    if not target_section:
        print(f"   ⚠️  Section '{current_section}' not found in resume")
        return {
            "user_modifications": [],
            "errors": [f"Section '{current_section}' not found"]
        }
    
    # Extract section content for context
    section_title = target_section.get("title", current_section)
    section_content = target_section.get("content", "")
    
    # For Experience sections, include items
    if section_kind == "EXPERIENCE":
        items = target_section.get("items", [])
        section_content = f"{section_title}\n"
        for item in items[:3]:  # Limit to first 3 jobs
            section_content += f"\n{item.get('headline', '')} at {item.get('subheadline', '')}\n"
            bullets = item.get("bullets", [])
            section_content += "\n".join([f"  • {b}" for b in bullets[:5]])  # Limit bullets
    
    # Build focused prompt for this section only
    prompt = f"""You are a resume optimization assistant. Generate suggestions to improve the {current_section} section of this resume for the target job.

CRITICAL FORMATTING RULES:
{formatting_state if formatting_state else "Preserve the exact format of the original content."}

IMPORTANT: You MUST preserve the exact format:
- If original has bullet points (• or -), return bullet points
- If original is a paragraph, return a paragraph
- Match the structure, spacing, and formatting style of the original

{current_section} Section Content:
{section_content[:1500]}

Target Job Description:
{job_description[:1000]}

Instructions:
1. Analyze this {current_section} section against the job description
2. Generate 3-5 specific, actionable suggestions to improve ONLY this section
3. Each suggestion should:
   - Be specific and actionable
   - Relate to the job requirements
   - Preserve the existing format
4. Return suggestions as a JSON array: ["suggestion 1", "suggestion 2", ...]

Suggestions:"""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=512,
            stream=False
        )
        
        suggestions_text = completion.choices[0].message.content
        
        # Parse JSON from response
        suggestions = []
        json_match = re.search(r'\[.*?\]', suggestions_text, re.DOTALL)
        if json_match:
            try:
                suggestions = json.loads(json_match.group())
            except:
                suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and not s.strip().startswith('#')]
        else:
            suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and len(s.strip()) > 10]
        
        suggestions = suggestions[:5]
        
        print(f"   Generated {len(suggestions)} suggestions for {current_section} section")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion[:80]}...")
        
        return {
            "user_modifications": suggestions
        }
    
    except Exception as e:
        print(f"   Error generating suggestions: {e}")
        return {
            "user_modifications": [],
            "errors": [f"Failed to generate suggestions: {str(e)}"]
        }

# ============================================================================
# HUMAN-IN-THE-LOOP & PREPARE DOCUMENTS
# ============================================================================

def human_in_loop(state: ResumeState):
    """
    Pauses workflow for user decision.
    Shows section-by-section progress.
    """
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    user_decision = state.get("user_decision")
    ats_score = state.get("ats_score")
    workflow_complete = state.get("workflow_complete", False)
    current_section = state.get("current_section")
    optimized_sections = state.get("optimized_sections", [])
    
    # If we have analysis results but no user_decision and no optimization artifacts:
    if ats_score is not None and not user_decision and not formatting_state and not user_modifications:
        if workflow_complete:
            print("\n=== Showing Updated Results ===")
            print("Analysis results shown above. Workflow complete.")
            return {"workflow_complete": True}
        else:
            # Fresh analysis - need user decision
            print("\n=== Analysis Results ===")
            print(f"ATS Score: {ats_score}/100")
            skill_gap_score = state.get("skill_gap_score", 0)
            print(f"Skill Gap: {skill_gap_score} missing skills")
            print(f"Relevant Experience: {state.get('relevant_experience', 'N/A')[:100]}...")
            
            if optimized_sections:
                print(f"\n✓ Optimized sections: {', '.join(optimized_sections)}")
            if current_section:
                print(f"📝 Next section to optimize: {current_section}")
            
            print("\nDo you want to proceed with optimizations?")
            print("1. Yes (optimize next section)")
            print("2. No (exit)")
            interrupt("Please enter your choice")
            return {}
    
    # Otherwise, we're waiting for user to make a decision
    print("\n=== Analysis Results ===")
    print(f"ATS Score: {ats_score}/100")
    skill_gap_score = state.get("skill_gap_score", 0)
    print(f"Skill Gap: {skill_gap_score} missing skills")
    print(f"Relevant Experience: {state.get('relevant_experience', 'N/A')[:100]}...")
    
    if optimized_sections:
        print(f"\n✓ Optimized sections: {', '.join(optimized_sections)}")
    if current_section:
        print(f"📝 Next section to optimize: {current_section}")
    
    print("\nDo you want to proceed with optimizations?")
    print("1. Yes (optimize next section)")
    print("2. No (exit)")
    interrupt("Please enter your choice")
    return {}

def prepare_documents_for_job(resume_id: str, job_id: str, graph):
    """
    Simulates API endpoint: POST /api/resumes/:id/customize
    """
    resume_data = load_resume_data()
    job_description = load_job_description()

    initial_state = {
        "resume_data": resume_data,
        "job_description": job_description
    }
    
    import uuid
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(initial_state, config=config)
    current_state = graph.get_state(config=config)
    
    if current_state.next:
        import threading
        
        choice = None
        
        def get_input():
            nonlocal choice
            try:
                choice = input("Please enter your choice: ")
            except:
                pass
        
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()
        input_thread.join(timeout=30)
        
        if choice is None:
            print("\nTimeout: No response received. Defaulting to 'No' (2).")
            choice = "2"
        
        global _user_decision_from_command
        _user_decision_from_command = choice
        
        result = graph.invoke(
            Command(resume={"user_decision": choice}),
            config=config
        )

    return result

# Global variable for user decision workaround
_user_decision_from_command = None

def process_user_decision(state: ResumeState) -> Dict[str, Any]:
    """Process user decision after resume from interrupt."""
    global _user_decision_from_command
    
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        return {}
    
    user_decision = state.get("user_decision")
    if user_decision and user_decision != "pending" and user_decision in ["1", "2"]:
        return {}
    
    if _user_decision_from_command:
        decision = _user_decision_from_command
        _user_decision_from_command = None
        return {"user_decision": decision}
    
    if user_decision and user_decision != "pending":
        return {}
    else:
        return {"user_decision": "2"}

# ============================================================================
# FORMATTING AGENT (reuse from original)
# ============================================================================

def extract_formatting_state_node(state: ResumeState) -> Dict[str, Any]:
    """Extract formatting state before generating suggestions."""
    print("\n📋 Formatting Agent (Pre-check): Extracting current format state...")
    
    resume_data = state.get("resume_data", {})
    if not resume_data:
        return {"formatting_state": "No resume data"}
    
    sections = resume_data.get("sections", [])
    formatting_info = []
    
    for section in sections:
        title = section.get("title", "")
        content = section.get("content", "")
        if not content:
            continue
        
        has_bullets = bool(re.search(r'[•\-\*]', content))
        is_paragraph = '\n\n' in content or (not has_bullets and len(content) > 50)
        format_type = "bullets" if has_bullets else "paragraph" if is_paragraph else "mixed"
        formatting_info.append(f"{title}: {format_type}")
    
    formatting_state = "Current format: " + "; ".join(formatting_info) if formatting_info else "No formatting info"
    print(f"   {formatting_state}")
    
    return {"formatting_state": formatting_state}

def format_resume(state: ResumeState) -> Dict[str, Any]:
    """Formatting validation after applying suggestions."""
    print("\n📝 Formatting Agent (Post-check): Validating resume formatting...")
    
    resume_data = state.get("resume_data", {})
    if not resume_data:
        return {
            "formatting_issues": ["No resume data found"],
            "is_properly_formatted": False
        }
    
    # Simplified validation (full version in original file)
    issues = []
    sections = resume_data.get("sections", [])
    
    for section in sections:
        if section.get("kind") == "EXPERIENCE":
            items = section.get("items", [])
            for item in items:
                if not item.get("dates"):
                    issues.append(f"Experience entry '{item.get('headline', 'Unknown')}' missing dates")
    
    is_properly_formatted = len(issues) == 0
    
    if issues:
        print(f"   Found {len(issues)} formatting issues")
        for issue in issues[:3]:
            print(f"   - {issue}")
    else:
        print("   ✓ Resume formatting is correct!")
    
    return {
        "formatting_issues": issues,
        "is_properly_formatted": is_properly_formatted
    }

def resolve_conflicts(state: ResumeState) -> Dict[str, Any]:
    """Conflict resolution between Optimization and Formatting agents."""
    print("\n⚖️  Conflict Resolution: Resolving conflicts...")
    
    formatting_issues = state.get("formatting_issues", [])
    user_modifications = state.get("user_modifications", [])
    is_properly_formatted = state.get("is_properly_formatted", True)
    
    if is_properly_formatted or not formatting_issues:
        print("   ✓ No conflicts - formatting is correct!")
        return {
            "conflicts_resolved": True,
            "conflict_messages": [],
            "resolved_suggestions": user_modifications if user_modifications else []
        }
    
    # Filter suggestions that might cause formatting issues
    resolved_suggestions = user_modifications.copy() if user_modifications else []
    
    if len(resolved_suggestions) > 3:
        resolved_suggestions = resolved_suggestions[:3]
        print(f"   Filtered to top 3 suggestions to avoid formatting conflicts")
    
    print(f"   Resolved suggestions (safe to apply): {len(resolved_suggestions)}")
    
    return {
        "conflicts_resolved": True,
        "conflict_messages": formatting_issues,
        "resolved_suggestions": resolved_suggestions
    }

def clear_optimization_state(state: ResumeState) -> Dict[str, Any]:
    """Clear optimization state and mark current section as optimized."""
    current_section = state.get("current_section", "Experience")
    optimized_sections = state.get("optimized_sections", [])
    
    if current_section and current_section not in optimized_sections:
        optimized_sections = optimized_sections + [current_section]
    
    print(f"\n🧹 Clearing optimization state...")
    print(f"   ✓ Completed optimization for: {', '.join(optimized_sections)}")
    
    # Determine next section
    next_section = get_next_section_to_optimize({
        **state,
        "optimized_sections": optimized_sections
    })
    
    result = {
        "user_decision": "show_results" if not next_section else None,
        "formatting_state": None,
        "user_modifications": [],
        "formatting_issues": None,
        "conflicts_resolved": None,
        "optimized_sections": optimized_sections,
        "current_section": next_section if next_section else None,
        "workflow_complete": True if not next_section else False
    }
    
    if next_section:
        print(f"   📝 Next section to optimize: {next_section}")
    else:
        print(f"   ✓ All sections optimized!")
    
    return result

# ============================================================================
# ORCHESTRATOR
# ============================================================================

def orchestrator(state: ResumeState) -> Dict[str, Any]:
    """Orchestrator agent coordinates the workflow."""
    print("\n🎯 Orchestrator: Coordinating agents...")
    
    resume_data = state.get("resume_data")
    extracted_sections = state.get("extracted_sections")
    ats_score = state.get("ats_score")
    user_decision = state.get("user_decision")
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    formatting_issues = state.get("formatting_issues")
    conflicts_resolved = state.get("conflicts_resolved")
    errors = state.get("errors", [])
    current_section = state.get("current_section")
    
    completed_steps = []
    if resume_data:
        completed_steps.append("resume_loaded")
    if extracted_sections:
        completed_steps.append("resume_parsed")
    if ats_score is not None:
        completed_steps.append("analysis_complete")
    
    print(f"   Completed steps: {', '.join(completed_steps) if completed_steps else 'none'}")
    
    next_action = None
    reason = ""
    
    workflow_complete = state.get("workflow_complete", False)
    
    if errors:
        next_action = "error_handler"
        reason = "Errors detected"
    elif workflow_complete:
        next_action = END
        reason = "Workflow complete"
    elif user_decision and user_decision != "pending":
        if user_decision == "1":
            if not current_section:
                # Determine which section to optimize
                next_section = get_next_section_to_optimize(state)
                if next_section:
                    next_action = "set_current_section"
                    reason = f"Starting optimization for {next_section} section"
                else:
                    next_action = END
                    reason = "All sections optimized"
            elif not formatting_state:
                next_action = "extract_formatting_state"
                reason = f"Extracting formatting state for {current_section}"
            elif not user_modifications:
                next_action = "generate_suggestions"
                reason = f"Generating suggestions for {current_section}"
            elif formatting_issues is None:
                next_action = "format_resume"
                reason = "Validating formatting"
            elif conflicts_resolved is None:
                next_action = "resolve_conflicts"
                reason = "Resolving conflicts"
            else:
                next_action = "clear_optimization_state"
                reason = f"Completing optimization for {current_section}"
        elif user_decision == "2":
            next_action = END
            reason = "User chose to exit"
        else:
            next_action = END
            reason = f"Invalid user decision '{user_decision}'"
    elif ats_score is not None and (not user_decision or user_decision == "pending"):
        next_action = "human_in_loop"
        reason = "Analysis complete - waiting for user decision"
    elif not extracted_sections and resume_data and not ats_score:
        next_action = "parse_resume"
        reason = "Resume loaded but not parsed"
    else:
        next_action = END
        reason = "No clear next step"
    
    print(f"   Next action: {next_action} ({reason})")
    return {}

def set_current_section(state: ResumeState) -> Dict[str, Any]:
    """Set the current section to optimize."""
    next_section = get_next_section_to_optimize(state)
    if next_section:
        print(f"\n📝 Setting current section to optimize: {next_section}")
        return {"current_section": next_section}
    else:
        return {"workflow_complete": True}

def route_based_on_orchestrator(state: ResumeState):
    """Routing function based on orchestrator logic."""
    resume_data = state.get("resume_data")
    extracted_sections = state.get("extracted_sections")
    ats_score = state.get("ats_score")
    user_decision = state.get("user_decision")
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    formatting_issues = state.get("formatting_issues")
    conflicts_resolved = state.get("conflicts_resolved")
    errors = state.get("errors", [])
    current_section = state.get("current_section")
    
    workflow_complete = state.get("workflow_complete", False)
    
    if errors:
        return "error_handler"
    elif workflow_complete:
        return END
    elif user_decision and user_decision != "pending" and user_decision in ["1", "2"]:
        if user_decision == "1":
            if not current_section:
                return "set_current_section"
            elif not formatting_state:
                return "extract_formatting_state"
            elif not user_modifications:
                return "generate_suggestions"
            elif formatting_issues is None:
                return "format_resume"
            elif conflicts_resolved is None:
                return "resolve_conflicts"
            else:
                return "clear_optimization_state"
        elif user_decision == "2":
            return END
        else:
            return END
    elif ats_score is not None and (not user_decision or user_decision == "pending"):
        return "human_in_loop"
    elif not extracted_sections and resume_data and not ats_score:
        return "parse_resume"
    else:
        return END

def route_after_aggregate(state: ResumeState):
    """Route after comprehensive analysis."""
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        return END
    
    ats_score = state.get("ats_score")
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    user_decision = state.get("user_decision")
    
    if ats_score is not None and not formatting_state and not user_modifications:
        if user_decision == "show_results":
            return END
        elif not user_decision:
            if workflow_complete:
                return END
            else:
                return "human_in_loop"
        else:
            return "human_in_loop"
    else:
        return "human_in_loop"

def route_after_human_loop(state: ResumeState):
    """Route after human_in_loop."""
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        return END
    else:
        return "process_user_decision"

# ============================================================================
# GRAPH DEFINITION
# ============================================================================

graph = StateGraph(ResumeState)

# Resume Analyzer Agent
graph.add_node("parse_resume", parse_resume)
graph.add_node("analyze_resume_comprehensive", analyze_resume_comprehensive)  # Single optimized analysis

# Optimization Suggestion Agent (section-by-section)
graph.add_node("set_current_section", set_current_section)
graph.add_node("generate_suggestions", generate_suggestions_for_section)  # Section-specific
graph.add_node("apply_suggestions", apply_suggestions)

# Formatting Agent
graph.add_node("extract_formatting_state", extract_formatting_state_node)
graph.add_node("format_resume", format_resume)

# Orchestrator Agent
graph.add_node("orchestrator", orchestrator)
graph.add_node("human_in_loop", human_in_loop)
graph.add_node("process_user_decision", process_user_decision)
graph.add_node("clear_optimization_state", clear_optimization_state)

# Conflict Resolution Agent
graph.add_node("resolve_conflicts", resolve_conflicts)
graph.add_node("error_handler", error_handler)

# Workflow edges
graph.add_edge(START, "parse_resume")
graph.add_edge("parse_resume", "analyze_resume_comprehensive")  # Single analysis instead of 3 parallel
graph.add_conditional_edges(
    "analyze_resume_comprehensive",
    route_after_aggregate,
    {
        "human_in_loop": "human_in_loop",
        END: END
    }
)

graph.add_conditional_edges(
    "human_in_loop",
    route_after_human_loop,
    {
        "process_user_decision": "process_user_decision",
        END: END
    }
)

graph.add_edge("process_user_decision", "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route_based_on_orchestrator,
    {
        "set_current_section": "set_current_section",
        "extract_formatting_state": "extract_formatting_state",
        "generate_suggestions": "generate_suggestions",
        "format_resume": "format_resume",
        "resolve_conflicts": "resolve_conflicts",
        "clear_optimization_state": "clear_optimization_state",
        "parse_resume": "parse_resume",
        "human_in_loop": "human_in_loop",
        "error_handler": "error_handler",
        END: END
    }
)

# Optimization flow
graph.add_edge("set_current_section", "extract_formatting_state")
graph.add_edge("extract_formatting_state", "generate_suggestions")
graph.add_edge("generate_suggestions", "apply_suggestions")
graph.add_edge("apply_suggestions", "format_resume")
graph.add_edge("format_resume", "resolve_conflicts")
graph.add_edge("resolve_conflicts", "clear_optimization_state")
graph.add_edge("clear_optimization_state", "parse_resume")  # Re-parse to get updated scores
graph.add_edge("error_handler", END)

checkpointer = MemorySaver()

if __name__ == "__main__":
    compiled_graph = graph.compile(checkpointer=checkpointer)
    graph_return = prepare_documents_for_job("123", "456", compiled_graph)


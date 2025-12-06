from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from typing import TypedDict, List, Dict, Any, Annotated
from operator import add
import re
import json

from lesson11_parallel_nodes import parse_resume, check_ats_score, check_skill_gap_score, check_relevant_experience, aggregate_analyses
from lesson11_parallel_nodes import merge_dicts

# Extend ResumeState to include additional fields for multi-agent workflow
# TypedDict doesn't support inheritance, so we recreate it with all fields
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
})
from lesson12_human_in_loop import apply_suggestions, error_handler
from langgraph.types import interrupt, Command
from helpers import load_resume_data, load_job_description

def human_in_loop(state: ResumeState):
    """
    Pauses workflow for user decision.
    Only prompts if we're waiting for a NEW decision (not just showing results).
    """
    # Check if we've already completed an optimization cycle
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    user_decision = state.get("user_decision")
    ats_score = state.get("ats_score")
    workflow_complete = state.get("workflow_complete", False)
    
    # If we have analysis results but no user_decision and no optimization artifacts:
    # - If workflow_complete is True: we're showing results after optimization
    # - If workflow_complete is False: this is a fresh analysis - need user decision
    if ats_score is not None and not formatting_state and not user_modifications:
        # Also check user_decision="show_results"
        if workflow_complete or user_decision == "show_results":
            # Showing results after optimization
            print("\n=== Showing Updated Results ===")
            print("Analysis results shown above. Workflow complete.")
            print("(To optimize again, you would need to restart the workflow)")
            return {"workflow_complete": True}
        elif not user_decision:
            # Fresh analysis - need user decision
            print("\n=== Analysis Results ===")
            print(f"ATS Score: {ats_score}/100")
            skill_gap_score = state.get("skill_gap_score", 0)
            print(f"Skill Gap: {skill_gap_score} missing skills")
            print(f"Relevant Experience: {state.get('relevant_experience', 'N/A')[:100]}...")
            print("\nDo you want to proceed with optimizations?")
            print("1. Yes")
            print("2. No")
            interrupt("Please enter your choice")
            return {}
    
    # Otherwise, we're waiting for user to make a decision about optimizing
    print("\n=== Analysis Results ===")
    print(f"ATS Score: {ats_score}/100")
    skill_gap_score = state.get("skill_gap_score", 0)
    print(f"Skill Gap: {skill_gap_score} missing skills")
    print(f"Relevant Experience: {state.get('relevant_experience', 'N/A')[:100]}...")
    print("\nDo you want to proceed with optimizations?")
    print("1. Yes")
    print("2. No")
    interrupt("Please enter your choice")
    return {}

def prepare_documents_for_job(resume_id: str, job_id: str, graph):
    """
    Simulates API endpoint: POST /api/resumes/:id/customize
    Fixed version with correct Command usage for state updates.
    """
    # Simulate loading from database
    resume_data = load_resume_data()
    job_description = load_job_description()

    # Invoke the workflow
    initial_state = {
        "resume_data": resume_data,
        "job_description": job_description
    }
    config = {"configurable": {"thread_id": "1"}}
    # print(f"\n{'='*60}")
    # print(f"[DEBUG] prepare_documents_for_job: Starting graph.invoke()")
    # print(f"[DEBUG] Initial state keys: {list(initial_state.keys())}")
    # print(f"{'='*60}")
    result = graph.invoke(initial_state, config=config)
    current_state = graph.get_state(config=config)
    # print(f"\n[DEBUG] After first invoke:")
    # print(f"   - current_state.next = {current_state.next}")
    # print(f"   - State values keys: {list(current_state.values.keys()) if current_state.values else 'None'}")
    # print(f"   - ats_score in result: {result.get('ats_score')}")
    
    if current_state.next:
        import threading
        
        choice = None
        
        def get_input():
            nonlocal choice
            try:
                choice = input("Please enter your choice: ")
            except:
                pass
        
        # Start input thread
        input_thread = threading.Thread(target=get_input)
        input_thread.daemon = True
        input_thread.start()
        
        # Wait for input or timeout
        input_thread.join(timeout=30)
        
        if choice is None:
            print("\nTimeout: No response received. Defaulting to 'No' (2).")
            choice = "2"
        
        # WORKAROUND: Command(resume={...}) should update state, but it's not working
        # Store the user_decision globally so process_user_decision node can set it
        global _user_decision_from_command
        _user_decision_from_command = choice
        
        print(f"\n[DEBUG] Resuming with user_decision: {choice}")
        print(f"[DEBUG] Stored in global: {_user_decision_from_command}")
        print(f"[DEBUG] State before resume: {current_state.values.get('user_decision') if current_state.values else 'None'}")
        
        # Try Command - it should update state, but we have a workaround if it doesn't
        # print(f"\n{'='*60}")
        # print(f"[DEBUG] Resuming with Command(resume={{'user_decision': '{choice}'}})")
        # print(f"{'='*60}")
        result = graph.invoke(
            Command(resume={"user_decision": choice}),
            config=config
        )
        
        # Check what happened
        updated_state = graph.get_state(config=config)
        state_user_decision = updated_state.values.get("user_decision") if updated_state.values else None
        # print(f"\n[DEBUG] After Command resume:")
        # print(f"   - user_decision in result: {result.get('user_decision')}")
        # print(f"   - user_decision in state.values: {state_user_decision}")
        # print(f"   - ats_score in result: {result.get('ats_score')}")
        
        # If Command worked, great! If not, process_user_decision will set it from global

    return result
from helpers import parse_sections
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables and initialize Groq client
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def orchestrator(state: ResumeState) -> Dict[str, Any]:
    """
    ============================================================================
    ORCHESTRATOR AGENT: The "Brain" of the Multi-Agent System
    ============================================================================
    
    What is an Orchestrator?
    -------------------------
    An orchestrator is a special agent that doesn't do the actual work (like
    analyzing resumes or generating suggestions). Instead, it COORDINATES all
    the other agents - deciding which agent should run next, when, and in what order.
    
    Think of it like a project manager:
    - Resume Analyzer Agent = does the analysis work
    - Optimization Agent = does the optimization work  
    - Formatting Agent = does the formatting checks
    - Orchestrator = decides when each agent should work
    
    How It Works:
    -------------
    1. Checks the current state to see what's been completed
    2. Determines which agent should run next based on:
       - Dependencies (e.g., can't format until suggestions are applied)
       - User decisions (e.g., user chose to optimize)
       - Error conditions (e.g., errors detected)
    3. Routes to the appropriate next agent
    
    Decision Logic (Priority Order):
    --------------------------------
    1. Errors? → Route to error_handler
    2. User decision made?
       - "1" (optimize) → Check what step we're on:
         * No formatting state? → extract_formatting_state
         * No suggestions? → generate_suggestions
         * No formatting check? → format_resume
         * No conflict resolution? → resolve_conflicts
         * All done? → re-parse resume
       - "2" (exit) → END
    3. Analysis complete but no user decision? → human_in_loop
    4. Resume loaded but not parsed? → parse_resume
    5. Default → END
    
    Why Use an Orchestrator?
    ------------------------
    - Centralized control: All routing logic in one place
    - Dependency management: Ensures agents run in correct order
    - Error handling: Can catch and route errors appropriately
    - Flexibility: Easy to change workflow without modifying individual agents
    - Debugging: One place to see what's happening in the system
    
    Example Flow:
    ------------
    Parse → Analyze → Orchestrator → "Analysis done, need user input" → Human Loop
    Human Loop → User chooses "1" → Orchestrator → "User wants optimize, extract format" → Formatting Agent
    Formatting Agent → Orchestrator → "Format extracted, generate suggestions" → Optimization Agent
    ... and so on
    """
    print("\n🎯 Orchestrator: Coordinating agents...")
    
    # Check what's been completed
    resume_data = state.get("resume_data")
    extracted_sections = state.get("extracted_sections")
    ats_score = state.get("ats_score")
    user_decision = state.get("user_decision")
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    formatting_issues = state.get("formatting_issues")
    conflicts_resolved = state.get("conflicts_resolved")
    errors = state.get("errors", [])
    
    # Track what's been done
    completed_steps = []
    if resume_data:
        completed_steps.append("resume_loaded")
    if extracted_sections:
        completed_steps.append("resume_parsed")
    if ats_score is not None:
        completed_steps.append("analysis_complete")
    if formatting_state:
        completed_steps.append("formatting_state_extracted")
    if user_modifications:
        completed_steps.append("suggestions_generated")
    if formatting_issues is not None:
        completed_steps.append("formatting_validated")
    if conflicts_resolved is not None:
        completed_steps.append("conflicts_resolved")
    
    print(f"   Completed steps: {', '.join(completed_steps) if completed_steps else 'none'}")
    
    # Decision logic: What should happen next?
    next_action = None
    reason = ""
    
    # Priority 1: Handle errors
    if errors:
        next_action = "error_handler"
        reason = "Errors detected - routing to error handler"
    
    # Priority 2: Check if user has made a decision
    elif user_decision and user_decision != "pending":
        if user_decision == "1":  # User wants to optimize
            if not formatting_state:
                next_action = "extract_formatting_state"
                reason = "User chose to optimize - extracting formatting state first"
            elif not user_modifications:
                next_action = "generate_suggestions"
                reason = "Formatting state extracted - generating suggestions"
            elif formatting_issues is None:
                next_action = "format_resume"
                reason = "Suggestions applied - validating formatting"
            elif conflicts_resolved is None:
                next_action = "resolve_conflicts"
                reason = "Formatting validated - resolving conflicts"
            else:
                # All optimization steps complete - show results to user (don't loop)
                next_action = "human_in_loop"
                reason = "Optimization complete - showing updated results"
        elif user_decision == "2":  # User wants to exit
            next_action = END
            reason = "User chose to exit"
        else:
            next_action = END
            reason = f"Invalid user decision '{user_decision}' - ending workflow"
    
    # Priority 3: Check if workflow is complete (after showing results)
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        next_action = END
        reason = "Workflow complete - ending"
    
    # Priority 4: Check if analysis is complete but no user decision yet
    elif ats_score is not None and (not user_decision or user_decision == "pending" or user_decision is None):
        next_action = "human_in_loop"
        reason = "Analysis complete - waiting for user decision"
    
    # Priority 5: Check if resume needs parsing
    # BUT: Don't re-parse if we already have analysis results and are waiting for user decision
    # This prevents loops when resuming from interrupt
    elif not extracted_sections and resume_data and not ats_score:
        next_action = "parse_resume"
        reason = "Resume loaded but not parsed - starting analysis"
    
    # Default: End workflow
    else:
        next_action = END
        reason = "No clear next step - ending workflow"
    
    print(f"   Next action: {next_action} ({reason})")
    
    # Return state update (orchestrator doesn't modify state, just coordinates)
    return {}

# Global variable to store user decision (workaround for Command not updating state)
_user_decision_from_command = None

def process_user_decision(state: ResumeState) -> Dict[str, Any]:
    """
    Process user decision after resume from interrupt.
    WORKAROUND: Command(resume={...}) isn't updating state correctly.
    We use a global variable to pass the user_decision from prepare_documents_for_job
    to this node, then set it in state.
    """
    global _user_decision_from_command
    
    # Check if workflow is already complete - if so, just pass through
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        # print("\n[DEBUG] process_user_decision: Workflow already complete, passing through")
        return {}
    
    # Check if user_decision is already set in state (from Command or previous call)
    user_decision = state.get("user_decision")
    if user_decision and user_decision != "pending" and user_decision in ["1", "2"]:
        # print(f"\n[DEBUG] process_user_decision: user_decision already set in state: {user_decision}")
        return {}  # Already set, pass through
    
    # If we have a user decision from Command (stored globally), use it
    if _user_decision_from_command:
        # print(f"\n[DEBUG] process_user_decision: Using user_decision from global: {_user_decision_from_command}")
        decision = _user_decision_from_command
        _user_decision_from_command = None  # Clear after use
        return {"user_decision": decision}
    
    # Otherwise, check if it's in state (Command might have worked)
    # print(f"\n[DEBUG] process_user_decision: user_decision from state: {user_decision}")
    
    if user_decision and user_decision != "pending":
        # Already set, just pass through
        return {}
    else:
        # Not set - Command didn't work, but we don't have the value here
        # This shouldn't happen if we set _user_decision_from_command correctly
        # print(f"   ⚠️  WARNING: user_decision not set! Command may have failed.")
        # Default to "2" (exit) to prevent infinite loop
        # print(f"   [DEBUG] Defaulting to '2' (exit) to prevent loop")
        return {"user_decision": "2"}

def clear_optimization_state(state: ResumeState) -> Dict[str, Any]:
    """
    Clear optimization-related state after showing results.
    This prevents loops and allows user to start fresh optimization if desired.
    Set workflow_complete flag so we know to end after showing results.
    Also set user_decision to a special value to indicate we're showing results.
    """
    print("\n🧹 Clearing optimization state for fresh start...")
    print(f"[DEBUG] clear_optimization_state: Setting workflow_complete = True")
    # Use a special user_decision value to indicate we're showing results after optimization
    # This is more reliable than workflow_complete since it's in the original TypedDict
    result = {
        "user_decision": "show_results",  # Special value to indicate showing results after optimization
        "formatting_state": None,  # Will be re-extracted if user optimizes again
        "user_modifications": [],  # Clear suggestions
        "formatting_issues": None,  # Will be re-validated if user optimizes again
        "conflicts_resolved": None,  # Will be re-resolved if user optimizes again
        "workflow_complete": True  # Set flag so we know to end after showing results
    }
    print(f"[DEBUG] clear_optimization_state: Returning user_decision = 'show_results', workflow_complete = {result.get('workflow_complete')}")
    return result

def route_based_on_orchestrator(state: ResumeState):
    """
    Routing function that uses orchestrator logic.
    This is called by conditional edges to determine next node.
    """
    # Get orchestrator's decision
    resume_data = state.get("resume_data")
    extracted_sections = state.get("extracted_sections")
    ats_score = state.get("ats_score")
    user_decision = state.get("user_decision")
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    formatting_issues = state.get("formatting_issues")
    conflicts_resolved = state.get("conflicts_resolved")
    errors = state.get("errors", [])
    
    # Debug: Print what we're seeing
    print(f"\n   [DEBUG] user_decision: {repr(user_decision)}, formatting_state: {bool(formatting_state)}, user_modifications: {bool(user_modifications)}")
    
    # Same logic as orchestrator, but return node name directly
    if errors:
        print("   [DEBUG] Routing to error_handler")
        return "error_handler"
    elif user_decision and user_decision != "pending" and user_decision is not None:
        print(f"   [DEBUG] User decision is: {user_decision}")
        if user_decision == "1":
            if not formatting_state:
                print("   [DEBUG] Routing to extract_formatting_state")
                return "extract_formatting_state"
            elif not user_modifications:
                print("   [DEBUG] Routing to generate_suggestions")
                return "generate_suggestions"
            elif formatting_issues is None:
                print("   [DEBUG] Routing to format_resume")
                return "format_resume"
            elif conflicts_resolved is None:
                print("   [DEBUG] Routing to resolve_conflicts")
                return "resolve_conflicts"
            else:
                # All done - clear user_decision to prevent loop, show results
                print("   [DEBUG] All optimization complete, showing results")
                # Return a special marker that we'll handle
                return "human_in_loop"
        elif user_decision == "2":
            print("   [DEBUG] User chose to exit")
            return END
        else:
            print(f"   [DEBUG] Invalid user_decision: {user_decision}")
            return END
    # Check if workflow is complete first
    workflow_complete = state.get("workflow_complete", False)
    if workflow_complete:
        print("   [DEBUG] Workflow complete, ending")
        return END
    elif ats_score is not None and (not user_decision or user_decision == "pending" or user_decision is None):
        print("   [DEBUG] Analysis complete, waiting for user decision")
        return "human_in_loop"
    # Don't re-parse if we already have analysis results (prevents loops)
    elif not extracted_sections and resume_data and not ats_score:
        print("   [DEBUG] Resume needs parsing")
        return "parse_resume"
    else:
        print("   [DEBUG] Default: ending workflow")
        return END
import re
from typing import Set

# ============================================================================
# FORMATTING AGENT: Comprehensive Resume Formatting Checks
# ============================================================================

def check_ats_friendly_formatting(resume_data: Dict[str, Any]) -> List[str]:
    """Check for ATS-friendly formatting issues."""
    issues = []
    
    # Check for complex tables/graphics (would be in content_rich structure)
    # Check for standard fonts (can't really check without PDF, but note it)
    # Check date formats
    date_patterns = [
        r'\d{1,2}/\d{4}',  # MM/YYYY
        r'\d{4}-\d{2}',     # YYYY-MM
        r'[A-Z][a-z]+ \d{4}',  # Month YYYY
    ]
    
    # Check section headers
    required_sections = {'Summary', 'Experience', 'Education', 'Skills'}
    sections_found = set()
    
    for section in resume_data.get('sections', []):
        title = section.get('title', '').strip()
        if title:
            sections_found.add(title)
        
        # Check bullet points
        content = section.get('content', '')
        if content:
            # Check for special characters that aren't standard bullets
            if re.search(r'[→▶▸▹►]', content):
                issues.append(f"Section '{title}': Non-standard bullet characters found (use • or -)")
    
    # Check for missing required sections
    missing = required_sections - sections_found
    if missing:
        issues.append(f"Missing required sections: {', '.join(missing)}")
    
    return issues

def check_format_preservation(resume_data: Dict[str, Any]) -> List[str]:
    """Check that format is preserved (bullets stay bullets, paragraphs stay paragraphs)."""
    issues = []
    
    for section in resume_data.get('sections', []):
        title = section.get('title', '')
        content = section.get('content', '')
        content_rich = section.get('content_rich', {})
        
        # Check if content_rich structure matches content format
        if content_rich and content:
            # Check for list items in content_rich
            has_list_items = False
            if 'root' in content_rich:
                root = content_rich['root']
                if 'children' in root:
                    for child in root.get('children', []):
                        if child.get('type') == 'list' or child.get('type') == 'listitem':
                            has_list_items = True
                            break
            
            # If content_rich has lists but content doesn't have bullets, that's a mismatch
            if has_list_items and not re.search(r'[•\-\*]', content):
                issues.append(f"Section '{title}': Format mismatch - content_rich has lists but content doesn't show bullets")
    
    return issues

def check_resume_structure(resume_data: Dict[str, Any]) -> List[str]:
    """Check resume structure and completeness."""
    issues = []
    
    # Check contact information
    contact = resume_data.get('contact', {})
    if not contact.get('name'):
        issues.append("Missing contact name")
    if not contact.get('email'):
        issues.append("Missing contact email")
    
    # Check email format
    email = contact.get('email', '')
    if email and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email):
        issues.append(f"Invalid email format: {email}")
    
    # Check section ordering and completeness
    sections = resume_data.get('sections', [])
    if not sections:
        issues.append("No sections found in resume")
    
    # Check Experience section has bullets
    for section in sections:
        if section.get('kind') == 'EXPERIENCE':
            items = section.get('items', [])
            if not items:
                issues.append("Experience section has no job entries")
            else:
                for item in items:
                    bullets = item.get('bullets', [])
                    if not bullets and item.get('meta', {}).get('entry_rich'):
                        # Check if entry_rich has content but no bullets
                        entry_rich = item.get('meta', {}).get('entry_rich', {})
                        if entry_rich and not bullets:
                            issues.append(f"Experience entry '{item.get('headline', 'Unknown')}' has rich content but no bullets")
    
    return issues

def check_content_quality(resume_data: Dict[str, Any]) -> List[str]:
    """Check content quality standards."""
    issues = []
    
    for section in resume_data.get('sections', []):
        title = section.get('title', '')
        content = section.get('content', '')
        
        if not content:
            continue
        
        # Check for first-person pronouns
        first_person = re.findall(r'\b(I|me|my|myself)\b', content, re.IGNORECASE)
        if first_person:
            issues.append(f"Section '{title}': Contains first-person pronouns: {', '.join(set(first_person))}")
        
        # Check bullets start with action verbs (for Experience section)
        if section.get('kind') == 'EXPERIENCE':
            items = section.get('items', [])
            for item in items:
                bullets = item.get('bullets', [])
                for bullet in bullets:
                    if bullet:
                        # Common action verbs
                        action_verbs = ['developed', 'created', 'built', 'implemented', 'designed', 
                                       'managed', 'led', 'improved', 'optimized', 'increased', 
                                       'reduced', 'achieved', 'delivered', 'launched', 'established']
                        first_word = bullet.split()[0].lower() if bullet.split() else ''
                        if first_word not in action_verbs and len(bullet.split()) > 3:
                            # Not a critical issue, just a suggestion
                            pass
        
        # Check bullet length (should be 1-2 lines, max 2-3)
        if section.get('kind') == 'EXPERIENCE':
            items = section.get('items', [])
            for item in items:
                bullets = item.get('bullets', [])
                for bullet in bullets:
                    if bullet and len(bullet) > 200:  # Very long bullet
                        issues.append(f"Section '{title}': Bullet too long (over 200 chars) - keep bullets concise")
    
    return issues

def check_section_specific(resume_data: Dict[str, Any]) -> List[str]:
    """Check section-specific requirements."""
    issues = []
    
    sections = resume_data.get('sections', [])
    
    for section in sections:
        title = section.get('title', '')
        kind = section.get('kind', '')
        content = section.get('content', '')
        items = section.get('items', [])
        
        # Summary checks
        if title.lower() == 'summary' or kind == 'RICHTEXT':
            if content:
                # Should be single paragraph
                paragraphs = content.split('\n\n')
                if len(paragraphs) > 1:
                    issues.append(f"Summary section: Should be a single paragraph, found {len(paragraphs)} paragraphs")
        
        # Experience checks
        if kind == 'EXPERIENCE':
            for item in items:
                if not item.get('headline'):
                    issues.append(f"Experience entry missing headline")
                if not item.get('subheadline'):
                    issues.append(f"Experience entry missing company/subheadline")
                if not item.get('dates'):
                    issues.append(f"Experience entry '{item.get('headline', 'Unknown')}' missing dates")
        
        # Education checks
        if kind == 'EDUCATION':
            for item in items:
                if not item.get('headline'):  # School name
                    issues.append("Education entry missing school name")
                if not item.get('subheadline'):  # Degree
                    issues.append("Education entry missing degree")
        
        # Skills checks
        if title.lower() == 'skills' or kind == 'SKILLS':
            if items:
                for item in items:
                    # Skills should be organized list, not paragraphs
                    if item.get('content') and len(item.get('content', '')) > 100:
                        issues.append(f"Skills section: Should be a list, not paragraphs")
    
    return issues

def check_consistency(resume_data: Dict[str, Any]) -> List[str]:
    """Check formatting consistency across resume."""
    issues = []
    
    sections = resume_data.get('sections', [])
    
    # Check date format consistency
    date_formats_found = set()
    for section in sections:
        if section.get('kind') == 'EXPERIENCE' or section.get('kind') == 'EDUCATION':
            for item in section.get('items', []):
                dates = item.get('dates', '')
                if dates:
                    if re.search(r'\d{1,2}/\d{4}', dates):
                        date_formats_found.add('MM/YYYY')
                    elif re.search(r'\d{4}-\d{2}', dates):
                        date_formats_found.add('YYYY-MM')
                    elif re.search(r'[A-Z][a-z]+ \d{4}', dates):
                        date_formats_found.add('Month YYYY')
    
    if len(date_formats_found) > 1:
        issues.append(f"Inconsistent date formats found: {', '.join(date_formats_found)} - use one format consistently")
    
    # Check bullet point style consistency
    bullet_styles = set()
    for section in sections:
        content = section.get('content', '')
        if '•' in content:
            bullet_styles.add('•')
        if '-' in content and re.search(r'^\s*-\s+', content, re.MULTILINE):
            bullet_styles.add('-')
        if '*' in content and re.search(r'^\s*\*\s+', content, re.MULTILINE):
            bullet_styles.add('*')
    
    if len(bullet_styles) > 1:
        issues.append(f"Mixed bullet point styles found: {bullet_styles} - use one style consistently")
    
    return issues

def check_professional_standards(resume_data: Dict[str, Any]) -> List[str]:
    """Check professional standards."""
    issues = []
    
    contact = resume_data.get('contact', {})
    
    # Check email professionalism
    email = contact.get('email', '')
    if email:
        unprofessional_patterns = ['hotmail', 'yahoo', 'aol', 'gmail']  # These are fine, but check for obviously unprofessional
        if re.search(r'(test|example|fake|temp)', email, re.IGNORECASE):
            issues.append(f"Unprofessional email address: {email}")
    
    # Check URL formatting
    for key in ['github', 'linkedin', 'portfolio']:
        url = contact.get(key, '')
        if url and not url.startswith('http'):
            # Not necessarily an issue, but note it
            pass
    
    # Check resume length (rough estimate)
    sections = resume_data.get('sections', [])
    total_content_length = 0
    for section in sections:
        content = section.get('content', '')
        total_content_length += len(content)
        for item in section.get('items', []):
            bullets = item.get('bullets', [])
            for bullet in bullets:
                total_content_length += len(bullet)
    
    # Rough estimate: 1 page ≈ 2000-2500 characters, 2 pages ≈ 4000-5000
    if total_content_length > 5000:
        issues.append(f"Resume may be too long (estimated {total_content_length} characters, ~{total_content_length/2500:.1f} pages)")
    
    return issues

def extract_formatting_state(resume_data: Dict[str, Any]) -> str:
    """
    Lightweight formatting state extraction (BEFORE suggestions).
    Describes current format without validation - used to inform Optimization Agent.
    Returns a string description of the current formatting state.
    """
    formatting_info = []
    
    sections = resume_data.get('sections', [])
    
    for section in sections:
        title = section.get('title', '')
        kind = section.get('kind', '')
        content = section.get('content', '')
        
        if not content:
            continue
        
        # Describe format type
        has_bullets = bool(re.search(r'[•\-\*]', content))
        is_paragraph = '\n\n' in content or (not has_bullets and len(content) > 50)
        
        format_type = "bullets" if has_bullets else "paragraph" if is_paragraph else "mixed"
        
        formatting_info.append(f"{title}: {format_type}")
        
        # For Experience, note bullet count
        if kind == 'EXPERIENCE':
            items = section.get('items', [])
            for item in items:
                bullets = item.get('bullets', [])
                if bullets:
                    formatting_info.append(f"  {item.get('headline', 'Job')}: {len(bullets)} bullets")
    
    return "Current format: " + "; ".join(formatting_info) if formatting_info else "No formatting info available"

def extract_formatting_state_node(state: ResumeState) -> Dict[str, Any]:
    """
    Lightweight node: Extract formatting state before generating suggestions.
    This informs the Optimization Agent about current formatting.
    """
    print("\n📋 Formatting Agent (Pre-check): Extracting current format state...")
    
    resume_data = state.get("resume_data", {})
    if not resume_data:
        return {"formatting_state": "No resume data"}
    
    formatting_state = extract_formatting_state(resume_data)
    print(f"   {formatting_state}")
    
    return {
        "formatting_state": formatting_state
    }

def format_resume(state: ResumeState) -> Dict[str, Any]:
    """
    Formatting Agent: Comprehensive resume formatting validation (AFTER applying suggestions).
    Checks all formatting requirements and returns issues found.
    This is the full validation that catches issues introduced during optimization.
    """
    print("\n📝 Formatting Agent (Post-check): Validating resume formatting...")
    
    resume_data = state.get("resume_data", {})
    if not resume_data:
        return {
            "formatting_issues": ["No resume data found"],
            "is_properly_formatted": False
        }
    
    all_issues = []
    
    # Run all formatting checks
    all_issues.extend(check_ats_friendly_formatting(resume_data))
    all_issues.extend(check_format_preservation(resume_data))
    all_issues.extend(check_resume_structure(resume_data))
    all_issues.extend(check_content_quality(resume_data))
    all_issues.extend(check_section_specific(resume_data))
    all_issues.extend(check_consistency(resume_data))
    all_issues.extend(check_professional_standards(resume_data))
    
    is_properly_formatted = len(all_issues) == 0
    
    if all_issues:
        print(f"   Found {len(all_issues)} formatting issues")
        for issue in all_issues[:5]:  # Show first 5
            print(f"   - {issue}")
        if len(all_issues) > 5:
            print(f"   ... and {len(all_issues) - 5} more issues")
    else:
        print("   ✓ Resume formatting is correct!")
    
    return {
        "formatting_issues": all_issues,
        "is_properly_formatted": is_properly_formatted
    }

def generate_suggestions(state: ResumeState) -> Dict[str, Any]:
    """
    Optimization Suggestion Agent: Generates optimization suggestions.
    Now uses formatting_state from Formatting Agent to preserve format.
    """
    print("\n💡 Optimization Agent: Generating suggestions...")
    
    resume_data = state.get("resume_data", {})
    job_description = state.get("job_description", "")
    formatting_state = state.get("formatting_state", "")
    
    if not resume_data:
        return {
            "user_modifications": [],
            "errors": ["No resume data available for suggestions"]
        }
    
    # Get resume sections for context
    sections = parse_sections(resume_data)
    resume_context = "\n".join(sections[:5])  # Use first 5 sections
    
    # Build prompt that includes formatting state
    prompt = f"""You are a resume optimization assistant. Generate suggestions to improve this resume for the target job.

CRITICAL FORMATTING RULES:
{formatting_state if formatting_state else "No formatting constraints specified - preserve existing format."}

IMPORTANT: You MUST preserve the exact format of the original content:
- If original has bullet points (• or -), return bullet points
- If original is a paragraph, return a paragraph
- If original has numbered lists, return numbered lists
- Match the structure, spacing, and formatting style of the original

Resume Content:
{resume_context}

Target Job Description:
{job_description[:1000]}  # Limit to first 1000 chars

Instructions:
1. Analyze the resume against the job description
2. Generate 3-5 specific, actionable suggestions to improve the resume
3. Each suggestion should:
   - Be specific and actionable
   - Relate to the job requirements
   - Preserve the existing format (see formatting rules above)
4. Return suggestions as a JSON array: ["suggestion 1", "suggestion 2", ...]

Suggestions:"""
    
    try:
        # Call Groq API directly with custom prompt
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=512,
            stream=False
        )
        suggestions_text = completion.choices[0].message.content
        
        # Try to parse JSON from response (LLM might return JSON or plain text)
        suggestions = []
        
        # Try to extract JSON array from response
        json_match = re.search(r'\[.*?\]', suggestions_text, re.DOTALL)
        if json_match:
            try:
                suggestions = json.loads(json_match.group())
            except:
                # If JSON parsing fails, split by lines or bullets
                suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and not s.strip().startswith('#')]
        else:
            # Fallback: split by lines
            suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip() and len(s.strip()) > 10]
        
        # Limit to 5 suggestions
        suggestions = suggestions[:5]
        
        print(f"   Generated {len(suggestions)} suggestions")
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

def resolve_conflicts(state: ResumeState) -> Dict[str, Any]:
    """
    Conflict Resolution: Resolves conflicts between Optimization Agent and Formatting Agent.
    
    Strategy: Formatting Agent wins (prioritize formatting rules)
    
    Process:
    1. Identifies conflicts between optimization suggestions and formatting rules
    2. Filters out suggestions that violate formatting rules
    3. Modifies remaining suggestions to comply with formatting
    4. Returns resolved suggestions that can be safely applied
    """
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
    
    # Identify conflict types
    length_conflicts = []
    format_preservation_conflicts = []
    consistency_conflicts = []
    other_conflicts = []
    
    for issue in formatting_issues:
        issue_lower = issue.lower()
        
        if "too long" in issue_lower or "exceed" in issue_lower or "length" in issue_lower:
            length_conflicts.append(issue)
        elif "format mismatch" in issue_lower or ("format" in issue_lower and "preserve" in issue_lower):
            format_preservation_conflicts.append(issue)
        elif "inconsistent" in issue_lower:
            consistency_conflicts.append(issue)
        else:
            other_conflicts.append(issue)
    
    # Start with all suggestions
    resolved_suggestions = user_modifications.copy() if user_modifications else []
    filtered_count = 0
    modified_count = 0
    
    # Strategy 1: Filter suggestions that cause length conflicts
    if length_conflicts:
        print(f"   Length conflicts detected: {len(length_conflicts)}")
        original_count = len(resolved_suggestions)
        
        # Filter out suggestions that add too much content
        # Keep only the most important suggestions (prioritize quality over quantity)
        if len(resolved_suggestions) > 3:
            # If we have many suggestions and length is an issue, keep only top 3
            resolved_suggestions = resolved_suggestions[:3]
            filtered_count = original_count - len(resolved_suggestions)
            print(f"   Filtered {filtered_count} suggestions to comply with length constraints")
        
        # Also modify remaining suggestions to be more concise
        for i, suggestion in enumerate(resolved_suggestions):
            if len(suggestion) > 150:  # Long suggestions
                # Truncate and add note
                resolved_suggestions[i] = suggestion[:140] + "... (truncated for length)"
                modified_count += 1
    
    # Strategy 2: Filter/modify suggestions that cause format preservation conflicts
    if format_preservation_conflicts:
        print(f"   Format preservation conflicts detected: {len(format_preservation_conflicts)}")
        
        # Remove suggestions that explicitly change format
        filtered = []
        for suggestion in resolved_suggestions:
            # Check if suggestion mentions changing format
            suggestion_lower = suggestion.lower()
            format_changing_keywords = [
                "change to paragraph", "convert to bullets", "switch to",
                "change format", "reformat", "restructure"
            ]
            
            if any(keyword in suggestion_lower for keyword in format_changing_keywords):
                filtered_count += 1
                print(f"   Filtered suggestion (format change): {suggestion[:60]}...")
            else:
                filtered.append(suggestion)
        
        resolved_suggestions = filtered
    
    # Strategy 3: Modify suggestions to fix consistency conflicts
    if consistency_conflicts:
        print(f"   Consistency conflicts detected: {len(consistency_conflicts)}")
        
        # Add note to suggestions about maintaining consistency
        for i, suggestion in enumerate(resolved_suggestions):
            if "date" in suggestion.lower() or "format" in suggestion.lower():
                # Add consistency reminder
                if "maintain consistency" not in suggestion.lower():
                    resolved_suggestions[i] = suggestion + " (maintain consistent format)"
                    modified_count += 1
    
    # Strategy 4: Handle other conflicts (general filtering)
    if other_conflicts:
        print(f"   Other conflicts detected: {len(other_conflicts)}")
        # For other conflicts, we might need to be more conservative
        # Keep only high-priority suggestions
        if len(resolved_suggestions) > 2:
            resolved_suggestions = resolved_suggestions[:2]
            filtered_count += len(user_modifications) - 2 if user_modifications else 0
    
    # Compile conflict messages
    all_conflicts = length_conflicts + format_preservation_conflicts + consistency_conflicts + other_conflicts
    
    if all_conflicts:
        print(f"\n   Resolution Summary:")
        print(f"   - Conflicts found: {len(all_conflicts)}")
        print(f"   - Suggestions filtered: {filtered_count}")
        print(f"   - Suggestions modified: {modified_count}")
        print(f"   - Remaining suggestions: {len(resolved_suggestions)}")
        
        if resolved_suggestions:
            print(f"\n   Resolved suggestions (safe to apply):")
            for i, suggestion in enumerate(resolved_suggestions[:3], 1):
                print(f"   {i}. {suggestion[:80]}...")
        else:
            print("\n   ⚠️  All suggestions filtered - no safe suggestions to apply")
        
        return {
            "conflicts_resolved": True,  # We resolved them (even if by filtering)
            "conflict_messages": all_conflicts,
            "resolved_suggestions": resolved_suggestions,
            "filtered_count": filtered_count,
            "modified_count": modified_count
        }
    else:
        print("   ✓ No conflicts detected!")
        return {
            "conflicts_resolved": True,
            "conflict_messages": [],
            "resolved_suggestions": resolved_suggestions
        }

# score → review → optimize again OR exit → optimize → score → review → optimize again OR exit
graph = StateGraph(ResumeState)
# Resume Analyzer Agent
graph.add_node("parse_resume", parse_resume)
graph.add_node("check_ats_score", check_ats_score)
graph.add_node("check_skill_gap_score", check_skill_gap_score)
graph.add_node("check_relevant_experience", check_relevant_experience)
graph.add_node("aggregate_analyses", aggregate_analyses)
# Optimization Suggestion Agent
graph.add_node("generate_suggestions", generate_suggestions)
graph.add_node("apply_suggestions", apply_suggestions)
# Formatting Agent (Hybrid Approach)
graph.add_node("extract_formatting_state", extract_formatting_state_node)  # Light check BEFORE suggestions
graph.add_node("format_resume", format_resume)  # Full validation AFTER applying suggestions

# Orchestrator Agent
graph.add_node("orchestrator", orchestrator)  # Dedicated orchestrator node
graph.add_node("human_in_loop", human_in_loop)
graph.add_node("process_user_decision", process_user_decision)  # Process user decision after Command (renamed function)
graph.add_node("clear_optimization_state", clear_optimization_state)  # Clear state after showing results

# Conflict Resolution Agent
graph.add_node("resolve_conflicts", resolve_conflicts)

# Feedback Collection Agent

# Workflow Execution Dashboard
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
# Orchestrator coordinates the workflow
# After analysis, check if we should show results or wait for decision
def route_after_aggregate(state: ResumeState):
    """Route after aggregate: check if we're showing results or need user decision."""
    # Check if we're in "show results" mode (after optimization, state cleared)
    formatting_state = state.get("formatting_state")
    user_modifications = state.get("user_modifications")
    user_decision = state.get("user_decision")
    ats_score = state.get("ats_score")
    workflow_complete = state.get("workflow_complete", False)
    extracted_sections = state.get("extracted_sections", [])
    
    # If workflow is already complete, end
    # print(f"\n[DEBUG] route_after_aggregate: workflow_complete = {workflow_complete}")
    if workflow_complete:
        # print("\n[DEBUG] route_after_aggregate: workflow_complete = True, routing to END")
        return END
    
    # If we have results but no decision and no optimization artifacts:
    # Check if user_decision is "show_results" (set by clear_optimization_state after optimization)
    # This is more reliable than workflow_complete since user_decision is in the original TypedDict
    if ats_score is not None and not formatting_state and not user_modifications:
        # Check if user_decision is "show_results" (indicates we're showing results after optimization)
        if user_decision == "show_results":
            # print("\n[DEBUG] route_after_aggregate: Showing results mode detected (user_decision='show_results'), routing to END")
            return END
        elif not user_decision:
            # No user_decision yet - check workflow_complete as fallback
            if workflow_complete:
                # print("\n[DEBUG] route_after_aggregate: Showing results mode detected (workflow_complete=True), routing to END")
                return END
            else:
                # Fresh analysis - need user decision
                # print("\n[DEBUG] route_after_aggregate: Fresh analysis complete, routing to human_in_loop")
                return "human_in_loop"
        else:
            # Has user_decision (not "show_results") - need to process it
            # print("\n[DEBUG] route_after_aggregate: Has user_decision, routing to human_in_loop")
            return "human_in_loop"
    else:
        # Need user decision - go to human_in_loop to prompt
        # print("\n[DEBUG] route_after_aggregate: Need user decision, routing to human_in_loop")
        return "human_in_loop"

graph.add_conditional_edges(
    "aggregate_analyses",
    route_after_aggregate,
    {
        "human_in_loop": "human_in_loop",
        END: END
    }
)

# After human_in_loop, always go to process_user_decision (if it prompted) or END (if it showed results)
def route_after_human_loop(state: ResumeState):
    """Route after human_in_loop: END if workflow complete, otherwise process decision."""
    workflow_complete = state.get("workflow_complete", False)
    # print(f"\n[DEBUG] route_after_human_loop: workflow_complete = {workflow_complete}")
    if workflow_complete:
        # print("   [DEBUG] Routing to END (workflow complete)")
        return END
    else:
        # print("   [DEBUG] Routing to process_user_decision (waiting for decision)")
        return "process_user_decision"

graph.add_conditional_edges(
    "human_in_loop",
    route_after_human_loop,
    {
        "process_user_decision": "process_user_decision",
        END: END
    }
)

# After processing user decision, go to orchestrator
graph.add_edge("process_user_decision", "orchestrator")
graph.add_conditional_edges(
    "orchestrator",
    route_based_on_orchestrator,
    {
        "extract_formatting_state": "extract_formatting_state",  # User chose to optimize
        "parse_resume": "parse_resume",  # Re-parse if needed
        "human_in_loop": "human_in_loop",  # Should not happen if workflow_complete is checked
        "error_handler": "error_handler",
        END: END  # User chose to exit or workflow complete
    }
)

# Hybrid Formatting Approach:
# 1. BEFORE suggestions: Extract formatting state (lightweight)
graph.add_edge("extract_formatting_state", "generate_suggestions")

# 2. Generate and apply suggestions
graph.add_edge("generate_suggestions", "apply_suggestions")

# 3. AFTER applying: Full formatting validation
graph.add_edge("apply_suggestions", "format_resume")

# 4. Resolve conflicts between Optimization and Formatting agents
#    This merges/prioritizes suggestions: filters conflicting ones, modifies others
#    Result: resolved_suggestions (safe to apply) stored in state
graph.add_edge("format_resume", "resolve_conflicts")

# 5. After conflict resolution, clear optimization state and show results
#    This prevents loops and allows user to start fresh if they want to optimize again
graph.add_edge("resolve_conflicts", "clear_optimization_state")
# After clearing state, re-parse to get updated scores, then show results
graph.add_edge("clear_optimization_state", "parse_resume")  # Re-parse to get updated scores
# After re-parsing, it will go through analysis → aggregate → route_after_aggregate
# route_after_aggregate should detect workflow_complete and route to END

graph.add_edge("error_handler", END)

checkpointer = MemorySaver()
# End after aggregation
# Only compile if this file is executed directly (not imported)
# This prevents the graph from executing during import
if __name__ == "__main__":
    compiled_graph = graph.compile(checkpointer=checkpointer)
    graph_return = prepare_documents_for_job("123", "456", compiled_graph)
    # print("graph_return???: ", graph_return)
else:
    # If imported, create a placeholder that can be compiled later
    compiled_graph = None
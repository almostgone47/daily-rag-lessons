"""
Day 15: REST API Wrapper for Resume Optimization

This module provides simple, stateless API functions for the React app.
Unlike the LangGraph workflow (which was great for learning), these functions
are designed for on-demand, per-section optimization requests.

Key Design Decisions:
- Modular, composable functions (easy to test, reuse, maintain)
- Fast job-specific ATS scoring (keyword matching, no LLM needed)
- Section-specific suggestions (not full resume analysis)
- Estimation-based improvement calculation (fast preview)

Usage:
    from lesson15_api_wrapper import get_suggestions_for_section
    
    result = get_suggestions_for_section(
        resume_data=resume,
        section_id="experience",
        section_text=section_content,
        job_description=job_desc,
        context={}
    )
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from langsmith import traceable
# Import helpers functions (with fallback if import fails)
try:
    from helpers import extract_keywords_from_job_description, parse_sections
except ImportError:
    # Fallback: define minimal versions if helpers can't be imported
    def extract_keywords_from_job_description(job_description: str) -> List[str]:
        """Fallback keyword extraction if helpers can't be imported."""
        # Simple extraction - just find capitalized words that look like tech terms
        keywords = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', job_description)
        return [k.lower() for k in keywords if len(k) > 2][:20]
    
    def parse_sections(resume):
        """Fallback section parsing if helpers can't be imported."""
        chunks = []
        sections = resume.get('sections', [])
        for section in sections:
            kind = section.get('kind', '')
            title = section.get('title', '')
            content = section.get('content', '')
            if content:
                chunks.append(f"{title}: {content}")
        return chunks

# Load environment variables
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_keywords_from_text(text: str) -> List[str]:
    """
    Extract keywords from any text (for suggestions, resume sections, etc.).
    Simple approach: extract words that look like technical terms.
    
    TODO: Can be enhanced with NLP for better accuracy.
    """
    # Simple keyword extraction: words that are capitalized or common tech terms
    words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    # Also catch common tech patterns (e.g., "React.js", "Node.js")
    tech_patterns = re.findall(r'\b\w+\.(?:js|ts|py|java|cpp|go|rs)\b', text, re.IGNORECASE)
    
    keywords = [w.lower() for w in words if len(w) > 2] + [p.lower() for p in tech_patterns]
    # Remove duplicates and common stop words
    stop_words = {'the', 'and', 'for', 'with', 'from', 'this', 'that', 'was', 'were'}
    keywords = [k for k in set(keywords) if k not in stop_words]
    
    return keywords[:20]  # Limit to top 20


def get_section_text(resume_data: Dict[str, Any], section_id: str) -> str:
    """
    Extract text content from a specific section of the resume.
    
    Args:
        resume_data: Full resume JSON structure
        section_id: Section identifier (e.g., "experience", "summary", "skills")
    
    Returns:
        Plain text representation of the section
    """
    sections = resume_data.get("sections", [])
    
    # Map section_id to section kind
    section_map = {
        "experience": "EXPERIENCE",
        "summary": "RICHTEXT",
        "skills": "SKILLS",
        "education": "EDUCATION",
    }
    
    target_kind = section_map.get(section_id.lower(), section_id.upper())
    
    # Find the section
    for section in sections:
        if section.get("kind") == target_kind:
            title = section.get("title", "")
            content = section.get("content", "")
            
            # For Experience sections, include items
            if target_kind == "EXPERIENCE":
                items = section.get("items", [])
                text_parts = [title] if title else []
                for item in items:
                    headline = item.get("headline", "")
                    subheadline = item.get("subheadline", "")
                    bullets = item.get("bullets", [])
                    
                    if headline:
                        text_parts.append(f"{headline} at {subheadline}" if subheadline else headline)
                    if bullets:
                        text_parts.extend(bullets)
                
                return " ".join(text_parts)
            
            # For Skills sections
            elif target_kind == "SKILLS":
                items = section.get("items", [])
                skill_names = [item.get("headline", "").strip() for item in items if item.get("headline")]
                return ", ".join(skill_names)
            
            # For other sections (RICHTEXT, EDUCATION, etc.)
            else:
                return f"{title} {content}" if title else content
    
    return ""


def get_all_resume_text(resume_data: Dict[str, Any]) -> str:
    """
    Extract all text content from the resume for full-context analysis.
    
    Args:
        resume_data: Full resume JSON structure
    
    Returns:
        Plain text representation of entire resume
    """
    sections = parse_sections(resume_data)
    return " ".join(sections)


def matches_keyword(keyword: str, text: str) -> bool:
    """
    Check if a keyword matches in text (case-insensitive, word boundary aware).
    
    Args:
        keyword: The keyword to search for
        text: The text to search in
    
    Returns:
        True if keyword found, False otherwise
    """
    # Normalize both to lowercase
    keyword_lower = keyword.lower().strip()
    text_lower = text.lower()
    
    # Exact word match (handles "React" matching "React" but not "Reacting")
    pattern = r'\b' + re.escape(keyword_lower) + r'\b'
    return bool(re.search(pattern, text_lower))


# ============================================================================
# CORE API FUNCTIONS
# ============================================================================

def calculate_job_ats_score(
    resume_data: Dict[str, Any],
    job_description: str,
    current_section_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate job-specific ATS score based on keyword matching.
    
    This is lightweight and fast - no LLM needed, just text matching.
    Uses weighted analysis: entire resume + focus on current section.
    
    Args:
        resume_data: Full resume JSON structure
        job_description: Job description text
        current_section_id: Optional section ID being optimized (for weighting)
    
    Returns:
        {
            "score": float (0-100),
            "keywordMatches": List[str],
            "missingKeywords": List[str],
            "sectionBreakdown": {
                "currentSection": float or None,
                "overall": float
            }
        }
    """
    # Extract keywords from job description
    job_keywords = extract_keywords_from_job_description(job_description)
    
    if not job_keywords:
        return {
            "score": 0,
            "keywordMatches": [],
            "missingKeywords": [],
            "sectionBreakdown": {"currentSection": None, "overall": 0}
        }
    
    # Get all resume text
    all_resume_text = get_all_resume_text(resume_data)
    all_resume_lower = all_resume_text.lower()
    
    # Calculate matches across entire resume
    matched_keywords = []
    missing_keywords = []
    
    for keyword in job_keywords:
        if matches_keyword(keyword, all_resume_lower):
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)
    
    # Calculate overall score
    overall_score = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    
    # If optimizing a specific section, calculate weighted score
    if current_section_id:
        current_section_text = get_section_text(resume_data, current_section_id)
        current_section_lower = current_section_text.lower()
        
        # Count matches in current section
        section_matched = [kw for kw in job_keywords if matches_keyword(kw, current_section_lower)]
        section_score = (len(section_matched) / len(job_keywords)) * 100 if job_keywords else 0
        
        # Weighted combination: 60% current section, 40% rest of resume
        final_score = (section_score * 0.6) + (overall_score * 0.4)
        
        return {
            "score": round(final_score, 1),
            "keywordMatches": matched_keywords,
            "missingKeywords": missing_keywords,
            "sectionBreakdown": {
                "currentSection": round(section_score, 1),
                "overall": round(overall_score, 1)
            }
        }
    else:
        return {
            "score": round(overall_score, 1),
            "keywordMatches": matched_keywords,
            "missingKeywords": missing_keywords,
            "sectionBreakdown": {
                "currentSection": None,
                "overall": round(overall_score, 1)
            }
        }


def calculate_improvement_delta(
    current_score: float,
    suggestion_text: str,
    missing_keywords: List[str],
    job_description: str
) -> Dict[str, Any]:
    """
    Estimate score improvement if suggestion is applied.
    Fast calculation without actually applying the suggestion.
    
    Args:
        current_score: Current ATS score (0-100)
        suggestion_text: The suggested text to add/modify
        missing_keywords: List of keywords missing from resume
        job_description: Job description (for keyword weighting)
    
    Returns:
        {
            "improvement": float (estimated score increase),
            "confidence": str ("high" | "medium" | "low"),
            "reasoning": str (explanation)
        }
    """
    if not missing_keywords:
        return {
            "improvement": 0.0,
            "confidence": "low",
            "reasoning": "No missing keywords to match"
        }
    
    # Extract keywords from suggestion text
    suggestion_keywords = extract_keywords_from_text(suggestion_text)
    suggestion_lower = suggestion_text.lower()
    
    # Check which missing keywords appear in suggestion
    newly_matched = []
    for missing_kw in missing_keywords:
        if matches_keyword(missing_kw, suggestion_lower):
            newly_matched.append(missing_kw)
    
    if not newly_matched:
        return {
            "improvement": 0.0,
            "confidence": "low",
            "reasoning": "Suggestion does not contain missing keywords"
        }
    
    # Estimate improvement: each matched missing keyword adds points
    # Weight by keyword importance (simple: all keywords weighted equally for now)
    # Each match adds ~2-3 points (depending on total keywords)
    total_job_keywords = len(extract_keywords_from_job_description(job_description))
    points_per_match = max(2.0, 100.0 / total_job_keywords) if total_job_keywords > 0 else 2.0
    
    improvement = len(newly_matched) * points_per_match
    
    # Cap improvement at reasonable max (e.g., +15 points per suggestion)
    improvement = min(improvement, 15.0)
    
    # Determine confidence
    if len(newly_matched) >= 3:
        confidence = "high"
    elif len(newly_matched) >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    
    reasoning = f"Suggestion contains {len(newly_matched)} missing keywords: {', '.join(newly_matched[:3])}"
    
    return {
        "improvement": round(improvement, 1),
        "confidence": confidence,
        "reasoning": reasoning
    }


def generate_section_suggestions(
    resume_data: Dict[str, Any],
    section_id: str,
    section_text: str,
    job_description: str,
    context: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """
    Generate AI suggestions for a specific section.
    
    Args:
        resume_data: Full resume JSON structure (for context)
        section_id: Section identifier (e.g., "experience", "summary")
        section_text: Current text content of the section
        job_description: Job description text
        context: Optional context (company research, achievements, etc.)
    
    Returns:
        List of suggestion objects:
        [
            {
                "text": str (the suggested text),
                "rationale": str (why this improves the section)
            },
            ...
        ]
    """
    if context is None:
        context = {}
    
    # Build context string from optional context
    context_parts = []
    if context.get("companyResearch"):
        context_parts.append(f"Company Research: {context['companyResearch']}")
    if context.get("keyAchievements"):
        context_parts.append(f"Key Achievements: {context['keyAchievements']}")
    if context.get("skillsInventory"):
        context_parts.append(f"Skills: {', '.join(context['skillsInventory'])}")
    
    context_str = "\n".join(context_parts) if context_parts else ""
    
    # Map section_id to readable name
    section_names = {
        "experience": "Experience",
        "summary": "Summary",
        "skills": "Skills",
        "education": "Education"
    }
    section_name = section_names.get(section_id.lower(), section_id)
    
    # Build prompt
    prompt = f"""You are a resume optimization assistant. Generate specific, actionable suggestions to improve the {section_name} section of this resume for the target job.

{section_name} Section Content:
{section_text[:1500]}

Target Job Description:
{job_description[:1000]}

{f"Additional Context:\n{context_str}" if context_str else ""}

Instructions:
1. Analyze this {section_name} section against the job description
2. Generate 3-5 specific, actionable suggestions to improve ONLY this section
3. Each suggestion should:
   - Be specific and actionable (not vague like "improve this")
   - Relate directly to the job requirements
   - Preserve the existing format and style
   - Include concrete examples or improvements
4. For each suggestion, provide:
   - The suggested text or modification
   - A brief rationale explaining why this improves the section

Return your response as a JSON array of objects:
[
  {{
    "text": "Specific suggestion text here",
    "rationale": "Why this improves the section"
  }},
  ...
]

Suggestions:"""
    
    try:
        # completion = groq_client.chat.completions.create(
        #     model="llama-3.1-8b-instant",
        #     messages=[{"role": "user", "content": prompt}],
        #     temperature=0.7,
        #     max_tokens=1024,
        #     stream=False
        # )
        response_text = call_groq_llm(prompt, model="llama-3.1-8b-instant", temperature=0.7, max_tokens=1024, stream=False)
        
        # response_text = completion.choices[0].message.content.strip()
        
        # Parse JSON from response
        suggestions = []
        
        # Try to extract JSON array
        json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
        if json_match:
            try:
                suggestions = json.loads(json_match.group())
            except json.JSONDecodeError:
                # Fallback: try to parse as lines
                lines = [s.strip() for s in response_text.split('\n') if s.strip() and not s.strip().startswith('#')]
                suggestions = [{"text": line, "rationale": ""} for line in lines[:5]]
        else:
            # Fallback: treat each line as a suggestion
            lines = [s.strip() for s in response_text.split('\n') if s.strip() and len(s.strip()) > 10]
            suggestions = [{"text": line, "rationale": ""} for line in lines[:5]]
        
        # Ensure all suggestions have required fields
        formatted_suggestions = []
        for sug in suggestions[:5]:
            if isinstance(sug, dict):
                formatted_suggestions.append({
                    "text": sug.get("text", str(sug)),
                    "rationale": sug.get("rationale", "Improves job match")
                })
            else:
                formatted_suggestions.append({
                    "text": str(sug),
                    "rationale": "Improves job match"
                })
        
        return formatted_suggestions
    
    except Exception as e:
        print(f"Error generating suggestions: {e}")
        return []


# ============================================================================
# MAIN API FUNCTION (Composes everything)
# ============================================================================

def get_suggestions_for_section(
    resume_data: Dict[str, Any],
    section_id: str,
    section_text: str,
    job_description: str,
    context: Dict[str, Any] = None,
    include_ats_score: bool = True
) -> Dict[str, Any]:
    """
    Main API function: Get suggestions for a section with optional ATS scoring.
    
    This is the function your React app would call:
    POST /api/ai/resume/suggestions
    
    Args:
        resume_data: Full resume JSON structure
        section_id: Section identifier (e.g., "experience", "summary")
        section_text: Current text content of the section
        job_description: Job description text
        context: Optional context (company research, achievements, etc.)
        include_ats_score: Whether to calculate and include ATS score
    
    Returns:
        {
            "suggestions": [
                {
                    "text": str,
                    "rationale": str,
                    "improvement": float (optional)
                },
                ...
            ],
            "jobAtsScore": {
                "score": float,
                "keywordMatches": List[str],
                "missingKeywords": List[str],
                "sectionBreakdown": {...}
            } (optional)
        }
    """
    # Generate suggestions
    suggestions = generate_section_suggestions(
        resume_data=resume_data,
        section_id=section_id,
        section_text=section_text,
        job_description=job_description,
        context=context or {}
    )
    
    result = {
        "suggestions": suggestions
    }
    
    # Calculate ATS score if requested
    if include_ats_score:
        ats_score = calculate_job_ats_score(
            resume_data=resume_data,
            job_description=job_description,
            current_section_id=section_id
        )
        
        result["jobAtsScore"] = ats_score
        
        # Calculate improvement for each suggestion
        for suggestion in suggestions:
            improvement = calculate_improvement_delta(
                current_score=ats_score["score"],
                suggestion_text=suggestion["text"],
                missing_keywords=ats_score["missingKeywords"],
                job_description=job_description
            )
            suggestion["improvement"] = improvement["improvement"]
            suggestion["improvementConfidence"] = improvement["confidence"]
            suggestion["improvementReasoning"] = improvement["reasoning"]
    
    return result

@traceable(name="llm_call", run_type="llm")
def call_groq_llm(prompt: str, model: str = "llama-3.1-8b-instant", **kwargs):
    """Wrapper function to trace Groq LLM calls."""
    from lesson15_api_wrapper import groq_client
    
    completion = groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        **kwargs
    )
    return completion.choices[0].message.content.strip()


# ============================================================================
# DEMO / TESTING
# ============================================================================

if __name__ == "__main__":
    """
    Simple demo of the API functions.
    In production, these would be called from your Express.js API endpoints.
    """
    try:
        from helpers import load_resume_data, load_job_description
    except ImportError:
        # Fallback for demo if helpers can't be imported
        import json
        def load_resume_data():
            with open('resume.json', 'r') as f:
                return json.load(f)
        def load_job_description():
            with open('jobDescription.json', 'r') as f:
                return f.read()
    
    print("=" * 60)
    print("Day 15: REST API Wrapper Demo")
    print("=" * 60)
    
    # Load test data
    print("\n📄 Loading resume and job description...")
    resume_data = load_resume_data()
    job_description = load_job_description()
    
    # Get section text (Experience section)
    print("\n📝 Extracting Experience section...")
    section_text = get_section_text(resume_data, "experience")
    print(f"   Section text length: {len(section_text)} characters")
    
    # Calculate ATS score
    print("\n📊 Calculating job-specific ATS score...")
    ats_score = calculate_job_ats_score(
        resume_data=resume_data,
        job_description=job_description,
        current_section_id="experience"
    )
    print(f"   Overall Score: {ats_score['score']}/100")
    print(f"   Current Section Score: {ats_score['sectionBreakdown']['currentSection']}/100")
    print(f"   Matched Keywords: {len(ats_score['keywordMatches'])}")
    print(f"   Missing Keywords: {len(ats_score['missingKeywords'])}")
    if ats_score['missingKeywords']:
        print(f"   Missing: {', '.join(ats_score['missingKeywords'][:5])}...")
    
    # Generate suggestions
    print("\n💡 Generating suggestions for Experience section...")
    result = get_suggestions_for_section(
        resume_data=resume_data,
        section_id="experience",
        section_text=section_text,
        job_description=job_description,
        context={},
        include_ats_score=True
    )
    
    print(f"\n✅ Generated {len(result['suggestions'])} suggestions:")
    for i, suggestion in enumerate(result['suggestions'], 1):
        print(f"\n   {i}. {suggestion['text'][:80]}...")
        if suggestion.get('rationale'):
            print(f"      Rationale: {suggestion['rationale'][:60]}...")
        if suggestion.get('improvement'):
            print(f"      Estimated Improvement: +{suggestion['improvement']} points ({suggestion.get('improvementConfidence', 'unknown')} confidence)")
    
    print("\n" + "=" * 60)
    print("Demo complete! These functions are ready for your React app.")
    print("=" * 60)


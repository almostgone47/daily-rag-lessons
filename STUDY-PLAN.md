# RAG, Agentic Workflows & Evaluation Mastery Plan

## 4-Week Intensive Study Plan (2 hours/day)

**Goal**: Master RAG (LangChain), Agentic Workflows (LangGraph), and Evaluation
(LangSmith) by building a **Resume Optimization Agent**.

**Context**: All exercises will build toward a production-ready Resume
Optimization Agent that can analyze resumes, suggest improvements, and provide
actionable feedback.

---

## Phase 1: Mastery of Retrieval (Days 1-8)

**Focus**: Loading, Chunking, Vector Stores, Hybrid Search, and Reranking

_COMPLETED_ ### Day 1: Document Loading & Basic Setup

**Tasks**:

- Build a document loader that reads PDF resumes using LangChain's document
  loaders
- Implement a text extraction pipeline that handles multiple resume formats
  (PDF, DOCX, TXT)
- Create a data preprocessing module that cleans extracted text (remove headers,
  footers, normalize whitespace)
- Set up a project structure with proper dependency management (requirements.txt
  or poetry)

**Definition of Done**:

- Successfully load 5 sample resumes in different formats
- Extract and clean text from all formats without errors
- Project structure is organized with separate modules for loading, processing,
  and utilities

---

### Day 2: Text Chunking Strategies

**Tasks**:

- Implement recursive character text splitter with custom chunk size and overlap
- Build a semantic chunker that splits resumes by sections (Education,
  Experience, Skills)
- Create a code-aware chunker for technical resumes (preserves code blocks,
  formatting)
- Implement chunk metadata extraction (section type, page number, character
  position)

**Definition of Done**:

- Chunk 3 sample resumes using all three strategies
- Compare chunk quality: semantic chunker produces meaningful section-based
  chunks
- Each chunk has associated metadata for traceability

---

### Day 3: Embeddings & Vector Store Setup

**Tasks**:

- Implement embedding generation using OpenAI embeddings (or alternative like
  Cohere)
- Build a vector store integration with Chroma (local) or Pinecone (cloud)
- Create an embedding cache system to avoid re-embedding unchanged documents
- Implement batch embedding for efficient processing of multiple resumes

**Definition of Done**:

- Generate embeddings for all resume chunks
- Store embeddings in vector database with proper indexing
- Query system returns top 3 most similar chunks for a test query
- Embedding cache prevents redundant API calls

---

### Day 4: Basic Retrieval & Similarity Search

**Tasks**:

- Build a retrieval chain using LangChain's RetrievalQA
- Implement similarity search with configurable top-k results
- Create a query preprocessing function (expand abbreviations, normalize terms)
- Build a retrieval pipeline that returns chunks with relevance scores

**Definition of Done**:

- Query "What are the candidate's technical skills?" returns relevant skill
  sections
- Retrieval system handles queries about experience, education, and achievements
- All retrieved chunks include similarity scores above a threshold (e.g., 0.7)

---

### Day 5: Advanced Chunking for Resumes

**Tasks**:

- Implement a hierarchical chunking system (document → sections → sentences)
- Build a resume parser that identifies structured sections (Name, Contact,
  Summary, Experience, Education, Skills)
- Create chunking logic that preserves context (e.g., job title + company +
  duration as one chunk)
- Implement overlap strategies that maintain semantic coherence across chunk
  boundaries

**Definition of Done**:

- Resume parser correctly identifies all major sections in 5 test resumes
- Hierarchical chunks maintain parent-child relationships
- Chunks preserve full context (no broken sentences or incomplete information)

---

### Day 6: Hybrid Search Implementation

**Tasks**:

- Build a hybrid search system combining vector similarity (semantic) and
  keyword matching (BM25)
- Implement a weighted scoring mechanism (70% semantic, 30% keyword)
- Create a query analyzer that determines when to use hybrid vs. pure semantic
  search
- Build a result fusion algorithm that merges and deduplicates results from both
  search methods

**Definition of Done**:

- Hybrid search outperforms pure semantic search on 10 test queries
- System handles both semantic queries ("leadership experience") and exact
  keyword queries ("Python")
- Result fusion eliminates duplicates and ranks by combined relevance score

---

### Day 7: Reranking Implementation

**Tasks**:

- Integrate a reranking model (Cohere Rerank API or cross-encoder from
  sentence-transformers)
- Build a two-stage retrieval pipeline: retrieve top-20, rerank to top-5
- Implement reranking with query context (e.g., "optimize for ATS
  compatibility")
- Create a cost optimization system that only reranks when necessary

**Definition of Done**:

- Reranking improves precision: top-5 results are more relevant than pre-rerank
  top-5
- System handles different reranking contexts (ATS optimization, skill matching,
  experience relevance)
- Cost tracking shows reranking only applied when retrieval confidence is below
  threshold

---

### Day 8: Complete Retrieval System Integration

**Tasks**:

- Integrate all components: loading → chunking → embedding → hybrid search →
  reranking
- Build a unified retrieval interface that accepts queries and returns optimized
  results
- Implement error handling and fallback mechanisms (if reranking fails, return
  semantic results)
- Create a retrieval evaluation script that tests the system on 20 diverse
  queries

**Definition of Done**:

- End-to-end retrieval pipeline processes a resume and answers queries in <3
  seconds
- System handles edge cases: empty queries, no matches, malformed resumes
- Evaluation script shows >85% relevance for top-3 retrieved chunks across test
  queries

---

## Phase 2: Mastery of Workflow (Days 9-15)

**Focus**: LangGraph, State, Conditional Edges, and Human-in-the-Loop

### Day 9: LangGraph Basics & State Management

**Tasks**:

- Build a simple LangGraph workflow with 3 nodes: Parse Resume → Extract Skills
  → Generate Summary
- Implement state management using TypedDict for type-safe state passing
- Create state reducers that merge information from multiple nodes
- Build a workflow visualizer that shows node execution order

**Definition of Done**:

- LangGraph workflow executes all 3 nodes in sequence
- State is properly typed and accessible across nodes
- Workflow visualization shows clear node dependencies
- Each node logs its input/output for debugging

---

### Day 10: Conditional Edges & Routing

**Tasks**:

- Build a conditional routing system: if resume has <2 years experience → route
  to "Junior" path, else "Senior" path
- Implement multi-conditional edges (route based on skill count, education
  level, industry)
- Create a dynamic routing node that analyzes resume quality and routes to
  appropriate optimization strategy
- Build error handling routes (if parsing fails → route to manual review)

**Definition of Done**:

- Conditional routing correctly categorizes 10 test resumes (Junior vs. Senior)
- Multi-conditional logic handles complex scenarios (e.g., experienced but
  missing skills)
- Error routes gracefully handle failures without crashing the workflow

---

### Day 11: Parallel Node Execution

**Tasks**:

- Build parallel execution for independent analyses: ATS compatibility check,
  skill gap analysis, formatting review
- Implement a synchronization node that waits for all parallel branches before
  proceeding
- Create a result aggregator that combines outputs from parallel nodes
- Build a performance monitor that tracks execution time for each parallel
  branch

**Definition of Done**:

- Three parallel analyses complete simultaneously (not sequentially)
- Synchronization node correctly waits for all branches
- Aggregated results combine insights from all parallel analyses
- Performance logs show parallel execution is faster than sequential

---

### Day 12: Human-in-the-Loop Integration

**Tasks**:

- Build a human-in-the-loop node that pauses workflow for user feedback
- Implement a feedback collection system (approve/reject/modify suggestions)
- Create a resume that resumes workflow execution after human input
- Build a timeout mechanism (if no human response in 5 minutes, use default
  action)

**Definition of Done**:

- Workflow pauses at designated checkpoints and waits for human approval
- User can approve, reject, or modify optimization suggestions
- Workflow resumes correctly after human input
- Timeout mechanism prevents indefinite waiting

---

### Day 13: State Persistence & Workflow Recovery

**Tasks**:

- Implement state persistence to disk/database (save workflow state after each
  node)
- Build a workflow recovery system that can resume from last saved state
- Create a checkpoint system that allows rolling back to previous states
- Implement state versioning to track workflow evolution

**Definition of Done**:

- Workflow state is saved after each node execution
- System can recover and resume from any checkpoint
- Rollback functionality restores previous state correctly
- State history shows complete workflow evolution

---

### Day 14: Complex Multi-Agent Workflow

**Tasks**:

- Build a multi-agent system: Resume Analyzer Agent, Optimization Suggestion
  Agent, Formatting Agent
- Implement agent communication via shared state
- Create an orchestrator agent that coordinates other agents
- Build a conflict resolution system when agents provide conflicting suggestions

**Definition of Done**:

- Three agents execute in coordinated sequence
- Agents communicate effectively via shared state
- Orchestrator manages agent execution order and dependencies
- Conflict resolution merges or prioritizes conflicting suggestions

---

### Day 15: Complete Resume Optimization Workflow

**Tasks**:

- Integrate all workflow components into a complete Resume Optimization Agent
- Build the full pipeline: Load → Parse → Analyze (parallel) → Optimize → Format
  → Review (human-in-loop) → Finalize
- Implement comprehensive error handling and retry logic
- Create a workflow execution dashboard that shows real-time progress

**Definition of Done**:

- Complete workflow processes a resume end-to-end with all features
- Dashboard shows real-time node execution and state changes
- Error handling gracefully manages failures at any stage
- Final output is an optimized resume with tracked changes and reasoning

---

## Phase 3: Mastery of Evaluation (Days 16-20)

**Focus**: LangSmith integration and dataset testing

### Day 16: LangSmith Setup & Basic Tracing

**Tasks**:

- Set up LangSmith account and configure API keys
- Implement LangSmith tracing for all LangChain/LangGraph operations
- Build a tracing dashboard that shows all chain executions
- Create trace filters to find specific runs (by date, query, error status)

**Definition of Done**:

- All LangChain operations are traced in LangSmith
- Dashboard displays complete execution traces with timing
- Filters successfully isolate specific runs for analysis
- Traces include input/output for each step

---

### Day 17: Dataset Creation & Management

**Tasks**:

- Build a dataset of 50 resume-query pairs (resume + expected optimization
  suggestions)
- Create dataset versioning system (v1, v2, etc.) for iterative improvement
- Implement dataset validation (check for duplicates, missing fields, invalid
  formats)
- Build a dataset loader that integrates with LangSmith's dataset API

**Definition of Done**:

- Dataset contains 50 diverse resume-query pairs with ground truth
- Versioning system tracks dataset changes over time
- Validation catches data quality issues before evaluation
- Dataset loads successfully into LangSmith for testing

---

### Day 18: Evaluation Metrics & Scoring

**Tasks**:

- Implement custom evaluation metrics: relevance score, suggestion quality, ATS
  compatibility improvement
- Build evaluators using LangSmith's evaluation framework
- Create scoring functions that compare generated suggestions to ground truth
- Implement multi-criteria evaluation (accuracy, completeness, actionability)

**Definition of Done**:

- Three custom evaluators measure different aspects of optimization quality
- Evaluation runs on entire dataset and produces aggregate scores
- Scoring functions provide detailed breakdown (per-resume and aggregate)
- Multi-criteria evaluation shows strengths and weaknesses across dimensions

---

### Day 19: A/B Testing & Experimentation

**Tasks**:

- Build an A/B testing framework that compares different chunking strategies
- Implement experiment tracking: test hybrid search vs. pure semantic search
- Create experiment comparison dashboard showing performance differences
- Build statistical significance testing for experiment results

**Definition of Done**:

- A/B tests compare at least 3 different configurations
- Experiment tracking shows clear performance differences
- Dashboard visualizes results with confidence intervals
- Statistical tests determine if differences are significant

---

### Day 20: Production Evaluation & Monitoring

**Tasks**:

- Build a production monitoring system that tracks evaluation metrics in
  real-time
- Implement alerting for performance degradation (accuracy drops below
  threshold)
- Create a feedback loop: collect user feedback and update evaluation dataset
- Build a continuous evaluation pipeline that runs nightly on new resumes

**Definition of Done**:

- Monitoring dashboard shows real-time evaluation metrics
- Alerts trigger when performance degrades below acceptable thresholds
- Feedback loop successfully incorporates user corrections into dataset
- Continuous evaluation pipeline runs automatically and reports results

---

## Final Project Deliverable

**Resume Optimization Agent** with:

- ✅ Robust retrieval system (hybrid search + reranking)
- ✅ Complete LangGraph workflow (conditional routing, parallel execution,
  human-in-loop)
- ✅ Comprehensive evaluation system (LangSmith integration, custom metrics, A/B
  testing)
- ✅ Production-ready monitoring and feedback loops

---

## Success Metrics

- **Phase 1**: Retrieval system achieves >85% relevance on test queries
- **Phase 2**: Workflow processes resumes end-to-end with <5% error rate
- **Phase 3**: Evaluation system tracks all metrics and identifies improvement
  opportunities

---

## Resources & Tools

- **LangChain**: Document loaders, embeddings, vector stores, retrieval chains
- **LangGraph**: State management, conditional edges, human-in-loop
- **LangSmith**: Tracing, datasets, evaluation, monitoring
- **Vector Stores**: Chroma (local) or Pinecone (cloud)
- **Reranking**: Cohere Rerank API or sentence-transformers
- **Embeddings**: OpenAI, Cohere, or HuggingFace

---

## Daily Workflow Template

1. **Review** (10 min): Review previous day's Definition of Done
2. **Build** (80 min): Implement day's tasks
3. **Test** (20 min): Verify Definition of Done criteria
4. **Document** (10 min): Log learnings and blockers

---

**Remember**: Each day builds on the previous. If you fall behind, prioritize
completing the Definition of Done over perfection. The goal is consistent
progress, not perfection.

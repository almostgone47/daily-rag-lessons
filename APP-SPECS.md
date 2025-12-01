# Job Blaster Application Specification

## Overview

**Job Blaster** is a comprehensive job application management platform that
combines a web application with a Chrome extension to streamline the job search
and application process. The platform helps users save jobs, generate tailored
resumes and cover letters using AI, autofill application forms, and track their
progress through the entire application lifecycle.

**Core Value Proposition:** Apply faster. Track everything.

---

## Application Architecture

### Three-Component System

1. **Web Application** (`/web`)

   - React + Vite + Tailwind CSS frontend
   - TanStack Query for data fetching
   - React Router for navigation
   - TypeScript throughout

2. **API Server** (`/api`)

   - Node.js + Express backend
   - Prisma ORM with PostgreSQL (via Supabase/Neon)
   - RESTful API design
   - AI integration (OpenAI/Gemini) for document generation

3. **Chrome Extension** (`/extension`)
   - Manifest V3
   - Content scripts for page interaction
   - Panel UI (iframe-based) for extension interface
   - ATS adapters for job board integration

---

## Core Features & User Flows

### Primary User Journey

1. **Job Discovery & Capture**

   - User lands on a job posting (Greenhouse, Lever, Ashby, LinkedIn, etc.)
   - Chrome extension detects job page and shows panel
   - User clicks "Save Job" → Extension scrapes job details (title, company,
     location, description, URL)
   - Job is saved to web app with status: `SAVED`

2. **Document Preparation**

   - User navigates to `/jobs/:jobId/prepare` in web app
   - AI generates tailored Resume and Cover Letter based on:
     - User's existing resume content
     - Job description
     - Company information
     - Role requirements
   - Documents are marked as "Active for this job" and linked via `linkedJobId`

3. **Application Autofill**

   - User returns to job application page
   - Extension panel shows "Autofill" button
   - Extension fills required fields: **first name, last name, email, phone**
   - User manually attaches the generated resume/cover letter PDFs
   - User submits application normally

4. **Application Tracking**
   - User marks application as `APPLIED`
   - System tracks status progression: `SAVED` → `APPLIED` → `INTERVIEW` →
     `OFFER` → `REJECTED`
   - Follow-up reminders and interview scheduling
   - Notes and activity logging

---

## Data Models & Schema

### Core Entities

#### User

- `id` (String, cuid)
- `email` (String, unique)
- `name` (String, optional)
- `supabaseUserId` (String, optional, for auth)
- `createdAt` (DateTime)

#### Job

- `id` (String, cuid)
- `userId` (String, foreign key)
- `title` (String)
- `company` (String)
- `url` (String)
- `normalizedUrl` (String, unique per user)
- `source` (String, optional - e.g., "greenhouse.io", "lever.co")
- `status` (JobStatus enum: `SAVED`, `APPLIED`, `INTERVIEW`, `OFFER`,
  `REJECTED`)
- `location` (String, optional)
- `locationCity`, `locationState`, `locationCountry` (String, optional)
- `isRemote` (Boolean)
- `descriptionHtml` (String, optional - full HTML job description)
- `descriptionText` (String, optional - plain text version)
- `salaryMin`, `salaryMax` (Int, optional)
- `salaryCurrency` (String, default: "USD")
- `salaryType` (SalaryType enum: `ANNUAL`, `HOURLY`, etc.)
- `deadline` (DateTime, optional)
- `postedAt` (DateTime, optional)
- `createdAt`, `updatedAt`, `lastActivityAt` (DateTime)

#### Resume

- `id` (String, cuid)
- `userId` (String, foreign key)
- `name` (String)
- `fileUrl` (String, optional - PDF storage URL)
- `content` (String, optional - plain text content)
- `contentJson` (Json, optional - structured JSON content)
- `linkedJobId` (String, optional - links resume to specific job)
- `isActive` (Boolean, default: true)
- `version` (Int, default: 1)
- `parentId` (String, optional - for version control)
- `aiFeedback` (String, optional - AI-generated suggestions)
- `meta` (Json, optional - theme, fontFamily, fontSize, accentColor)
- `createdAt`, `updatedAt` (DateTime)

#### ResumeSection

- `id` (String, cuid)
- `resumeId` (String, foreign key)
- `title` (String - e.g., "Experience", "Education", "Skills")
- `kind` (SectionKind enum, optional)
- `template` (SectionTemplate enum)
- `order` (Int)
- `content` (String, optional)
- `content_rich` (Json, optional)

#### ResumeItem

- `id` (String, cuid)
- `sectionId` (String, foreign key)
- `headline` (String - e.g., job title or degree)
- `subheadline` (String, optional - e.g., company or school)
- `startDate`, `endDate` (String, optional)
- `location` (String, optional)
- `bullets` (String[] - achievement bullets)
- `bullets_rich` (Json, optional)
- `links` (String[], optional)
- `tags` (String[], optional)
- `order` (Int)

#### CoverLetter

- `id` (String, cuid)
- `userId` (String, foreign key)
- `resumeId` (String, foreign key)
- `jobId` (String, optional - links to specific job)
- `name` (String)
- `content` (String)
- `contentJson` (Json, optional)
- `isActive` (Boolean, default: true)
- `aiFeedback` (String, optional)
- `meta` (Json, optional)
- `createdAt`, `updatedAt` (DateTime)

#### Application

- `id` (String, cuid)
- `userId` (String, foreign key)
- `jobId` (String, foreign key)
- `resumeId` (String, optional)
- `status` (AppStatus enum: `DRAFT`, `APPLIED`, `INTERVIEW`, `OA`, `OFFER`,
  `REJECTED`)
- `coverNote` (String, optional)
- `appliedAt` (DateTime, optional)
- `nextAction` (DateTime, optional - follow-up reminder)
- `nextFollowUpDate` (DateTime, optional)
- `notes` (String, optional)
- `decision` (ApplicationDecision enum, optional)
- `interviewAt` (DateTime, optional)
- `offerCapturedAt`, `offerRespondedAt` (DateTime, optional)
- `createdAt`, `updatedAt` (DateTime)

#### Interview

- `id` (String, cuid)
- `userId` (String, foreign key)
- `jobId` (String, foreign key)
- `applicationId` (String, optional, foreign key)
- `scheduledAt` (DateTime)
- `type` (InterviewType enum: `PHONE`, `VIDEO`, `ONSITE`, etc.)
- `notes` (String, optional)
- `outcome` (String, optional)

#### FollowUp

- `id` (String, cuid)
- `userId` (String, foreign key)
- `applicationId` (String, foreign key)
- `dueDate` (DateTime)
- `completedAt` (DateTime, optional)
- `notes` (String, optional)
- `type` (FollowUpType enum)

#### CompanyResearch

- `id` (String, cuid)
- `userId` (String, foreign key)
- `company` (String)
- `content` (String - research notes)
- `createdAt`, `updatedAt` (DateTime)

---

## AI Integration & Document Generation

### AI Features

1. **Resume Customization**

   - Analyzes job description for keywords and requirements
   - Suggests modifications to resume content (JSON patches)
   - Reorders experience sections based on relevance
   - Optimizes bullet points with quantifiable achievements
   - ATS keyword optimization

2. **Cover Letter Generation**

   - Creates personalized cover letters for specific jobs
   - Incorporates company research and job requirements
   - Highlights relevant experience from resume
   - Maintains professional tone and structure

3. **AI Assistant Panel**

   - Docked side panel in document builders
   - Per-section chat threads for resumes/cover letters
   - Suggestion cards with Insert/Append/Replace/Copy actions
   - Diff view for changes
   - Context controls (chips) for:
     - Company Research
     - Job Description
     - Achievement Library
     - Interview Notes
     - Role Tech Signals

4. **Interview Preparation**
   - Generates practice questions based on job description
   - Provides answer suggestions
   - Company-specific insights

### AI Provider Configuration

- Supports **OpenAI** and **Gemini** with unified interface
- Configurable via environment variables:
  - `AI_PROVIDER`: `openai` | `gemini` | `auto`
  - `AI_MODEL`: optional (defaults per provider)
  - `OPENAI_API_KEY`: required for OpenAI
  - `GEMINI_API_KEY`: required for Gemini

### AI Prompt Context

When generating documents, AI receives:

- **Required Context:**

  - Active section text (for resumes/cover letters)
  - Job description (full HTML and plain text)
  - Document style rules
  - Current draft summary

- **Optional Context (via chips):**
  - Company research notes
  - Scraped company facts
  - Achievement library
  - Interview notes
  - Skills inventory
  - Portfolio links

---

## API Endpoints

### Core Endpoints

#### Jobs

- `POST /api/jobs` - Create new job
- `GET /api/jobs` - List user's jobs (with filters: status, location, etc.)
- `GET /api/jobs/:id` - Get job details
- `PATCH /api/jobs/:id` - Update job
- `DELETE /api/jobs/:id` - Delete job
- `POST /api/extension/jobs` - Extension-specific job creation

#### Resumes

- `POST /api/resumes` - Create resume
- `GET /api/resumes` - List user's resumes
- `GET /api/resumes/:id` - Get resume details
- `PATCH /api/resumes/:id` - Update resume
- `DELETE /api/resumes/:id` - Delete resume
- `POST /api/resumes/:id/customize` - Customize resume for job
- `GET /api/resumes/:id/export` - Export resume as PDF

#### Cover Letters

- `POST /api/cover-letters` - Create cover letter
- `GET /api/cover-letters` - List user's cover letters
- `GET /api/cover-letters/:id` - Get cover letter details
- `PATCH /api/cover-letters/:id` - Update cover letter
- `POST /api/cover-letters/generate` - Generate cover letter for job

#### Applications

- `POST /api/applications` - Create application
- `GET /api/applications` - List user's applications
- `GET /api/applications/:id` - Get application details
- `PATCH /api/applications/:id` - Update application status
- `POST /api/extension/applications` - Mark as applied from extension

#### AI Endpoints

- `POST /api/ai/resume/suggestions` - Get AI suggestions for resume
- `POST /api/ai/cover-letter/generate` - Generate cover letter
- `POST /api/ai/chat` - AI chat endpoint (for assistant panel)
- `POST /api/ai/ingest` - Ingest document to vector store
- `POST /api/ai/interview-prep` - Generate interview questions

#### Extension Endpoints

- `GET /api/extension/profile` - Get user profile for autofill
- `POST /api/extension/jobs` - Save job from extension
- `POST /api/extension/applications` - Mark application from extension

---

## Chrome Extension Architecture

### Components

1. **Background Service Worker** (`background.js`)

   - Handles extension lifecycle
   - Manages API communication
   - Stores configuration

2. **Content Script** (`content.js`)

   - Injected into job board pages
   - Detects ATS type (Greenhouse, Lever, Ashby)
   - Scrapes job information
   - Handles form field detection and autofill
   - Injects panel iframe

3. **Panel UI** (`panel.html`, `panel.js`, `panel.css`)

   - Floating iframe overlay on job pages
   - Shows job information
   - Provides "Save Job", "Autofill", "Open in App" buttons
   - Displays confidence scores for field detection

4. **ATS Adapters**
   - `BaseAdapter` - Common functionality
   - `GreenhouseAdapter` - Greenhouse-specific selectors
   - `LeverAdapter` - Lever-specific selectors
   - `AshbyAdapter` - Ashby-specific selectors
   - Adapter registry for automatic selection

### Extension Flow

1. User lands on job page → Content script detects ATS type
2. Content script injects panel iframe
3. Panel fetches user profile from `/api/extension/profile`
4. User clicks "Save Job" → Content script scrapes job data → POSTs to
   `/api/extension/jobs`
5. User clicks "Autofill" → Content script fills required fields (first, last,
   email, phone)
6. User clicks "Open in App" → Opens web app to `/jobs/:jobId`

---

## Key Technical Details

### Authentication

- Supabase Auth integration
- Cookie-based sessions (`jb_sess`)
- Extension uses `credentials: 'include'` for API calls

### Data Storage

- **Database:** PostgreSQL (via Supabase or Neon)
- **File Storage:** Supabase Storage (for resume/cover letter PDFs)
- **Vector Store:** (Future) For RAG capabilities

### Document Formats

- **Resumes:** Structured JSON (`contentJson`) + PDF export
- **Cover Letters:** Plain text (`content`) + JSON (`contentJson`) + PDF export
- **Job Descriptions:** HTML (`descriptionHtml`) + plain text
  (`descriptionText`)

### Version Control

- Resumes support versioning via `parentId` relationship
- Each version can be linked to a specific job via `linkedJobId`
- Version history tracked in database

---

## RAG API Requirements

### Use Cases for RAG Integration

1. **Job Description Analysis**

   - Extract key requirements, skills, and qualifications
   - Identify company culture signals
   - Parse salary ranges and benefits
   - Extract location and remote work preferences

2. **Resume Matching & Optimization**

   - Match user's resume against job requirements
   - Identify missing keywords
   - Suggest relevant experience to highlight
   - Quantify achievements based on job context

3. **Cover Letter Personalization**

   - Generate company-specific opening paragraphs
   - Incorporate company research and values
   - Match tone to company culture
   - Reference specific job requirements

4. **Interview Preparation**

   - Generate role-specific interview questions
   - Provide answer templates based on user's experience
   - Company-specific insights and talking points
   - Behavioral question preparation

5. **Application Tracking Intelligence**
   - Suggest follow-up actions based on application status
   - Recommend similar jobs based on saved applications
   - Identify patterns in successful applications

### Data to Index for RAG

1. **User's Resume Content**

   - Structured sections (Experience, Education, Skills)
   - Achievement bullets
   - Skills and technologies
   - Work history and dates

2. **Job Descriptions**

   - Full HTML and plain text descriptions
   - Requirements and qualifications
   - Company information
   - Role responsibilities

3. **Company Research**

   - User-generated research notes
   - Scraped company information
   - Company values and culture
   - Recent news and updates

4. **Application History**

   - Past applications and outcomes
   - Interview notes and feedback
   - Successful application patterns
   - Rejection reasons (if available)

5. **Cover Letters**

   - Generated cover letter templates
   - Successful cover letter patterns
   - Company-specific openings

6. **Interview Data**
   - Interview questions and answers
   - Company-specific insights
   - Interview outcomes

### RAG Query Patterns

1. **Semantic Search**

   - "Find jobs similar to [job title] at [company]"
   - "What skills are required for [role]?"
   - "What are common interview questions for [role]?"

2. **Contextual Retrieval**

   - "Retrieve job description for job ID: [id]"
   - "Get user's resume sections relevant to [job requirement]"
   - "Find company research for [company name]"

3. **Hybrid Search**
   - Combine semantic search with metadata filters (userId, jobId, etc.)
   - Filter by date ranges, status, location
   - Rank by relevance and recency

### RAG API Endpoints (Proposed)

- `POST /api/rag/query` - General semantic search
- `POST /api/rag/jobs/similar` - Find similar jobs
- `POST /api/rag/resume/match` - Match resume to job
- `POST /api/rag/cover-letter/context` - Get context for cover letter generation
- `POST /api/rag/interview/prep` - Get interview preparation context
- `POST /api/rag/ingest/job` - Ingest job description
- `POST /api/rag/ingest/resume` - Ingest resume content
- `POST /api/rag/ingest/company` - Ingest company research

---

## Development Context

### Technology Stack

- **Frontend:** React 18, TypeScript, Vite, Tailwind CSS, TanStack Query
- **Backend:** Node.js, Express, TypeScript, Prisma
- **Database:** PostgreSQL (Supabase/Neon)
- **Storage:** Supabase Storage
- **AI:** OpenAI GPT-4, Google Gemini
- **Extension:** Chrome Manifest V3, TypeScript
- **Testing:** Playwright (E2E), Vitest (unit)

### Key Design Principles

1. **User Control**

   - No auto-submit of applications
   - No automatic file selection (browser security)
   - All AI suggestions require explicit user action
   - Version control for all documents

2. **Privacy & Security**

   - CSP-safe content scripts
   - No page-world script injection
   - User data isolation (userId scoping)
   - Secure cookie handling

3. **Extensibility**

   - Adapter pattern for ATS systems
   - Plugin architecture for AI providers
   - Modular document generation
   - Flexible section system for resumes

4. **Performance**
   - Efficient database indexing
   - Lazy loading of document content
   - Optimistic UI updates
   - Caching strategies

---

## Future Enhancements (Roadmap)

- **v0.1** ✅ MVP: Greenhouse autofill, Save Job, Prepare Docs, Mark as Applied
- **v0.2** Guided file attach UX, improved selectors
- **v0.3** Lever adapter
- **v0.4** Ashby adapter
- **v0.5** Telemetry, per-site adapters, batch actions
- **Future:** RAG-powered job matching, intelligent application suggestions,
  automated follow-up generation

---

## Notes for RAG Tutorial Development

When creating RAG tutorials for this application:

1. **Start with Job Descriptions**

   - Most structured and abundant data
   - Clear use case: extract requirements, match to resumes
   - Good for teaching embedding, chunking, and retrieval

2. **Progress to Resume Matching**

   - More complex: structured data (JSON) + unstructured text
   - Teaches hybrid search (semantic + metadata filtering)
   - Demonstrates relevance scoring

3. **Cover Letter Generation**

   - Multi-document context (resume + job + company research)
   - Teaches context window management
   - Demonstrates prompt engineering with retrieved context

4. **Interview Preparation**

   - Query-time generation with RAG
   - Dynamic context assembly
   - Multi-turn conversation patterns

5. **Application Intelligence**
   - Time-series data analysis
   - Pattern recognition across applications
   - Recommendation systems

### Practical Tutorial Examples

- **Tutorial 1:** Build a job description analyzer that extracts skills and
  requirements
- **Tutorial 2:** Create a resume-to-job matcher with relevance scoring
- **Tutorial 3:** Build a cover letter generator that uses company research
- **Tutorial 4:** Implement interview question generation with job context
- **Tutorial 5:** Create a similar jobs finder using semantic search
- **Tutorial 6:** Build a hybrid search system (semantic + metadata)
- **Tutorial 7:** Implement multi-document RAG for cover letters
- **Tutorial 8:** Create an application success predictor
- **Tutorial 9:** Build a personalized job recommendation system
- **Tutorial 10:** Implement real-time document ingestion pipeline

---

## Environment Variables

### API Server

- `DATABASE_URL` - PostgreSQL connection string
- `DIRECT_URL` - Direct database connection (for migrations)
- `AI_PROVIDER` - `openai` | `gemini` | `auto`
- `OPENAI_API_KEY` - OpenAI API key
- `GEMINI_API_KEY` - Gemini API key
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_KEY` - Supabase service role key
- `COOKIE_SECRET` - Session cookie encryption secret

### Web App

- `VITE_API_URL` - API server URL
- `VITE_SUPABASE_URL` - Supabase project URL
- `VITE_SUPABASE_ANON_KEY` - Supabase anonymous key

---

## File Structure Reference

```
job-blaster/
├── api/                    # Backend API server
│   ├── src/
│   │   ├── routes/         # API route handlers
│   │   ├── lib/            # Shared utilities
│   │   └── llm.ts          # AI/LLM integration
│   └── prisma/             # Database schema and migrations
├── web/                    # Frontend React app
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── lib/            # Utilities and helpers
│   │   └── types.ts        # TypeScript types
│   └── tests/              # E2E tests (Playwright)
├── extension/              # Chrome extension
│   ├── src/
│   │   ├── adapters/       # ATS adapters
│   │   ├── panel/          # Panel UI
│   │   └── content.js      # Content script
│   └── manifest.json       # Extension manifest
└── docs/                   # Documentation and specs
```

---

This specification should provide comprehensive context for building a RAG API
that integrates seamlessly with Job Blaster's existing architecture and enhances
its AI-powered features.

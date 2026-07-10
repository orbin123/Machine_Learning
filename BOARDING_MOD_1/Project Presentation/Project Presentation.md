# Project Presentation Preparation

> **Project Status:** ✅ Live and deployed

---

# LingosAI – AI-Powered Personalized English Learning Platform

## Live Demo
- **Website:** https://www.lingosai.com
- **GitHub Repository:** https://github.com/orbin123/lingos-ai

---

# Project Overview

## What is LingosAI?

LingosAI is a full-stack, AI-powered English tutoring platform designed for non-native English speakers who want to develop career-ready communication skills.

The platform delivers a structured, multi-week curriculum through a real-time, chat-based learning experience. Each lesson follows a consistent learning cycle:

1. Teach
2. Task
3. Evaluate
4. Feedback

An agent-based AI system dynamically generates personalized learning activities based on each learner's strengths and weaknesses.

---

# Key Features

## Personalized AI Learning

- AI-generated lessons
- Personalized learning paths
- Writing practice
- Speaking practice
- Reading practice
- Listening practice

---

## Structured Learning Workflow

Each lesson follows the lifecycle:

- Teach
- Task
- Evaluate
- Feedback

---

## Agent-Based AI Architecture

Multiple specialized AI agents collaborate to deliver personalized learning.

### AI Agents

- Teacher Agent
- Task Generator Agent
- Evaluator Agent
- Feedback Agent
- Planner Agent

---

## RAG-Based Feedback Memory

The platform uses **Retrieval-Augmented Generation (RAG)** to remember a learner's previous mistakes.

This enables the AI tutor to:

- Recall recurring mistakes
- Track learner progress
- Personalize future lessons
- Reinforce weak areas over time

---

## Deterministic Scoring Engine

Unlike the AI-generated feedback, learner scoring is handled by a deterministic engine with **no LLM involvement**.

### Benefits

- Reproducible scores
- Explainable evaluation
- Consistent grading
- Transparent progress tracking

Scores are calculated across **seven measurable English sub-skills** and converted into an overall **0–10 dashboard score**.

---

# System Architecture

## Backend

### Framework

- FastAPI (Python 3.11+)

### Database Layer

- SQLAlchemy 2.0 (ORM)
- Alembic (Database Migrations)

### Architecture

Layered architecture:

```
Routes
    ↓
Service
    ↓
Repository
```

### Features

- JWT Authentication
- Google OAuth
- Role-Based Access Control
  - Learner
  - Admin
  - Super Admin
- Deterministic Scoring Engine

### Development Tools

- uv (Package Management)
- Ruff (Linting & Formatting)
- mypy (Static Type Checking)

---

# Frontend

### Framework

- Next.js 16 (App Router)
- React 19
- TypeScript (Strict Mode)

### Styling

- Tailwind CSS v4

### State Management

- TanStack Query
- Zustand

### Forms & Validation

- React Hook Form
- Zod

### Additional Features

- WebSocket-based real-time chat
- PDF Export using jsPDF

---

# AI & NLP Layer

## LLM Orchestration

Built using:

- LangChain
- LangGraph

### AI Components

- Teacher
- Planner
- Task Generator
- Evaluator
- Feedback Agent

---

## Feedback Memory

### Technology

- Pinecone Vector Database

### Purpose

- Long-term learner memory
- Personalized feedback
- Context-aware coaching

---

## Provider Abstraction

All AI capabilities are accessed through shared interfaces, making providers interchangeable without changing application logic.

Supported capabilities include:

- Large Language Models (LLMs)
- Text-to-Speech (TTS)
- Speech-to-Text (STT)
- Embeddings
- Image Generation

---

# External APIs & Services

| Service | Purpose |
|----------|---------|
| OpenAI API | Task generation, answer evaluation, feedback generation, TTS/STT |
| Azure Cognitive Services | Pronunciation assessment |
| Pinecone | Vector database for RAG |
| Razorpay | Subscription billing |
| Resend / AWS SES | Transactional emails |
| LangSmith | LLM tracing and evaluation |
| Sentry | Error monitoring |
| Google OAuth | Social login |

---

# Infrastructure

## Frontend

- Vercel
- Edge caching
- Preview deployments

---

## Backend

- AWS ECS Fargate
- Docker containers
- Amazon ECR
- Immutable image tags

---

## Database

- PostgreSQL 16
- AWS RDS

---

## Cache & Queue

- Redis
- Session caching
- Rate limiting
- Celery Broker

---

## Object Storage

- AWS S3
- CDN

Used for:

- Generated audio
- Generated images

---

## Infrastructure as Code

- Terraform
- GitHub Environments

---

## Local Development

Docker Compose for:

- PostgreSQL
- Redis

Application services run directly on the host machine.

---

# CI/CD Pipeline

## GitHub Actions

Automated workflows include:

- Ruff linting
- mypy type checking
- Unit tests
- Integration tests
- Alembic migration replay
- Coverage gate
- Next.js build
- OpenAPI drift detection
- Docker image build
- Trivy security scanning

---

## Deployment Pipeline

```
Pull Request
        ↓
GitHub Actions
        ↓
Quality Checks
        ↓
Merge to Main
        ↓
Docker Build
        ↓
Push to Amazon ECR
        ↓
AWS ECS Rolling Deployment
        ↓
Smoke Tests
        ↓
Automatic Rollback (if deployment fails)
```

---

# Code Quality

- DCO sign-off required for every commit
- All CI checks must pass before merging
- Automated security scanning
- Strict type checking
- Consistent code formatting

---

# Testing

## Backend

- pytest
- Unit Tests
- Integration Tests
- SQLite (Unit Testing)
- PostgreSQL (Integration Testing)
- LLM collaborators replaced with stubs

---

## Frontend

- Vitest
- React Testing Library
- MSW (API Mocking)

---

# Tech Stack

## Backend

- FastAPI
- Python 3.11+
- SQLAlchemy
- Alembic
- PostgreSQL
- Redis

---

## Frontend

- Next.js 16
- React 19
- TypeScript
- Tailwind CSS
- TanStack Query
- Zustand

---

## AI & ML

- LangChain
- LangGraph
- OpenAI
- Pinecone
- Azure Speech

---

## Cloud & DevOps

- AWS ECS
- Amazon ECR
- AWS RDS
- AWS S3
- Terraform
- Docker
- GitHub Actions
- Vercel

---

# Highlights

- ✅ Full-stack AI-powered learning platform
- ✅ Real-time chat-based lesson interface
- ✅ Personalized learning using AI agents
- ✅ RAG-powered feedback memory
- ✅ Deterministic and explainable scoring engine
- ✅ Subscription-based SaaS model
- ✅ Admin dashboard
- ✅ Production deployment
- ✅ Automated CI/CD pipeline
- ✅ Comprehensive testing strategy
- ✅ Cloud-native architecture
- ✅ Live project available for demonstration

---

# Project Links

## Website
https://www.lingosai.com

## GitHub Repository
https://github.com/orbin123/lingos-ai
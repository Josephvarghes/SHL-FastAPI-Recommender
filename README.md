# 🔍 SHL Assessment Recommender (FastAPI API)

This project is a **Generative AI-powered Recommender System** that suggests the most relevant **SHL assessments** based on a user’s job description or job role input. It's built using **FastAPI**, leveraging semantic search (Sentence Transformers + Cosine Similarity), and designed for easy integration into any HR platform or hiring pipeline.

---

## 🎯 Business Perspective

### ❓ Problem
Manually mapping job roles to the right assessment is time-consuming and error-prone for HR professionals and recruiters.

### 💡 Solution
This project solves that with:
- 🧠 AI-powered semantic search (no keyword matching)
- 🔎 Smart recommendation of relevant assessments
- ⚡ Fast, lightweight API (ready for integration)
- 🎯 Precision-tested (Recall@5 = 1.0, MAP@5 = 1.0)

> ✅ Ideal for ATS platforms, career portals, or enterprise HR solutions.

---

## ⚙️ Features

- Accepts natural language job role or JD input
- Recommends top N relevant SHL assessments
- RESTful API built using FastAPI
- Returns clean JSON output for frontend or API use
- Ready for deployment (e.g., Render.com, Railway, Replit)

---

## 🚀 Getting Started

### 🔧 1. Install Requirements

```bash
pip install -r requirements.txt

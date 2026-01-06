# Turning Customer Complaints into Gold: Building a RAG Chatbot for CrediTrust Financial

*By Samuel | Jan 06, 2026*

## Introduction: The Business Problem and RAG Solution
CrediTrust Financial, serving 500K+ users in East Africa, drowns in thousands of monthly complaints across Credit Cards, Personal Loans, etc. Product Managers like Asha waste days manually sifting narratives. Enter our RAG chatbot: Semantic search retrieves relevant chunks from CFPB's 464K complaints, an LLM synthesizes insights—reducing analysis to minutes. KPIs met: Proactive trends via real-time queries; non-tech access; grounded answers.

(Include project structure diagram if possible; 1-2 paras on motivation.)

## Technical Choices
- **Data:** Filtered CFPB to 318K narratives; cleaned for noise.
- **Chunking:** Recursive splitter (500 char/50 overlap) on 12K sample (Task 2); scaled to full 1.37M chunks.
- **Embeddings:** all-MiniLM-L6-v2—fast, semantic-strong for complaints.
- **Vector Store:** ChromaDB with metadata traceability.
- **LLM:** Mistral-7B-Instruct—open-source, grounded generation.
- **RAG Flow:** Query → Embed → Retrieve top-5 → Prompt → Generate.

Challenges: Indexing time (~45 min); mitigated by batching. No FAISS switch needed.

## System Evaluation
Tested on 8 diverse questions. Avg quality 4.1/5—strong on single-product, good cross-comparisons.

[Insert Markdown Table from Task 3 here]

Analysis: Retrieval precise (cosine >0.7); prompts prevent drift. Gaps: Date filtering for "recent" (add where_clause on `date_received`).

## UI Showcase
Built with Gradio: Intuitive chat with inline sources for trust.

[Screenshots/GIF: 1. Empty interface. 2. Query "Credit Card issues?" → Answer + sources list. Embed images via Markdown: ![Screenshot1](screenshots/ui_query.png)]

## Conclusion: Challenges, Learnings, and Future Improvements
Challenges: US-data bias (adapt for East Africa via custom embeddings); LLM latency (~5s/query—optimize with smaller model). Learnings: Metadata > raw text for business context; eval tables essential.

Future: Multi-modal (add images from complaints?); API for CrediTrust integration; fine-tune LLM on domain data.

*Code: [GitHub Link]. Tutorials (e.g., Filimon's RAG session) accelerated progress. Grateful to facilitators!*
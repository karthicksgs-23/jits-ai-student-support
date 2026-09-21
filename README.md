# JITS AI Student Support

An AI-powered conversational student support system for **Jyothishmathi Institute of Technology and Science (JITS)**.

The system answers student queries using official JITS academic regulations and the official JITS website. It combines a multi-agent architecture with Hybrid RAG, Self-RAG validation, guardrails, website fallback, conversational memory, logging, and evaluation.

---

## Features

- Conversational student support
- CrewAI multi-agent architecture
- Hybrid RAG using:
  - FAISS semantic search
  - BM25 keyword search
- Self-RAG evidence validation
- Automatic fallback to official JITS website
- Serper-powered website discovery
- Scrapling-based website content fetching
- Input guardrails
- Output guardrails
- Conversation context handling
- Daily interaction logging
- Streamlit chat UI
- Evaluation framework for:
  - routing accuracy
  - keyword coverage
  - answer quality checks

---

## System Architecture

```text
                    Student
                       │
                       ▼
               Streamlit Chat UI
                       │
                       ▼
              Conversation Manager
                       │
                       ▼
                Input Guardrail
                       │
                       ▼
                    Agent 1
                       │
              Hybrid RAG Retrieval
                 FAISS + BM25
                       │
                       ▼
                  Self-RAG
                 /        \
        SUFFICIENT        INSUFFICIENT
            │                  │
            ▼                  ▼
      Agent 1 Answer        Agent 2
                              │
                              ▼
                            Serper
                              │
                              ▼
                     Official JITS Website
                              │
                              ▼
                        Agent 2 Answer
                              │
                 ┌────────────┘
                 ▼
            Output Guardrail
                 │
                 ▼
              Agent 3
                 │
                 ▼
            Daily Log File
                 │
                 ▼
          Final Student Answer
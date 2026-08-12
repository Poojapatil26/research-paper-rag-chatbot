#  Research Paper Chatbot using RAG and LLM

An end-to-end intelligent question-answering system for research papers using Retrieval-Augmented Generation (RAG), FAISS, Sentence Transformers, Hugging Face LLMs, LangChain, and Gradio.

##  Project Overview

Research papers contain large volumes of technical information distributed across sections such as abstracts, methodology, experiments, results, and conclusions. Locating specific information manually can be time-consuming.

This project implements a Retrieval-Augmented Generation (RAG) based Research Paper Chatbot that enables users to upload a research paper in PDF format and interact with its content through natural-language queries.

Instead of relying solely on the language model's parametric knowledge, the system retrieves relevant sections from the uploaded document and uses them as contextual evidence for answer generation.

---

##  Objectives

- Enable natural-language interaction with research papers.
- Reduce the time required to manually search lengthy documents.
- Retrieve semantically relevant document sections.
- Generate context-grounded responses using an LLM.
- Provide source-page references for retrieved information.
- Reduce hallucination by restricting generation to retrieved context.

---

##  System Architecture

The complete processing pipeline follows:

**PDF → Text Extraction → Document Chunking → Semantic Embeddings → FAISS Vector Index → Similarity Retrieval → Context Construction → LLM Generation → Answer + Source Pages**

### Architecture Flow

```text
                 Research Paper PDF
                         │
                         ▼
                  PDF Text Extraction
                         │
                         ▼
                    Text Chunking
                         │
                         ▼
              Sentence Transformer
                    Embeddings
                         │
                         ▼
                   FAISS Index
                         │
                         │
                         │
User Question ───────────┘
                         │
                         ▼
                 Semantic Retrieval
                         │
                         ▼
                 Top-k Relevant Chunks
                         │
                         ▼
                 Context Construction
                         │
                         ▼
                  Hugging Face LLM
                         │
                         ▼
                  Generated Answer
                         │
                         ▼
                  Source Page IDs

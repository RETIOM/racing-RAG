# Formula Student Rules Assistant Documentation

## Introduction

The Formula Student Rules Assistant is an AI-powered system designed to help Formula Student teams quickly navigate and understand the complex rulebook governing the competition. Formula Student is an international engineering competition where students design, build, and compete with formula-style race cars. The rulebook contains hundreds of detailed technical regulations that teams must comply with.

This project uses advanced natural language processing and information retrieval techniques to:
1. Parse and structure the rulebook into a hierarchical format
2. Generate embeddings for efficient similarity search
3. Provide accurate answers to rule-related queries
4. Suggest potential new regulations based on questions

The system combines several state-of-the-art techniques including HyDE (Hypothetical Document Embeddings), RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval), and semantic search to deliver precise rule information to users.

## Code Structure

### 1. `HyDE.py` - Hypothetical Document Embeddings Generator
**Purpose**: Generates hypothetical rule embeddings based on user queries to improve retrieval accuracy.

Key components:
- Uses Google's Gemini model to generate hypothetical rule responses
- Creates embeddings of these hypothetical responses
- Returns averaged embeddings for better semantic matching

This implements the HyDE technique where instead of searching with the original query, we search with an AI-generated "hypothetical" answer, which often matches better with actual document embeddings.

### 2. `ingest.py` - Rulebook Processing Pipeline
**Purpose**: Processes raw PDF rulebooks into a structured, searchable format.

Key functions:
- `prep_pdf()`: Extracts and cleans text from PDF rulebooks
- `clean_abbrev()`: Expands all technical abbreviations in the text
- `create_tree()`: Builds a hierarchical tree structure of rules
- `embed_summarize()`: Generates summaries and embeddings for rule sections
- `save_tree()`: Serializes the processed rule structure to disk

The processing creates a multi-level tree where each node contains:
- Content (original text or summary)
- Embedding vector
- Child nodes (for hierarchical structure)

### 3. `retrieve.py` - Semantic Search Component
**Purpose**: Retrieves relevant rule sections based on semantic similarity.

Key features:
- Implements cosine similarity for vector matching
- Performs hierarchical search through the rule tree
- Returns top-k most relevant rule sections
- Uses both content and embedding vectors for retrieval

The retrieval system searches through each level of the hierarchy, maintaining the top matches at each level to ensure comprehensive coverage of relevant rules.

### 4. `main.py` - User Interface and Integration
**Purpose**: Provides a Gradio-based interface for user interaction and integrates all components.

Key functionality:
- Loads pre-processed rule data
- Handles user queries
- Coordinates between HyDE generation and retrieval
- Presents results in a clean interface
- Offers option to see raw context or generated answers

### 5. `pure_raptor.ipynb` - Advanced Clustering Implementation
**Purpose**: Experimental implementation of RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) clustering.

Key algorithms:
- UMAP for dimensionality reduction
- Gaussian Mixture Models for clustering
- Hierarchical clustering at multiple levels
- Recursive tree construction

This notebook contains the core algorithms for automatically building hierarchical structures from unstructured text using semantic clustering techniques.

## System Workflow

1. **Ingestion Phase**:
   - PDF rulebook is processed and cleaned
   - Text is split into hierarchical sections
   - Each section is summarized and embedded
   - Structure is saved for later use

2. **Query Phase**:
   - User submits a question
   - System generates hypothetical answer embedding (HyDE)
   - Retrieves most relevant rule sections
   - Either returns raw context or generates an answer

3. **Advanced Features**:
   - Automatic abbreviation expansion
   - Multi-level semantic search
   - Dynamic rule generation capability

The system is designed to handle the complex, technical nature of Formula Student rules while providing fast, accurate access to the information teams need to ensure compliance and make design decisions.
# Langchain Multi-Agent Research System

A research assistant powered by LangChain and LLMs that automates the end-to-end workflow of finding, reading, synthesizing, and critiquing information on a topic. The system combines web search, academic paper discovery, PDF extraction, report generation, and opinionated review in a single multi-agent pipeline.

## Overview

This project is designed to simulate an autonomous research workflow:

- Search the web for recent and relevant sources
- Search academic literature using Semantic Scholar
- Scrape important webpages and extract readable content
- Read and process PDF-based research articles
- Combine findings into a structured research report
- Critique the generated output for clarity, relevance, and quality

The application is exposed through a Streamlit interface, making it easy to test and explore without building a full custom frontend.

## Key Features

- Multi-agent research workflow with specialized responsibilities
- Web search and academic paper retrieval
- PDF-based research extraction
- URL content scraping and summarization
- Structured report writing using LLM prompting
- Built-in critic/review stage for quality assessment
- Interactive UI powered by Streamlit

## Architecture Overview

The system follows a modular multi-agent architecture:

1. Search Agent
   - Uses general web search and academic literature search tools
   - Collects both public web sources and research articles

2. Reader Agent
   - Evaluates the most relevant URLs and PDF sources
   - Extracts readable text from webpages and PDFs

3. Writer Chain
   - Synthesizes the gathered content into a formal academic-style report
   - Follows a fixed structure: Introduction, Key Findings, Conclusion, and Sources

4. Critic Chain
   - Reviews the generated report
   - Produces a score, strengths, areas for improvement, and a verdict

5. Streamlit Frontend
   - Accepts user input
   - Launches the pipeline
   - Displays the research flow and final output

In simple terms, the project moves from raw information discovery to evidence synthesis and then to evaluation.

## Tech Stack

- Python
- Streamlit
- LangChain
- LangChain Community
- LangChain Core
- OpenAI-compatible LLM integration via OpenRouter
- Tavily for web search
- Semantic Scholar API for academic paper discovery
- BeautifulSoup and readability for webpage extraction
- trafilatura for article-level content extraction
- PyPDFLoader for PDF parsing
- Python-dotenv for environment configuration
- Requests and aiohttp for network access

## Project Structure

```text
Copy/
├── app.py
├── main.py
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── agents.py
│   ├── pipelines/
│   │   ├── __init__.py
│   │   └── pipelines.py
│   └── tools/
│       ├── __init__.py
│       └── tools.py
```

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Langchain-Multiagent-Research-System/Copy
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and add the required API keys:

```env
OPENROUTER_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
SEMTANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key
```

> Note: `SEMTANTIC_SCHOLAR_API_KEY` is optional depending on how the API is used in your environment. The application will still attempt to work without it if the service allows unauthenticated access.

## Running the App

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal in your browser.

## Example Usage

Open the app, enter a research topic such as:

- "Large language models in healthcare"
- "AI agents for scientific discovery"
- "Retrieval-augmented generation in enterprise systems"

The system will then:

- retrieve relevant sources,
- read the most promising content,
- write a structured research report,
- and provide feedback on the final result.

## How the Workflow Works

A typical research run looks like this:

```text
User Topic
   ↓
Search Agent
   ↓
Web + Academic Results
   ↓
Reader Agent
   ↓
Scraped Web Pages + PDF Content
   ↓
Writer Chain
   ↓
Research Report
   ↓
Critic Chain
   ↓
Feedback + Evaluation
```

This makes the project useful for research exploration, AI-assisted literature review, and rapid report generation from multiple evidence sources.

## Use Cases

- Academic research support
- Literature review automation
- Market or technology trend analysis
- AI-assisted article synthesis
- Multi-source report generation

## Best Practices

- Use specific, well-scoped research topics for better search quality
- Keep API keys in a `.env` file rather than hardcoding them
- Prefer trusted sources and verify the generated report before publishing
- Use the app as a research accelerator rather than a substitute for expert review

## Contributing

Contributions are welcome. If you want to improve the system, consider:

- improving source extraction quality,
- adding more research tools,
- enhancing prompt quality,
- refining the report evaluation stage,
- or improving the Streamlit UI.

## Notes

This project is a practical prototype for an autonomous research pipeline and is intended to demonstrate how multiple agents and tools can be orchestrated around a common research task.

If you are using this project for production or academic work, validate the outputs carefully, especially when dealing with citations, factual claims, or sensitive research topics.

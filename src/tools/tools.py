from langchain_community.document_loaders import PyPDFLoader
from langchain.tools import tool
import requests
from dotenv import load_dotenv
import os
from tavily import TavilyClient
import certifi
from rich import print
import time
from bs4 import BeautifulSoup
from readability import Document
import trafilatura
import re
import cloudscraper
import asyncio
import aiohttp
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
load_dotenv()
from langchain_core.output_parsers import StrOutputParser




os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


tavily = TavilyClient(api_key=TAVILY_API_KEY)
llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),  
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

# ==========================================
# TOOL 1
# ==========================================
# Web search takes a query and searches the web for reliable information
#  on the topic. Returns 

@tool
def web_search(query: str) -> str:
    """ Search the web for recent and reliable information on the topic.
      Returns Titles , URLs and content snippets. """

    results = tavily.search(query=query, max_results=5)
    out = []
    for r in results['results']:
        out.append(
            f"Title: {r['title']}\nURL: {r['url']}\nSnippet: {r['content'][:300]}\n"
        )
    return "\n----\n".join(out)



# The Academic Paper Search tool takes a query and searches Semantic Scholar for relevant academic papers.
#  It returns the title, authors, year, citation count, TLDR (if available), and a link to the paper on Semantic Scholar.
#  If a free PDF is available, it will also provide a direct link to the PDF.


def extract_free_pdf_url(paper_data: dict) -> str | None:
    """
    Parses Semantic Scholar paper data to find a free, open-access PDF link.
    Strictly ensures the returned URL is an actual PDF file or a trusted PDF host.
    """
    external_ids = paper_data.get("externalIds", {})
    
    # Priority 1: Check explicit IDs where we KNOW the exact PDF URL structure
    if isinstance(external_ids, dict):
        if "ArXiv" in external_ids:
            return f"https://arxiv.org/pdf/{external_ids['ArXiv']}.pdf"
            
        if "ACL" in external_ids:
            return f"https://aclanthology.org/{external_ids['ACL']}.pdf"
            
        if "PubMedCentral" in external_ids:
            return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{external_ids['PubMedCentral']}/pdf/"

    # Priority 2: Check Semantic Scholar's OA link, but VERIFY it is a PDF or trusted host
    open_access = paper_data.get("openAccessPdf")
    if open_access and isinstance(open_access, dict) and "url" in open_access:
        oa_url = open_access["url"]
        url_lower = oa_url.lower()
        
        # Accept if it explicitly has a .pdf extension
        if ".pdf" in url_lower:
            return oa_url
            
        # Accept if it is hosted on a trusted directory that serves raw PDFs
        trusted_pdf_paths = [
            "arxiv.org/pdf/", 
            "aclanthology.org/", 
            "ncbi.nlm.nih.gov/pmc/articles/"
        ]
        if any(trusted_path in url_lower for trusted_path in trusted_pdf_paths):
            return oa_url

    return None

# ==========================================
# TOOL 2
# ==========================================

# Query Generator (Pydantic Schema)
class SearchQueries(BaseModel):
    queries: list[str] = Field(
        description="A list of exactly 3 highly specific academic search queries."
    )

def generate_optimized_queries(user_topic: str, llm) -> list[str]:
    """Translates a broad topic into 3 optimized academic search queries."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert academic research librarian. 
        Generate exactly 3 specific, distinct search queries that would yield the best academic papers on Semantic Scholar based on the user's topic. 
        Make them sound like paper titles or specific methodologies. Just return the 3 queries."""),
        ("user", "{topic}")
    ])
    
    structured_llm = llm.with_structured_output(SearchQueries)
    query_chain = prompt | structured_llm
    return query_chain.invoke({"topic": user_topic}).queries


BASE_URL = "https://api.semanticscholar.org/graph/v1"
api_key = os.getenv("SEMTANTIC_SCHOLAR_API_KEY")
HEADERS = {"x-api-key": api_key} if api_key else {}
# Tool 2



async def search_academic_papers_async(
    query: str, session: aiohttp.ClientSession, limit: int = 5, year_range: str = None
) -> str:
    """Searches for academic papers on Semantic Scholar and extracts ArXiv PDF links.

    Args:
        query: The search term or research question.
        limit: Number of results to return (max 10).
        year_range: Optional filter, e.g., '2022-2025' or '2023-'.
    """

    params = {
        "query": query,
        "limit": min(limit, 10),
        "fields": "title,authors,year,citationCount,abstract,tldr,externalIds,openAccessPdf,url"
    }

    if year_range:
        params["year"] = year_range

    max_retries = 3
    for attempt in range(max_retries):
        # Using aiohttp async network request instead of requests.get
        async with session.get(f"{BASE_URL}/paper/search", params=params, headers=HEADERS) as response:
            if response.status == 200:
                data = await response.json()
                break  
            elif response.status == 429:
                if attempt < max_retries - 1:
                    # MUST use asyncio.sleep, not time.sleep!
                    await asyncio.sleep(2 ** (attempt + 1)) 
                    continue
                else:
                    return f"Error: 429 - Semantic Scholar rate limit exceeded."
            else:
                error_text = await response.text()
                return f"Error: {response.status} - {error_text}"
    else:
        # Fallback if loop finishes without breaking
        data = {}

    papers = data.get("data", [])
    if not papers:
        return f"No papers found for the query: '{query}'"

    results = []
    for p in papers: 
        authors = ", ".join([a["name"] for a in p.get("authors", []) if "name" in a])
        tldr = (
            p.get("tldr", {}).get("text")
            if p.get("tldr")
            else "No TLDR available."
        )

        pdf_url = extract_free_pdf_url(p)

        if not pdf_url:
            pdf_url = "No free PDF available (Not open access)."

        paper_url = p.get("url", "N/A")

        results.append(
            f"**{p.get('title')}** ({p.get('year', 'N/A')})\n"
            f"- Authors: {authors}\n"
            f"- Citations: {p.get('citationCount', 0)}\n"
            f"- TLDR: {tldr}\n"
            f"- Semantic Scholar URL: {paper_url}\n"
            f"- Open Access PDF: {pdf_url}\n"
        )

    return "\n---\n".join(results)





# Async Gather function
async def fetch_all_academic_papers_async(queries: list[str]) -> str:
    """Creates an async session and fetches all 3 queries at the exact same time."""
    combined_results = []
    
    # Open a single aiohttp session to reuse connections (much faster)
    async with aiohttp.ClientSession() as session:
        # Create a list of async tasks
        tasks = [
            search_academic_papers_async(query, session, limit=5) 
            for query in queries
        ]
        
        # Run them all concurrently using gather!
        # return_exceptions=True prevents one failed query from crashing the others
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format the combined results
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                combined_results.append(f"### Results for: '{query}' ###\nError: {str(result)}")
            else:
                combined_results.append(f"### Results for: '{query}' ###\n{result}")
                
    return "\n\n".join(combined_results)


@tool
def deep_academic_research(topic: str) -> str:
    """
    Searches for academic papers on Semantic Scholar.
    Use this tool whenever you need deep, academic literature on a topic.
    You just provide the broad topic, and this tool automatically expands it 
    into 3 specialized queries and fetches 15 papers asynchronously.
    """
    try:
        # 1. Expand the broad topic into 3 specific queries
        optimized_queries = generate_optimized_queries(topic, llm)
        
        # 2. Fetch all 15 results concurrently
        results = asyncio.run(fetch_all_academic_papers_async(optimized_queries))
        
        return f"Searched 3 specialized sub-topics: {optimized_queries}\n\n{results}"
    except Exception as e:
        return f"Deep research failed: {str(e)}"



@tool
def scrape_url(url: str) -> str:
    """
    Scrape and extract clean readable content from a URL.
    Uses multiple extraction strategies for better reliability.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.google.com/",
    }

    # Network Request
    try:
        # ── Fetch page ─────────────────────────────────────
        
        response = requests.get(
            url,
            headers=headers,
            timeout=15
        )

        response.raise_for_status()

        html = response.text

        # ──────────────────────────────────────────────────
        # Strategy 1 → trafilatura (BEST for articles/blogs)
        # ──────────────────────────────────────────────────
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False
        )

        if extracted and len(extracted.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', extracted)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 2 → readability
        # ──────────────────────────────────────────────────
        doc = Document(html)
        clean_html = doc.summary()

        soup = BeautifulSoup(clean_html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        if text and len(text.strip()) > 200:
            cleaned = re.sub(r'\s+', ' ', text)
            return cleaned[:5000]

        # ──────────────────────────────────────────────────
        # Strategy 3 → fallback full page extraction
        # ──────────────────────────────────────────────────
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup([
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form"
        ]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)

        cleaned = re.sub(r'\s+', ' ', text)

        if cleaned:
            return cleaned[:15000]

        return "Could not extract meaningful content from the page."

    except requests.exceptions.Timeout:
        return "Request timed out while scraping the URL."

    except requests.exceptions.HTTPError as e:
        return f"HTTP error occurred: {str(e)}"

    except Exception as e:
        return f"Could not scrape URL: {str(e)}"


# Create tool to read pdf url links



@tool
def read_pdf_from_url(pdf_url: str) -> str:
    """
    Downloads and extracts text content from an online PDF using its URL.
    If the URL is a webpage instead of a PDF, it automatically scrapes the web content.
    """

    clean_url = pdf_url.strip().lower()
    
    # Check if the URL actually points to a PDF. 
    # (Accounts for .pdf, PubMed's /pdf/ ending, and URL parameters like .pdf?download=1)
    is_pdf = (
        clean_url.endswith(".pdf") or 
        clean_url.endswith("/pdf/") or 
        clean_url.endswith("/pdf") or 
        ".pdf?" in clean_url
    )

    if not is_pdf:
        # Route non-PDF URLs (like DOIs or publisher pages) to the custom web scraper
        return scrape_url.invoke(pdf_url)

    # If it is a PDF, proceed with PyPDFLoader to extract text content
    try:
        # PyPDFLoader automatically handles web URLs by temporarily downloading the file
        loader = PyPDFLoader(pdf_url)
        pages = loader.load()
        
        if not pages:
            return f"Error: No content could be extracted from {pdf_url}."
            
        # Combine the text from all extracted pages
        full_text = "\n\n".join([page.page_content for page in pages])
        
        # Optional: Truncate if you are hitting token limits on your LLM's context window
        # (e.g., returning only the first 50,000 characters)
        return full_text[:30000] 
        
        # return full_text
        
    except Exception as e:
        # Ultimate fallback: If the PDF loader crashes for any reason, try scraping it anyway!
        fallback_text = scrape_url.invoke(pdf_url)
        return f"PyPDFLoader failed ({str(e)}). Fallback Scraper Result:\n\n{fallback_text}"
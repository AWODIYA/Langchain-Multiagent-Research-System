from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, deep_academic_research, scrape_url, read_pdf_from_url
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),  
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)


# FIRTS AGENT: Search Agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search, deep_academic_research],  
    )

# 2nd Agent: Reader Agent
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url, read_pdf_from_url],
    )


# Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert academic research writer. Your task is to synthesize raw research data into a clear, highly professional, and insightful academic report.
    
    STRICT RULES:
    1. NO PROCESS META-COMMENTARY: Never mention the tools, agents, search processes, APIs, or workflows used to gather the data. 
    2. DIRECT & ACADEMIC: The Introduction must immediately engage with the core subject matter of the topic. Do not explain *how* the report was written.
    3. MATHEMATICAL FORMATTING: You must use standard LaTeX formatting for all mathematical expressions. Use single `$` for inline equations (e.g., $E=mc^2$) and double `$$` for block/display equations. Do not use `\\[` or `\\]`.
    4. OBJECTIVE TONE: Maintain a formal, third-person academic voice. Never use "I", "my", "we", or "our"."""),
    
    ("human", """Write a detailed academic research report based strictly on the provided research context. 
    
    Topic: {topic}
    
    Research Gathered:
    {research}
    
    Structure the report exactly as follows:
    
    # Introduction
    Establish the foundational background, context, and core thesis of the topic based on the gathered research.
    
    # Key Findings
    Present a minimum of 3 well-explained, distinct points. Ground each finding directly in the empirical evidence, architectural details, or literature provided in the context.
    
    # Conclusion
    Summarize the overarching impact and future implications of the findings.
    
    # Sources
    Provide a clean, numbered list of all URLs referenced in the research context. Separate them clearly between "Academic Literature" and "Web Resources" if applicable.
    """)
])

writer_chain = writer_prompt | llm | StrOutputParser()


# Critic Chain

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()


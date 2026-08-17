from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain
from rich import print

def run_research_pipeline(topic : str) -> dict:

    state = {}

    #search agent working 
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
    "messages": [
        ("user", f"""Find recent, reliable and detailed information about: {topic} 
        You have two tools to use. The first is the 'web search tool' to do a general search, the result will usually have these parts:
         - Title
         - URL
         - Snippet

         GIVE THE 5 RESULT FROM THIS TOOL

        For the second tool 'deep academic papers', Use the 'deep_academic_research' tool to fetch 15 highly relevant academic papers. (Just pass the topic to it). also to get your results in the format it returns including:
        - The title of paper
        - Authors
        - Citations
        - TLDR
        - Semantic Scholar URL
        - Open Access PDF
        
        
        
       YOUR JOB IS TO STORE THE RESULTS IN SUCH A WAY THAT IT IS IDENTIFIABLE FOR:
            1. Web search results
            2. Academic literature results""")
    ]
})
    state["search_results"] = search_result['messages'][-1].content

    print("\n search result ",state['search_results'])


    #step 2 - reader agent 
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', please gather deeper content.\n\n"
            f"RULES FOR TOOL USAGE:\n"
            f"1. For results from the 'web search tool', identify the most relevant URLs and process them using the `scrape_url` tool.\n"
            f"2. For results from the 'search academic papers' tool, identify the most relevant 'Open Access PDF' URLs and process them using the `read_pdf_from_url` tool.\n"
            f"3. Do not mix these up. Never use scrape_url on a .pdf link, and never use read_pdf_from_url on a standard webpage.\n\n"
            f"Search Results:\n{state['search_results']}" 
        )]
    })

    state['scraped_content'] = reader_result['messages'][-1].content

    print("\nscraped content: \n", state['scraped_content'])

    # Step 3 - Writer Chain
    print("\n"+" ="*50)
    print("step 3 - Writer again is writing the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])

    # Step 4 - Critic Chain
    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])

    return state

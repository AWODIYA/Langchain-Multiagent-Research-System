import streamlit as st
import time

# Import your agent builders and chains
from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain

# ==========================================
# 1. PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Researcher Agent Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Injecting Custom CSS to match image_21cfe1.png
st.markdown("""
    <style>
    /* Global background and text */
    .stApp {
        background-color: #0B101E; /* Deep dark blue background */
        color: #E2E8F0;
    }
    
    /* Header styling */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    .description {
        text-align: center;
        color: #94A3B8;
        font-size: 1.1rem;
        max-width: 800px;
        margin: 0 auto 40px auto;
        line-height: 1.5;
    }

    /* Left Column Styling */
    .section-label {
        color: #38BDF8;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    
    /* Input field styling */
    div[data-baseweb="input"] {
        background-color: #1E293B !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="input"] input {
        color: #F8FAFC !important;
    }

    /* Gradient Primary Button */
    button[kind="primary"] {
        background: linear-gradient(90deg, #38BDF8, #818CF8, #A855F7) !important;
        color: white !important;
        border: none !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.3s !important;
    }
    button[kind="primary"]:hover {
        opacity: 0.9 !important;
    }

    /* Try suggestion buttons */
    .try-label {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 20px;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    button[kind="secondary"] {
        background-color: #162032 !important;
        border: 1px solid #1E293B !important;
        color: #94A3B8 !important;
        border-radius: 20px !important;
        padding: 4px 15px !important;
        font-size: 0.85rem !important;
        display: block !important;
        margin-bottom: 10px !important;
    }

    /* Pipeline Cards */
    .pipeline-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 20px;
    }
    .pipeline-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        transition: all 0.3s ease;
    }
    .pipeline-card.active {
        border: 1px solid #38BDF8;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }
    .pipeline-card.completed {
        border: 1px solid #10B981;
    }
    .card-left {
        display: flex;
        flex-direction: column;
    }
    .card-title-row {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .step-num {
        color: #38BDF8;
        font-weight: 700;
        font-size: 0.9rem;
    }
    .step-name {
        font-weight: 700;
        font-size: 1.1rem;
        color: #F1F5F9;
    }
    .step-desc {
        color: #64748B;
        font-size: 0.85rem;
        margin-top: 5px;
    }
    .status-badge {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
        padding: 4px 10px;
        border-radius: 4px;
        text-transform: uppercase;
    }
    .status-waiting { color: #475569; }
    .status-running { color: #F59E0B; animation: pulse 1.5s infinite; }
    .status-completed { color: #10B981; }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# 2. HELPER FUNCTIONS FOR UI COMPONENTS
# ==========================================
def render_pipeline_card(step_num, title, desc, status):
    """Generates the HTML for a single pipeline card based on its status."""
    
    # Determine CSS classes based on status
    card_class = "pipeline-card"
    status_class = "status-waiting"
    display_status = status
    
    if status == "RUNNING":
        card_class += " active"
        status_class = "status-running"
    elif status == "COMPLETED":
        card_class += " completed"
        status_class = "status-completed"

    html = f"""
    <div class="{card_class}">
        <div class="card-left">
            <div class="card-title-row">
                <span class="step-num">{step_num}</span>
                <span class="step-name">{title}</span>
            </div>
            <div class="step-desc">{desc}</div>
        </div>
        <div class="{status_class} status-badge">{display_status}</div>
    </div>
    """
    return html

def set_topic(new_topic):
    """Callback to set the text input value from suggestion buttons."""
    st.session_state.topic = new_topic


# ==========================================
# 3. MAIN APP LAYOUT & LOGIC
# ==========================================
def main():
    # Initialize session state for the input
    if "topic" not in st.session_state:
        st.session_state.topic = ""

    # Top Header
    st.markdown('<div class="main-title">RESEARCHER AGENT</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">An autonomous multi-agent system powered by LangChain and LangGraph. It leverages specialized agents to search academic databases (Semantic Scholar/arXiv) and the web, scrape complex PDFs, synthesize comprehensive reports, and critically review the findings.</div>', unsafe_allow_html=True)

    # Two-Column Layout (Matching image_21cfe1.png)
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        # Input Section
        st.markdown('<div class="section-label">RESEARCH TOPIC</div>', unsafe_allow_html=True)
        
        topic_input = st.text_input(
            "topic", 
            key="topic", 
            label_visibility="collapsed", 
            placeholder="e.g. Roadmap for AGI development in next 5 years"
        )
        
        start_pipeline = st.button("⚡ Run Research Pipeline", type="primary", use_container_width=True)

        # Suggestions Section
        st.markdown('<div class="try-label">TRY &rarr;</div>', unsafe_allow_html=True)
        st.button("Future of LLM in Tech Industry", on_click=set_topic, args=("Future of LLM in Tech Industry",))
        st.button("All Latest AI Agents in 2026", on_click=set_topic, args=("All Latest AI Agents in 2026",))
        st.button("Roadmap for AGI development in next 5 years", on_click=set_topic, args=("Roadmap for AGI development in next 5 years",))


    with col_right:
        st.markdown('<div class="pipeline-header">Pipeline</div>', unsafe_allow_html=True)
        
        # We use st.empty() to create placeholders that we can update dynamically
        card_search = st.empty()
        card_reader = st.empty()
        card_writer = st.empty()
        card_critic = st.empty()

        # Initialize cards to WAITING state
        card_search.markdown(render_pipeline_card("01", "Search Agent", "Gathers recent web information & academic papers", "WAITING"), unsafe_allow_html=True)
        card_reader.markdown(render_pipeline_card("02", "Reader Agent", "Scrapes & extracts deep content from PDFs and URLs", "WAITING"), unsafe_allow_html=True)
        card_writer.markdown(render_pipeline_card("03", "Writer Chain", "Drafts the full research report based on scraped data", "WAITING"), unsafe_allow_html=True)
        card_critic.markdown(render_pipeline_card("04", "Critic Chain", "Reviews & scores the report for accuracy and depth", "WAITING"), unsafe_allow_html=True)


    # ==========================================
    # 4. PIPELINE EXECUTION
    # ==========================================
    if start_pipeline and topic_input:
        state = {}
        
        # --- Step 1: Search Agent ---
        card_search.markdown(render_pipeline_card("01", "Search Agent", "Gathers recent web information & academic papers", "RUNNING"), unsafe_allow_html=True)
        
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [
                ("user", f"""Find recent, reliable and detailed information about: {topic_input} 
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
        card_search.markdown(render_pipeline_card("01", "Search Agent", "Gathers recent web information & academic papers", "COMPLETED"), unsafe_allow_html=True)


        # --- Step 2: Reader Agent ---
        card_reader.markdown(render_pipeline_card("02", "Reader Agent", "Scrapes & extracts deep content from PDFs and URLs", "RUNNING"), unsafe_allow_html=True)
        
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                        f"Based on the following search results about '{topic_input}', please gather deeper content.\n\n"
                        f"RULES FOR TOOL USAGE:\n"
                        f"1. For results from the 'web search tool', identify the most relevant URLs and process them using the `scrape_url` tool.\n"
                        f"2. For results from the 'search academic papers' tool, identify the most relevant 'Open Access PDF' URLs and process them using the `read_pdf_from_url` tool.\n"
                        f"3. Do not mix these up. Never use scrape_url on a .pdf link, and never use read_pdf_from_url on a standard webpage.\n\n"
                        f"Search Results:\n{state['search_results']}" 
                    )]
        })
        state['scraped_content'] = reader_result['messages'][-1].content
        card_reader.markdown(render_pipeline_card("02", "Reader Agent", "Scrapes & extracts deep content from PDFs and URLs", "COMPLETED"), unsafe_allow_html=True)


        # --- Step 3: Writer Chain ---
        card_writer.markdown(render_pipeline_card("03", "Writer Chain", "Drafts the full research report based on scraped data", "RUNNING"), unsafe_allow_html=True)
        
        research_combined = f"SEARCH RESULTS : \n {state['search_results']} \n\n DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        state["report"] = writer_chain.invoke({
            "topic": topic_input,
            "research": research_combined
        })
        card_writer.markdown(render_pipeline_card("03", "Writer Chain", "Drafts the full research report based on scraped data", "COMPLETED"), unsafe_allow_html=True)


        # --- Step 4: Critic Chain ---
        card_critic.markdown(render_pipeline_card("04", "Critic Chain", "Reviews & scores the report for accuracy and depth", "RUNNING"), unsafe_allow_html=True)
        
        state["feedback"] = critic_chain.invoke({
            "report": state['report']
        })
        card_critic.markdown(render_pipeline_card("04", "Critic Chain", "Reviews & scores the report for accuracy and depth", "COMPLETED"), unsafe_allow_html=True)


        # ==========================================
        # 5. DISPLAY RESULTS
        # ==========================================
        st.markdown("---")
        st.markdown("## 📊 Research Results")
        
        tab1, tab2, tab3 = st.tabs(["📝 Final Report", "🧠 Critic Review", "⚙️ Raw Agent Data"])
        
        with tab1:
            st.markdown(state["report"])
        with tab2:
            st.info(state["feedback"])
        with tab3:
            with st.expander("Search Results"):
                st.write(state["search_results"])
            with st.expander("Scraped Context"):
                st.write(state["scraped_content"])

if __name__ == "__main__":
    main()
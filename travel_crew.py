"""
Travel Booking — FULL LangChain v1 Implementation
All 6 features with CORRECT imports for LangChain v1 / LangGraph:
  1. LCEL Chains              — prompt | llm | parser  (langchain_core)
  2. Memory                   — InMemoryChatMessageHistory + RunnableWithMessageHistory
  3. Tools / Tool-calling     — DuckDuckGo + Wikipedia (FREE, no key)
  4. RAG / Vector store       — FAISS (langchain_community)
  5. ReAct Agent              — create_react_agent (langgraph.prebuilt)
  6. Streaming                — chain.stream() token by token
"""

import os
import streamlit as st

# LLM + Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# LCEL core primitives
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Memory — correct location in v1
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Tools
# Free tools — no API key needed
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# Vector store
from langchain_community.vectorstores import FAISS

# ReAct Agent — moved to langgraph in LangChain v1
from langgraph.prebuilt import create_react_agent


# ══════════════════════════════════════════════════════════════════
# FEATURE 2 — Memory store (one InMemoryChatMessageHistory per phase)
# ══════════════════════════════════════════════════════════════════

def get_history(phase_key: str) -> InMemoryChatMessageHistory:
    key = f"_hist_{phase_key}"
    if key not in st.session_state:
        st.session_state[key] = InMemoryChatMessageHistory()
    return st.session_state[key]


def clear_all_memory():
    for k in list(st.session_state.keys()):
        if k.startswith("_hist_") or k == "_vector_store":
            del st.session_state[k]


# ══════════════════════════════════════════════════════════════════
# FEATURE 1 — LCEL Chain with memory wrapper
# ══════════════════════════════════════════════════════════════════

def build_chain_with_memory(llm, system_prompt: str, phase_key: str):
    """
    LCEL: prompt | llm | parser
    Wrapped with RunnableWithMessageHistory for persistent memory
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    chain = prompt | llm | StrOutputParser()

    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_history(session_id),
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_memory


# ══════════════════════════════════════════════════════════════════
# FEATURE 3 — Tools (DuckDuckGo + Wikipedia — FREE)
# ══════════════════════════════════════════════════════════════════

def get_free_tools():
    """Returns DuckDuckGo + Wikipedia — both 100% free, no API key required"""
    ddg = DuckDuckGoSearchRun(name="web_search")
    wiki = WikipediaQueryRun(
        name="wikipedia",
        api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=1000)
    )
    return [ddg, wiki]


# ══════════════════════════════════════════════════════════════════
# FEATURE 4 — RAG / FAISS Vector Store
# ══════════════════════════════════════════════════════════════════

def store_in_vector(text: str, metadata: dict, openai_key: str):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    doc = Document(page_content=text, metadata=metadata)
    if "_vector_store" not in st.session_state:
        st.session_state["_vector_store"] = FAISS.from_documents([doc], embeddings)
    else:
        st.session_state["_vector_store"].add_documents([doc])


def retrieve_context(query: str, openai_key: str, k: int = 2) -> str:
    if "_vector_store" not in st.session_state:
        return ""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    results = st.session_state["_vector_store"].similarity_search(query, k=k)
    return "\n\n---\n".join([r.page_content for r in results]) if results else ""


# ══════════════════════════════════════════════════════════════════
# FEATURE 5 — ReAct Agent via langgraph.prebuilt
# ══════════════════════════════════════════════════════════════════

def run_react_research(llm, tools, query: str) -> str:
    """
    LangGraph ReAct agent — autonomously decides what to search,
    executes tools, reasons over results, produces final answer
    """
    system_prompt = """You are a Senior Travel Researcher. Use your search tool to find
CURRENT real information: flight prices, hotel rates, activities, visa requirements, weather.
Make multiple targeted searches. After gathering data, write a comprehensive report with:
## Flights, ## Accommodation (3 tiers), ## Top Activities, ## Local Cuisine,
## Budget Breakdown (USD), ## Weather, ## Travel Tips"""

    agent = create_react_agent(llm, tools, prompt=system_prompt)

    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
    # Extract final text from langgraph message list
    messages = result.get("messages", [])
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if content and isinstance(content, str) and len(content) > 100:
            return content
    return str(result)


# ══════════════════════════════════════════════════════════════════
# FEATURE 6 — Streaming helper
# ══════════════════════════════════════════════════════════════════

def stream_with_memory(chain_with_memory, input_text: str, phase_key: str) -> str:
    """Stream LCEL chain output token by token into Streamlit"""
    full_response = ""
    container = st.empty()
    config = {"configurable": {"session_id": phase_key}}

    for chunk in chain_with_memory.stream(
        {"input": input_text},
        config=config,
    ):
        full_response += chunk
        container.markdown(full_response + "▌")

    container.markdown(full_response)
    return full_response


# ══════════════════════════════════════════════════════════════════
# Agent system prompts
# ══════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are an Expert Itinerary Planner crafting bespoke travel experiences.
For each day: Morning activity, Lunch, Afternoon activity, Evening/Dinner, daily spend, logistics tip.
Format: ## Day 1: [Title], ## Day 2: [Title] etc.
End with Packing List and Top 5 Travel Hacks."""

COORDINATOR_SYSTEM = """You are a meticulous Travel Booking Coordinator.
Produce:
## Booking Checklist — item | platform | when to book | cost | tip
## Budget Summary Table — Category | Estimated Cost | Notes
## Pre-Departure Checklist — documents, apps, arrangements
## Emergency Contacts — numbers, embassy, 6 local phrases"""


# ══════════════════════════════════════════════════════════════════
# PUBLIC PHASE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def run_research_task(api_key, destination, origin,
                      travel_dates, duration, travelers, budget, preferences) -> str:
    """FEATURE 3+5: ReAct Agent with DuckDuckGo + Wikipedia — free, no API key
       FEATURE 4: Result stored in FAISS for later retrieval"""
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

    query = (f"Research a complete {duration}-day trip from {origin} to {destination}. "
             f"Dates: {travel_dates}. Travelers: {travelers}. Budget: {budget}. "
             f"Preferences: {preferences}. "
             f"Search for current flights, hotels (3 price tiers), top activities, "
             f"local cuisine, visa info, and weather. Provide realistic USD prices.")

    # FEATURE 3+5: Always uses free tools — DuckDuckGo + Wikipedia
    tools = get_free_tools()
    research_text = run_react_research(llm, tools, query)

    # FEATURE 4: Store in FAISS
    store_in_vector(research_text, {
        "phase": "research", "destination": destination, "dates": travel_dates
    }, api_key)
    return research_text


def run_planning_task(api_key, research_output, destination,
                      duration, travelers, human_feedback, preferences) -> str:
    """FEATURE 1: LCEL Chain | FEATURE 2: Memory | FEATURE 4: RAG | FEATURE 6: Streaming"""
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

    # FEATURE 4: RAG retrieval
    rag_context = retrieve_context(f"travel {destination} activities hotels", api_key)

    # FEATURE 1+2: LCEL chain with memory
    chain = build_chain_with_memory(llm, PLANNER_SYSTEM, "planner")

    input_text = f"""Create a {duration}-day itinerary for {travelers} traveler(s) to {destination}.
Preferences: {preferences}

Research findings:
{research_output}

Additional retrieved context:
{rag_context if rag_context else 'None stored yet.'}

Human feedback to incorporate:
{human_feedback if human_feedback.strip() else 'None — proceed with research as-is.'}"""

    # FEATURE 6: Streaming
    result = stream_with_memory(chain, input_text, "planner")

    # FEATURE 4: Store itinerary
    store_in_vector(result, {"phase": "itinerary", "destination": destination}, api_key)
    return result


def run_booking_task(api_key, itinerary_output, destination,
                     origin, travel_dates, travelers, budget, human_feedback) -> str:
    """FEATURE 1: LCEL Chain | FEATURE 2: Memory | FEATURE 4: RAG | FEATURE 6: Streaming"""
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

    # FEATURE 4: Retrieve all stored context
    rag_context = retrieve_context(
        f"flights hotels prices booking {destination} {origin}", api_key
    )

    # FEATURE 1+2: LCEL chain with memory
    chain = build_chain_with_memory(llm, COORDINATOR_SYSTEM, "coordinator")

    input_text = f"""Create the complete booking action plan.
Trip: {origin} → {destination} | {travel_dates} | {travelers} traveler(s) | {budget}

Approved itinerary:
{itinerary_output}

Retrieved context (research + prices):
{rag_context if rag_context else 'Use itinerary data above.'}

Final traveler notes:
{human_feedback if human_feedback.strip() else 'Proceed as planned.'}"""

    # FEATURE 6: Streaming
    return stream_with_memory(chain, input_text, "coordinator")

"""
Travel Booking — FULL LangChain Implementation
Uses ALL 6 LangChain features:
  1. LCEL Chains        — pipe prompt | llm | parser
  2. Memory             — ConversationBufferMemory per session
  3. Tools/Tool-calling — Tavily web search for live data
  4. RAG/Vector store   — FAISS stores past trip research
  5. ReAct Agent        — Research agent reasons + searches autonomously
  6. Streaming          — token-by-token output to Streamlit
"""

import os
import streamlit as st

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


# ══════════════════════════════════════════════════════════════════
# FEATURE 1 — LCEL Chain builder
# ══════════════════════════════════════════════════════════════════

def build_lcel_chain(llm, system_prompt: str):
    """Returns a runnable LCEL chain: prompt | llm | parser"""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    return prompt | llm | StrOutputParser()


# ══════════════════════════════════════════════════════════════════
# FEATURE 2 — Memory (one per session, keyed by phase)
# ══════════════════════════════════════════════════════════════════

def get_memory(phase_key: str) -> ConversationBufferMemory:
    """Get or create session memory for a given phase"""
    mem_key = f"_memory_{phase_key}"
    if mem_key not in st.session_state:
        st.session_state[mem_key] = ConversationBufferMemory(
            return_messages=True,
            memory_key="history",
        )
    return st.session_state[mem_key]


def clear_all_memory():
    """Called on new trip — wipe all phase memories"""
    for k in list(st.session_state.keys()):
        if k.startswith("_memory_"):
            del st.session_state[k]
    if "_vector_store" in st.session_state:
        del st.session_state["_vector_store"]


# ══════════════════════════════════════════════════════════════════
# FEATURE 3 — Tools (Tavily web search)
# ══════════════════════════════════════════════════════════════════

def get_search_tool(tavily_key: str):
    os.environ["TAVILY_API_KEY"] = tavily_key
    return TavilySearchResults(max_results=4)


# ══════════════════════════════════════════════════════════════════
# FEATURE 4 — RAG / Vector store (FAISS)
# Store each research output so agents can retrieve context
# ══════════════════════════════════════════════════════════════════

def store_in_vector(text: str, metadata: dict, openai_key: str):
    """Embed and store a research/plan output in FAISS"""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    doc = Document(page_content=text, metadata=metadata)
    if "_vector_store" not in st.session_state:
        st.session_state["_vector_store"] = FAISS.from_documents([doc], embeddings)
    else:
        st.session_state["_vector_store"].add_documents([doc])


def retrieve_context(query: str, openai_key: str, k: int = 2) -> str:
    """Retrieve relevant stored content for a query"""
    if "_vector_store" not in st.session_state:
        return ""
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    results = st.session_state["_vector_store"].similarity_search(query, k=k)
    if not results:
        return ""
    return "\n\n---\n".join([r.page_content for r in results])


# ══════════════════════════════════════════════════════════════════
# FEATURE 5 — ReAct Agent (Research phase only)
# Autonomously decides when to search, what to search, combines results
# ══════════════════════════════════════════════════════════════════

REACT_SYSTEM = """You are a Senior Travel Researcher with 15 years of experience.
You have access to a web search tool. Use it to find CURRENT, REAL information.

Search for: flights, hotels, attractions, food, visa requirements, weather, budget estimates.
Make multiple searches to get complete information. Be specific — search for prices, timings, reviews.

After researching, provide a comprehensive report with these sections:
## Flights — airlines, prices, duration, booking tips
## Accommodation — 3 options (budget/mid/luxury) with real prices
## Top Activities — 8-10 with entry fees and timings  
## Local Cuisine — 5 must-try dishes and restaurants
## Budget Breakdown — realistic total in USD
## Weather & Best Time — during travel dates
## Travel Tips — visa, currency, customs, safety

Always use CURRENT data from your searches. Cite what you found."""

def build_react_agent(llm, tools):
    """Build a ReAct agent that reasons step by step and uses tools"""
    react_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}""")

    agent = create_react_agent(llm, tools, react_prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=8,
    )


# ══════════════════════════════════════════════════════════════════
# FEATURE 6 — Streaming (used in Planner + Coordinator)
# ══════════════════════════════════════════════════════════════════

def stream_chain_response(chain, input_data: dict, memory: ConversationBufferMemory) -> str:
    """Stream LCEL chain output token by token into a Streamlit container"""
    history = memory.load_memory_variables({})["history"]
    full_response = ""

    container = st.empty()
    for chunk in chain.stream({**input_data, "history": history}):
        full_response += chunk
        container.markdown(full_response + "▌")

    container.markdown(full_response)
    memory.save_context(
        {"input": input_data.get("input", "")},
        {"output": full_response}
    )
    return full_response


# ══════════════════════════════════════════════════════════════════
# Planner persona
# ══════════════════════════════════════════════════════════════════

PLANNER_SYSTEM = """You are an Expert Itinerary Planner who crafts bespoke travel experiences.
You balance must-see sights with authentic local experiences.

For each day provide:
- Morning activity (location, duration, cost)
- Lunch recommendation
- Afternoon activity  
- Evening/Dinner
- Daily spend estimate
- Logistics tip

Format as: ## Day 1: [Title], ## Day 2: [Title] etc.
End with a Packing List and Top 5 Travel Hacks."""

COORDINATOR_SYSTEM = """You are a meticulous Travel Booking Coordinator.
Convert approved itineraries into complete booking action plans.

Produce:
## Booking Checklist — what to book, platform, when, cost, tip
## Budget Summary Table — Category | Cost | Notes
## Pre-Departure Checklist — documents, apps, arrangements
## Emergency Contacts — numbers, embassy, 6 local phrases"""


# ══════════════════════════════════════════════════════════════════
# PUBLIC PHASE FUNCTIONS (called from app.py)
# ══════════════════════════════════════════════════════════════════

def run_research_task(api_key, tavily_key, destination, origin,
                      travel_dates, duration, travelers, budget, preferences) -> str:
    """
    FEATURE 3 + 5: ReAct Agent with Tavily tool — autonomously searches live web
    FEATURE 4: Stores result in FAISS vector store for later retrieval
    """
    os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

    query = f"""Research a complete trip:
Origin: {origin} | Destination: {destination} | Dates: {travel_dates}
Duration: {duration} days | Travelers: {travelers} | Budget: {budget}
Preferences: {preferences}

Search for current flights, hotels, activities, cuisine, visa requirements, and weather.
Provide realistic USD prices from current searches."""

    if tavily_key and tavily_key.strip():
        # FEATURE 3+5: Real ReAct agent with web search
        tools = [get_search_tool(tavily_key)]
        agent_executor = build_react_agent(llm, tools)
        result = agent_executor.invoke({"input": query})
        research_text = result.get("output", str(result))
    else:
        # Fallback: LCEL chain without search tools
        chain = build_lcel_chain(llm, REACT_SYSTEM)
        memory = get_memory("research")
        research_text = stream_chain_response(chain, {"input": query}, memory)
        return research_text

    # FEATURE 4: Store in FAISS vector store
    store_in_vector(research_text, {
        "phase": "research",
        "destination": destination,
        "dates": travel_dates,
    }, api_key)

    return research_text


def run_planning_task(api_key, tavily_key, research_output, destination,
                      duration, travelers, human_feedback, preferences) -> str:
    """
    FEATURE 1: LCEL chain (prompt | llm | parser)
    FEATURE 2: Memory — planner remembers conversation context
    FEATURE 4: RAG — retrieves stored research from FAISS
    FEATURE 6: Streaming — token by token output
    """
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, openai_api_key=api_key)

    # FEATURE 4: RAG — retrieve relevant stored context
    rag_context = retrieve_context(
        f"travel itinerary {destination} activities hotels", api_key
    )

    # FEATURE 1: LCEL chain
    chain = build_lcel_chain(llm, PLANNER_SYSTEM)

    # FEATURE 2: Memory
    memory = get_memory("planner")

    input_text = f"""Create a {duration}-day itinerary for {travelers} traveler(s) to {destination}.
Preferences: {preferences}

Research findings:
{research_output}

Additional context from knowledge base:
{rag_context if rag_context else 'No additional context stored yet.'}

Human feedback to incorporate:
{human_feedback if human_feedback.strip() else 'None — proceed with research as-is.'}"""

    # FEATURE 6: Streaming
    result = stream_chain_response(chain, {"input": input_text}, memory)

    # FEATURE 4: Store itinerary in vector store too
    store_in_vector(result, {"phase": "itinerary", "destination": destination}, api_key)

    return result


def run_booking_task(api_key, tavily_key, itinerary_output, destination,
                     origin, travel_dates, travelers, budget, human_feedback) -> str:
    """
    FEATURE 1: LCEL chain
    FEATURE 2: Memory — coordinator recalls full conversation
    FEATURE 3: Optional tool-call for final price checks
    FEATURE 4: RAG — retrieves both research + itinerary
    FEATURE 6: Streaming
    """
    os.environ["OPENAI_API_KEY"] = api_key
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=api_key)

    # FEATURE 4: RAG — pull both research and itinerary context
    rag_context = retrieve_context(
        f"booking flights hotels {destination} {origin} prices", api_key
    )

    # FEATURE 1: LCEL chain
    chain = build_lcel_chain(llm, COORDINATOR_SYSTEM)

    # FEATURE 2: Memory
    memory = get_memory("coordinator")

    input_text = f"""Create the complete booking action plan.

Trip: {origin} → {destination} | {travel_dates} | {travelers} traveler(s) | {budget}

Approved itinerary:
{itinerary_output}

Retrieved context (flights, hotels, prices):
{rag_context if rag_context else 'Use itinerary data above.'}

Final traveler notes:
{human_feedback if human_feedback.strip() else 'Proceed as planned.'}"""

    # FEATURE 6: Streaming
    return stream_chain_response(chain, {"input": input_text}, memory)

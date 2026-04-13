"""
Travel Booking Crew - CrewAI Agents with Human-in-the-Loop
"""

from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool
import os


def get_llm(api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key
    return LLM(
        model="gpt-4o-mini",
        temperature=0.3,
    )


def build_travel_crew(api_key: str, use_search: bool = False):
    llm = get_llm(api_key)
    tools = []

    if use_search and os.getenv("SERPER_API_KEY"):
        tools = [SerperDevTool()]

    # ── Agent 1: Travel Researcher ──────────────────────────────────────────
    researcher = Agent(
        role="Senior Travel Researcher",
        goal="Research the best flights, hotels, and activities for the destination",
        backstory=(
            "You are an expert travel researcher with 15 years of experience "
            "finding the best travel deals and hidden gems worldwide. "
            "You provide realistic price estimates and practical recommendations."
        ),
        llm=llm,
        tools=tools,
        verbose=True,
        allow_delegation=False,
    )

    # ── Agent 2: Itinerary Planner ──────────────────────────────────────────
    planner = Agent(
        role="Expert Itinerary Planner",
        goal="Create a detailed day-by-day travel itinerary based on research",
        backstory=(
            "You craft bespoke travel itineraries that balance must-see sights "
            "with authentic local experiences. You understand pacing, logistics, "
            "and how to make every day memorable without overwhelming travelers."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        allow_delegation=False,
    )

    # ── Agent 3: Booking Coordinator ───────────────────────────────────────
    coordinator = Agent(
        role="Travel Booking Coordinator",
        goal="Create a final booking summary with actionable next steps",
        backstory=(
            "You are a meticulous booking coordinator who turns approved itineraries "
            "into clear, actionable booking checklists. You know which platforms to use, "
            "what to book first, and how to get the best prices."
        ),
        llm=llm,
        tools=[],
        verbose=True,
        allow_delegation=False,
    )

    return researcher, planner, coordinator


def run_research_task(
    api_key: str,
    destination: str,
    origin: str,
    travel_dates: str,
    duration: int,
    travelers: int,
    budget: str,
    preferences: str,
) -> str:
    """Phase 1: Research - runs before human review"""
    researcher, _, _ = build_travel_crew(api_key)

    task = Task(
        description=f"""
        Research a trip with these details:
        - Origin: {origin}
        - Destination: {destination}
        - Travel Dates: {travel_dates}
        - Duration: {duration} days
        - Travelers: {travelers} person(s)
        - Budget: {budget}
        - Preferences/Interests: {preferences}

        Provide:
        1. Flight options (airlines, estimated costs, travel time)
        2. Accommodation options (3 choices at different price points)
        3. Top 8-10 activities and attractions
        4. Local cuisine highlights (5 must-try dishes/restaurants)
        5. Estimated total budget breakdown
        6. Best time to visit and weather during travel dates
        7. Important travel tips (visa, currency, local customs)

        Be specific with realistic price estimates in USD.
        """,
        expected_output="A comprehensive travel research report with flights, hotels, activities, budget breakdown, and travel tips.",
        agent=researcher,
    )

    crew = Crew(agents=[researcher], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def run_planning_task(
    api_key: str,
    research_output: str,
    destination: str,
    duration: int,
    travelers: int,
    human_feedback: str,
    preferences: str,
) -> str:
    """Phase 2: Planning - runs after human approves/modifies research"""
    _, planner, _ = build_travel_crew(api_key)

    task = Task(
        description=f"""
        Based on this research:
        ---
        {research_output}
        ---

        Human feedback and modifications:
        {human_feedback if human_feedback else "No modifications - proceed with research as-is."}

        Create a detailed {duration}-day itinerary for {travelers} traveler(s) to {destination}.
        Preferences: {preferences}

        For each day provide:
        - Morning activity (with location and duration)
        - Lunch recommendation
        - Afternoon activity
        - Evening activity / dinner
        - Estimated daily spend
        - Logistics tip

        Format as: Day 1: [Title], Day 2: [Title], etc.
        End with a packing list and top 5 travel hacks for this destination.
        """,
        expected_output="A complete day-by-day itinerary with morning/afternoon/evening activities, meals, and daily budget.",
        agent=planner,
    )

    crew = Crew(agents=[planner], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)


def run_booking_task(
    api_key: str,
    itinerary_output: str,
    destination: str,
    origin: str,
    travel_dates: str,
    travelers: int,
    budget: str,
    human_feedback: str,
) -> str:
    """Phase 3: Booking Summary - final step after human approves itinerary"""
    _, _, coordinator = build_travel_crew(api_key)

    task = Task(
        description=f"""
        Based on this approved itinerary:
        ---
        {itinerary_output}
        ---

        Final human notes: {human_feedback if human_feedback else "Proceed as planned."}

        Trip details: {origin} → {destination} | {travel_dates} | {travelers} traveler(s) | Budget: {budget}

        Create a complete booking action plan:

        ## BOOKING CHECKLIST (in priority order)
        List each booking item with:
        - What to book
        - Recommended platform/website
        - When to book (how far in advance)
        - Estimated cost
        - Booking tips

        ## BUDGET SUMMARY TABLE
        | Category | Estimated Cost | Notes |
        |----------|---------------|-------|
        List: Flights, Accommodation, Activities, Food, Transport, Misc

        ## PRE-DEPARTURE CHECKLIST
        - Documents needed
        - Apps to download
        - Things to arrange before leaving

        ## EMERGENCY CONTACTS & USEFUL INFO
        - Local emergency numbers
        - Embassy/consulate
        - Useful local phrases (5-6)

        Make this a complete, print-ready travel briefing document.
        """,
        expected_output="A complete booking checklist, budget summary, pre-departure checklist, and emergency contacts.",
        agent=coordinator,
    )

    crew = Crew(agents=[coordinator], tasks=[task], process=Process.sequential, verbose=False)
    result = crew.kickoff()
    return str(result)

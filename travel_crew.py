"""
Travel Booking - LangChain Agent Crew
3 Agents: Researcher → Planner → Booking Coordinator
Human-in-the-Loop between each step
Works on Python 3.14 (Streamlit Cloud compatible)
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


# ── Agent Personas ─────────────────────────────────────────────────────────

RESEARCHER_PERSONA = """You are a Senior Travel Researcher with 15 years of experience.
You provide detailed, realistic travel research with:
1. Flight options (airlines, estimated USD costs, travel time, best booking sites)
2. Hotel recommendations at 3 price points (budget / mid-range / luxury) with estimated nightly rates
3. Top 8-10 activities and attractions with entry fees
4. Local cuisine highlights (5 must-try dishes and recommended restaurants)
5. Full budget breakdown (flights + hotel + food + activities + misc)
6. Weather during the travel dates and best packing advice
7. Essential travel tips: visa requirements, currency, local customs, safety
Be specific with realistic USD estimates. Use clear section headers."""

PLANNER_PERSONA = """You are an Expert Itinerary Planner who crafts bespoke travel experiences.
You balance must-see sights with authentic local experiences.
For each day provide:
- 🌅 Morning: activity (location, duration, cost)
- 🍽️ Lunch: restaurant recommendation with dish to try
- ☀️ Afternoon: activity (location, duration, cost)
- 🌆 Evening/Dinner: activity + restaurant recommendation
- 💰 Daily spend estimate
- 💡 Logistics tip for the day
Format clearly as: ## Day 1: [Catchy Title], ## Day 2: [Title] etc.
End with a 📦 Packing List and 🔑 Top 5 Travel Hacks for this destination."""

COORDINATOR_PERSONA = """You are a meticulous Travel Booking Coordinator.
Convert approved itineraries into complete, actionable booking plans.

Always produce these exact sections:

## ✅ BOOKING CHECKLIST (priority order)
For each item: what to book | platform/website | how far in advance | estimated cost | pro tip

## 💰 BUDGET SUMMARY
| Category | Estimated Cost | Notes |
(include: Flights, Accommodation, Activities, Food & Dining, Local Transport, Misc/Emergency)

## 📋 PRE-DEPARTURE CHECKLIST
- Documents needed
- Apps to download
- Things to arrange before leaving
- 48-hour before departure reminders

## 🆘 EMERGENCY CONTACTS & LOCAL INFO
- Emergency numbers
- Embassy/consulate contact
- 6 useful local phrases with pronunciation
- Nearest hospital / clinic tip

Make this print-ready and comprehensive."""


# ── Core LangChain call ────────────────────────────────────────────────────

def _invoke_agent(api_key: str, persona: str, user_prompt: str) -> str:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.3,
        openai_api_key=api_key,
    )
    messages = [
        SystemMessage(content=persona),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content


# ── Phase 1: Research Agent ────────────────────────────────────────────────

def run_research_task(api_key, destination, origin, travel_dates,
                      duration, travelers, budget, preferences) -> str:
    prompt = f"""Research a complete trip with these details:

🛫 Origin: {origin}
🛬 Destination: {destination}
📅 Dates: {travel_dates}
🗓️ Duration: {duration} days
👥 Travelers: {travelers} person(s)
💰 Budget category: {budget}
🎯 Preferences & Interests: {preferences}

Provide ALL sections: flights, hotels (3 tiers), top activities, cuisine, 
full budget breakdown, weather, and travel tips. Be specific with USD prices."""

    return _invoke_agent(api_key, RESEARCHER_PERSONA, prompt)


# ── Phase 2: Planner Agent ─────────────────────────────────────────────────

def run_planning_task(api_key, research_output, destination, duration,
                      travelers, human_feedback, preferences) -> str:
    prompt = f"""Based on the research below, create a complete itinerary.

=== RESEARCH REPORT ===
{research_output}
=== END RESEARCH ===

✏️ Human feedback / modifications to incorporate:
{human_feedback if human_feedback.strip() else "None — proceed with the research as-is."}

Create a detailed {duration}-day itinerary for {travelers} traveler(s) to {destination}.
Traveler preferences: {preferences}

Cover every day with morning/lunch/afternoon/evening, daily budget, and logistics tip.
End with packing list and top 5 travel hacks."""

    return _invoke_agent(api_key, PLANNER_PERSONA, prompt)


# ── Phase 3: Booking Coordinator Agent ────────────────────────────────────

def run_booking_task(api_key, itinerary_output, destination, origin,
                     travel_dates, travelers, budget, human_feedback) -> str:
    prompt = f"""Convert this approved itinerary into a complete booking action plan.

=== APPROVED ITINERARY ===
{itinerary_output}
=== END ITINERARY ===

✏️ Final notes from traveler:
{human_feedback if human_feedback.strip() else "Proceed as planned."}

Trip summary: {origin} → {destination} | {travel_dates} | {travelers} traveler(s) | {budget}

Produce the complete booking checklist, budget table, pre-departure checklist, 
and emergency contacts. Make it print-ready."""

    return _invoke_agent(api_key, COORDINATOR_PERSONA, prompt)

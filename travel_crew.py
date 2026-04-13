"""
Travel Agent Crew — Pure OpenAI implementation (no CrewAI dependency)
3 Agents: Researcher → Planner → Booking Coordinator
Human-in-the-Loop between each step
"""

from openai import OpenAI


def _call(client: OpenAI, system: str, user: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    )
    return response.choices[0].message.content


# ── Agent system prompts ───────────────────────────────────────────────────

RESEARCHER_SYSTEM = """You are a Senior Travel Researcher with 15 years of experience.
You provide detailed, realistic travel research including flight options with estimated costs,
hotel recommendations at 3 price points, top activities, local cuisine, full budget breakdown,
weather/best time to visit, and essential travel tips (visa, currency, customs).
Be specific with realistic USD price estimates. Structure your output clearly with headers."""

PLANNER_SYSTEM = """You are an Expert Itinerary Planner who crafts bespoke travel experiences.
You balance must-see sights with authentic local experiences, understanding pacing and logistics.
For each day provide: Morning activity (location + duration), Lunch recommendation,
Afternoon activity, Evening/Dinner, estimated daily spend, and a logistics tip.
Format days clearly as 'Day 1: [Title]', 'Day 2: [Title]' etc.
End with a packing list and top 5 travel hacks for the destination."""

COORDINATOR_SYSTEM = """You are a meticulous Travel Booking Coordinator.
You turn approved itineraries into clear, actionable booking plans.
Always produce:
1. BOOKING CHECKLIST — each item with: what to book, recommended platform, when to book, estimated cost, booking tip
2. BUDGET SUMMARY TABLE — markdown table with Category, Estimated Cost, Notes
3. PRE-DEPARTURE CHECKLIST — documents, apps, arrangements
4. EMERGENCY CONTACTS & USEFUL INFO — local emergency numbers, embassy, 5-6 local phrases
Make this a complete, print-ready travel briefing."""


# ── Phase functions ────────────────────────────────────────────────────────

def run_research_task(api_key, destination, origin, travel_dates,
                      duration, travelers, budget, preferences) -> str:
    client = OpenAI(api_key=api_key)
    user_msg = f"""Research this trip:
- Origin: {origin}
- Destination: {destination}
- Dates: {travel_dates}
- Duration: {duration} days
- Travelers: {travelers}
- Budget: {budget}
- Preferences: {preferences}

Provide: flight options, 3 hotel tiers, top 8-10 activities, local cuisine (5 dishes/restaurants),
full budget breakdown, weather during travel dates, travel tips (visa, currency, customs)."""
    return _call(client, RESEARCHER_SYSTEM, user_msg)


def run_planning_task(api_key, research_output, destination, duration,
                      travelers, human_feedback, preferences) -> str:
    client = OpenAI(api_key=api_key)
    user_msg = f"""Based on this research:
---
{research_output}
---
Human feedback/modifications: {human_feedback or 'None — proceed as researched.'}

Create a detailed {duration}-day itinerary for {travelers} traveler(s) to {destination}.
Preferences: {preferences}

For each day: Morning, Lunch, Afternoon, Evening/Dinner, daily spend estimate, logistics tip.
End with packing list and top 5 travel hacks."""
    return _call(client, PLANNER_SYSTEM, user_msg)


def run_booking_task(api_key, itinerary_output, destination, origin,
                     travel_dates, travelers, budget, human_feedback) -> str:
    client = OpenAI(api_key=api_key)
    user_msg = f"""Based on this approved itinerary:
---
{itinerary_output}
---
Final notes: {human_feedback or 'Proceed as planned.'}
Trip: {origin} → {destination} | {travel_dates} | {travelers} traveler(s) | Budget: {budget}

Produce the complete booking action plan with checklist, budget table, pre-departure checklist,
and emergency contacts."""
    return _call(client, COORDINATOR_SYSTEM, user_msg)

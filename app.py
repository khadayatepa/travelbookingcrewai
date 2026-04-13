"""
✈️ AI Travel Booking Assistant
LangChain Agents + Human-in-the-Loop | Powered by GPT-4o-mini
"""

import streamlit as st
from travel_crew import run_research_task, run_planning_task, run_booking_task

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Travel Booking Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white; padding: 2.5rem 2rem; border-radius: 16px;
        margin-bottom: 2rem; text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .main-header h1 { font-family: 'Playfair Display', serif; font-size: 2.8rem; margin: 0; }
    .main-header p { color: #a8d8ea; font-size: 1rem; margin-top: 0.5rem; font-weight: 300; }
    .hitl-box {
        background: #fffbeb; border: 2px solid #f6ad55;
        border-radius: 12px; padding: 1.5rem; margin: 1.5rem 0;
    }
    .hitl-box h3 { color: #c05621; margin-top: 0; }
    .agent-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem;
    }
    .badge-researcher { background: #bee3f8; color: #2b6cb0; }
    .badge-planner    { background: #c6f6d5; color: #276749; }
    .badge-coordinator{ background: #fed7e2; color: #97266d; }
    .output-box {
        background: #1a202c; color: #e2e8f0; padding: 1.5rem;
        border-radius: 10px; font-family: 'Courier New', monospace;
        font-size: 0.85rem; line-height: 1.6; max-height: 500px;
        overflow-y: auto; white-space: pre-wrap; word-break: break-word;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────
for k, v in {
    "phase": 0, "research_output": "", "plan_output": "",
    "booking_output": "", "trip_details": {}
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")

    st.markdown("---")
    st.markdown("### 🤖 Agent Workflow")
    steps = [
        ("🔍", "Research Agent",      "#4a90e2", "Finds flights, hotels, activities"),
        ("👤", "YOU Review",           "#f6ad55", "Approve or modify research"),
        ("🗺️", "Planner Agent",       "#4a90e2", "Builds day-by-day itinerary"),
        ("👤", "YOU Review",           "#f6ad55", "Approve or modify itinerary"),
        ("📋", "Coordinator Agent",    "#4a90e2", "Final booking checklist"),
        ("✅", "Done!",                "#38a169", "Download your travel plan"),
    ]
    for icon, name, color, desc in steps:
        bg = "#fffbeb" if "YOU" in name else "#f0f7ff"
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin:5px 0;padding:6px 8px;
             background:{bg};border-radius:8px;border-left:3px solid {color}">
            <span style="font-size:1.1rem;margin-right:8px">{icon}</span>
            <div>
                <div style="font-weight:600;font-size:0.82rem;color:{color}">{name}</div>
                <div style="font-size:0.73rem;color:#718096">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Stack")
    st.markdown("- 🦜 **LangChain** — Agent orchestration\n- 🧠 **GPT-4o-mini** — LLM\n- 🎈 **Streamlit** — UI\n- 👤 **Human-in-the-Loop** — You decide")

    if st.session_state.phase > 0:
        st.markdown("---")
        if st.button("🔄 Start New Trip", use_container_width=True):
            for k in ["phase","research_output","plan_output","booking_output","trip_details"]:
                st.session_state[k] = 0 if k == "phase" else ({} if k == "trip_details" else "")
            st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ AI Travel Booking Assistant</h1>
    <p>LangChain Agents &times; Human-in-the-Loop &times; GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 0 — Input Form
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 0:
    st.markdown("## 📝 Tell us about your trip")
    col1, col2 = st.columns(2)
    with col1:
        origin       = st.text_input("🛫 Flying from", value="Mumbai, India")
        travel_dates = st.text_input("📅 Travel dates", value="July 15-22, 2025")
        travelers    = st.number_input("👥 Travelers", min_value=1, max_value=10, value=2)
    with col2:
        destination = st.text_input("🛬 Destination", value="Bali, Indonesia")
        duration    = st.number_input("🗓️ Duration (days)", min_value=1, max_value=30, value=7)
        budget      = st.selectbox("💰 Budget", [
            "Budget (~$500-1000 per person)",
            "Mid-range (~$1000-2500 per person)",
            "Comfort (~$2500-5000 per person)",
            "Luxury ($5000+ per person)"
        ])
    preferences = st.text_area(
        "🎯 Travel style & interests",
        value="Beach relaxation, local cuisine, cultural experiences, some adventure activities",
        height=90,
    )
    st.markdown("---")
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if not api_key:
            st.warning("⚠️ Enter your OpenAI API key in the sidebar.")
        else:
            if st.button("🚀 Start AI Research", use_container_width=True, type="primary"):
                if not destination or not origin:
                    st.error("Please fill in origin and destination!")
                else:
                    st.session_state.trip_details = dict(
                        origin=origin, destination=destination,
                        travel_dates=travel_dates, duration=duration,
                        travelers=travelers, budget=budget, preferences=preferences
                    )
                    with st.spinner("🔍 Research Agent working... (30-60 sec)"):
                        try:
                            st.session_state.research_output = run_research_task(
                                api_key=api_key, **st.session_state.trip_details)
                            st.session_state.phase = 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ {e}")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Human Reviews Research
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 1:
    td = st.session_state.trip_details
    st.markdown(f"## 🔍 Research Complete: {td['origin']} → {td['destination']}")
    st.markdown("""<div class="hitl-box">
        <h3>👤 Human-in-the-Loop — Review Research</h3>
        <p>The <strong>Research Agent</strong> has finished. Review below, add any feedback,
        then approve to generate your itinerary.</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent Output</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### ✏️ Your Feedback")
        fb1 = st.text_area("Corrections or preferences:", height=260,
            placeholder="- Skip Hilton, prefer boutique\n- No water sports\n- Add cooking class\n- Budget is $3000 total",
            key="fb1")
        st.markdown("---")
        if st.button("✅ Approve & Generate Itinerary", use_container_width=True, type="primary"):
            with st.spinner("🗺️ Planner Agent building itinerary..."):
                try:
                    st.session_state.plan_output = run_planning_task(
                        api_key=api_key,
                        research_output=st.session_state.research_output,
                        destination=td["destination"], duration=td["duration"],
                        travelers=td["travelers"], human_feedback=fb1,
                        preferences=td["preferences"])
                    st.session_state.phase = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        if st.button("🔄 Re-run Research", use_container_width=True):
            st.session_state.phase = 0; st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Human Reviews Itinerary
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    td = st.session_state.trip_details
    st.markdown(f"## 🗺️ Itinerary Ready: {td['duration']}-Day {td['destination']} Trip")
    st.markdown("""<div class="hitl-box">
        <h3>👤 Human-in-the-Loop — Review Itinerary</h3>
        <p>The <strong>Planner Agent</strong> has crafted your itinerary. Add final adjustments
        before the Booking Coordinator prepares your action plan.</p></div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        t1, t2 = st.tabs(["🗺️ Itinerary", "🔍 Research (Reference)"])
        with t1:
            st.markdown('<span class="agent-badge badge-planner">🗺️ Planner Agent Output</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)
        with t2:
            st.markdown('<span class="agent-badge badge-researcher">🔍 Research Reference</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("### ✏️ Final Adjustments")
        fb2 = st.text_area("Any changes before booking?", height=260,
            placeholder="- Swap Day 3 and Day 5\n- Add airport transfer\n- Book business class\n- Add gluten-free options",
            key="fb2")
        st.markdown("---")
        if st.button("✅ Approve & Get Booking Plan", use_container_width=True, type="primary"):
            with st.spinner("📋 Coordinator preparing booking plan..."):
                try:
                    st.session_state.booking_output = run_booking_task(
                        api_key=api_key,
                        itinerary_output=st.session_state.plan_output,
                        destination=td["destination"], origin=td["origin"],
                        travel_dates=td["travel_dates"], travelers=td["travelers"],
                        budget=td["budget"], human_feedback=fb2)
                    st.session_state.phase = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ {e}")
        if st.button("⬅️ Back to Research", use_container_width=True):
            st.session_state.phase = 1; st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Final Booking Plan
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 3:
    td = st.session_state.trip_details
    st.markdown("""<div style="background:linear-gradient(135deg,#276749,#38a169);color:white;
        padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;text-align:center">
        <h2 style="margin:0">🎉 Your AI Travel Plan is Ready!</h2>
        <p style="margin:0.3rem 0 0;opacity:0.9">All 3 agents completed. Review and download below.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, (icon, label, val) in zip([c1,c2,c3,c4], [
        ("🛫","Trip",f"{td['origin']} → {td['destination']}"),
        ("📅","Dates",td['travel_dates']),
        ("👥","Travelers",f"{td['travelers']} person(s)"),
        ("💰","Budget",td['budget'].split("(")[0].strip()),
    ]):
        with col:
            st.markdown(f"""<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                padding:1rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.06)">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-size:0.75rem;color:#718096;font-weight:500">{label}</div>
                <div style="font-size:0.85rem;font-weight:700;color:#2d3748">{val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["📋 Booking Plan", "🗺️ Itinerary", "🔍 Research"])
    with t1:
        st.markdown('<span class="agent-badge badge-coordinator">📋 Booking Coordinator</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.booking_output}</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<span class="agent-badge badge-planner">🗺️ Planner Agent</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)
    with t3:
        st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)

    st.markdown("---")
    full_report = f"""✈️ AI TRAVEL PLAN: {td['origin']} → {td['destination']}
{'='*60}
Dates: {td['travel_dates']} | Duration: {td['duration']} days
Travelers: {td['travelers']} | Budget: {td['budget']}
Preferences: {td['preferences']}

{'='*60}
RESEARCH REPORT
{'='*60}
{st.session_state.research_output}

{'='*60}
DAY-BY-DAY ITINERARY
{'='*60}
{st.session_state.plan_output}

{'='*60}
BOOKING ACTION PLAN
{'='*60}
{st.session_state.booking_output}

Generated by AI Travel Assistant (LangChain + GPT-4o-mini)
"""
    _, col_dl, _ = st.columns([1, 2, 1])
    with col_dl:
        st.download_button(
            "📥 Download Complete Travel Plan (.txt)", data=full_report,
            file_name=f"travel_{td['destination'].replace(', ','_').replace(' ','_')}.txt",
            mime="text/plain", use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Plan Another Trip", use_container_width=True):
            for k in ["phase","research_output","plan_output","booking_output","trip_details"]:
                st.session_state[k] = 0 if k == "phase" else ({} if k == "trip_details" else "")
            st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#a0aec0;font-size:0.8rem;padding:0.5rem">
    Built with ❤️ using <strong>LangChain</strong> + <strong>Streamlit</strong> + <strong>GPT-4o-mini</strong>
    &nbsp;|&nbsp; Human-in-the-Loop at every step
</div>""", unsafe_allow_html=True)

"""
✈️ AI Travel Booking Assistant
CrewAI + Human-in-the-Loop | Powered by GPT-4o-mini
"""

import streamlit as st
import time
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

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: white;
        padding: 2.5rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.8rem;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: #a8d8ea;
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }

    .phase-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .phase-active {
        border-left: 4px solid #2c5364;
        background: #f0f7ff;
    }

    .phase-done {
        border-left: 4px solid #38a169;
        background: #f0fff4;
    }

    .phase-pending {
        border-left: 4px solid #cbd5e0;
        background: #fafafa;
        opacity: 0.7;
    }

    .hitl-box {
        background: #fffbeb;
        border: 2px solid #f6ad55;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }

    .hitl-box h3 {
        color: #c05621;
        margin-top: 0;
    }

    .agent-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    .badge-researcher { background: #bee3f8; color: #2b6cb0; }
    .badge-planner    { background: #c6f6d5; color: #276749; }
    .badge-coordinator { background: #fed7e2; color: #97266d; }

    .stTextArea textarea {
        font-size: 0.9rem;
        border-radius: 8px;
    }

    .output-box {
        background: #1a202c;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 10px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.6;
        max-height: 500px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    .progress-steps {
        display: flex;
        justify-content: space-between;
        margin: 1.5rem 0;
        padding: 0;
    }

    .step-dot {
        text-align: center;
        flex: 1;
    }

    .tip-box {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }

    div[data-testid="stButton"] button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase": 0,                  # 0=input, 1=research done, 2=plan done, 3=complete
        "research_output": "",
        "plan_output": "",
        "booking_output": "",
        "trip_details": {},
        "api_key_valid": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Your OpenAI API key (stored only in session)"
    )

    st.markdown("---")
    st.markdown("### 🤖 Agent Workflow")

    phases = [
        ("🔍", "Research Agent", "Finds flights, hotels, activities"),
        ("👤", "YOU Review", "Approve or modify research"),
        ("🗺️", "Planner Agent", "Builds day-by-day itinerary"),
        ("👤", "YOU Review", "Approve or modify itinerary"),
        ("📋", "Coordinator Agent", "Final booking checklist"),
        ("✅", "Done!", "Download your travel plan"),
    ]

    for i, (icon, name, desc) in enumerate(phases):
        is_human = "YOU" in name
        color = "#f6ad55" if is_human else "#4a90e2"
        done_marker = "✓ " if st.session_state.phase > (i // 1) else ""
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin:6px 0;padding:6px 8px;
             background:{'#fffbeb' if is_human else '#f0f7ff'};border-radius:8px;
             border-left:3px solid {color}">
            <span style="font-size:1.2rem;margin-right:8px">{icon}</span>
            <div>
                <div style="font-weight:600;font-size:0.85rem;color:{'#c05621' if is_human else '#2c5364'}">{done_marker}{name}</div>
                <div style="font-size:0.75rem;color:#718096">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📦 Stack")
    st.markdown("""
    - 🐍 **CrewAI** - Agent orchestration
    - 🧠 **GPT-4o-mini** - LLM backbone
    - 🎈 **Streamlit** - UI
    - 👤 **Human-in-the-Loop** - You stay in control
    """)

    if st.session_state.phase > 0:
        st.markdown("---")
        if st.button("🔄 Start New Trip", use_container_width=True):
            for key in ["phase", "research_output", "plan_output", "booking_output", "trip_details"]:
                st.session_state[key] = 0 if key == "phase" else ({} if key == "trip_details" else "")
            st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ AI Travel Booking Assistant</h1>
    <p>CrewAI Agents × Human-in-the-Loop × GPT-4o-mini</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# PHASE 0 — Trip Input Form
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 0:
    st.markdown("## 📝 Tell us about your trip")

    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("🛫 Flying from", placeholder="Mumbai, India", value="Mumbai, India")
        travel_dates = st.text_input("📅 Travel dates", placeholder="July 15-22, 2025", value="July 15-22, 2025")
        travelers = st.number_input("👥 Number of travelers", min_value=1, max_value=10, value=2)

    with col2:
        destination = st.text_input("🛬 Destination", placeholder="Bali, Indonesia", value="Bali, Indonesia")
        duration = st.number_input("🗓️ Duration (days)", min_value=1, max_value=30, value=7)
        budget = st.selectbox("💰 Total budget", [
            "Budget (~$500-1000 per person)",
            "Mid-range (~$1000-2500 per person)",
            "Comfort (~$2500-5000 per person)",
            "Luxury ($5000+ per person)"
        ])

    preferences = st.text_area(
        "🎯 Travel style & interests",
        placeholder="e.g., Adventure sports, vegetarian food, beach relaxation, cultural temples, photography spots, avoid touristy areas...",
        height=100,
        value="Beach relaxation, local cuisine, cultural experiences, some adventure activities"
    )

    st.markdown("---")

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if not api_key:
            st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        else:
            if st.button("🚀 Start AI Research", use_container_width=True, type="primary"):
                if not destination or not origin:
                    st.error("Please fill in origin and destination!")
                else:
                    st.session_state.trip_details = {
                        "origin": origin,
                        "destination": destination,
                        "travel_dates": travel_dates,
                        "duration": duration,
                        "travelers": travelers,
                        "budget": budget,
                        "preferences": preferences,
                    }

                    with st.spinner("🔍 Research Agent is working... (30-60 seconds)"):
                        try:
                            result = run_research_task(
                                api_key=api_key,
                                **st.session_state.trip_details
                            )
                            st.session_state.research_output = result
                            st.session_state.phase = 1
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
                            if "api" in str(e).lower() or "key" in str(e).lower():
                                st.info("💡 Check your OpenAI API key and ensure you have GPT-4o-mini access.")


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Human Reviews Research
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 1:
    td = st.session_state.trip_details
    st.markdown(f"## 🔍 Research Complete: {td['origin']} → {td['destination']}")

    st.markdown("""
    <div class="hitl-box">
        <h3>👤 Human-in-the-Loop — Review Research</h3>
        <p>The <strong>Research Agent</strong> has completed its work. Review the findings below.
        You can add feedback, corrections, or preferences before the Planner Agent builds your itinerary.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent Output</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ✏️ Your Feedback")
        st.markdown("*Modify the AI's plan before proceeding:*")
        feedback_1 = st.text_area(
            "Add instructions or corrections:",
            height=250,
            placeholder=(
                "Examples:\n"
                "- Skip the Hilton, prefer boutique hotels\n"
                "- No water sports, focus on culture\n"
                "- Must include a cooking class\n"
                "- Budget is actually $3000 total\n"
                "- Add a day trip to Ubud"
            ),
            key="feedback_research"
        )

        st.markdown("---")
        if st.button("✅ Approve & Generate Itinerary", use_container_width=True, type="primary"):
            with st.spinner("🗺️ Planner Agent is building your itinerary..."):
                try:
                    result = run_planning_task(
                        api_key=api_key,
                        research_output=st.session_state.research_output,
                        destination=td["destination"],
                        duration=td["duration"],
                        travelers=td["travelers"],
                        human_feedback=feedback_1,
                        preferences=td["preferences"],
                    )
                    st.session_state.plan_output = result
                    st.session_state.phase = 2
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        if st.button("🔄 Re-run Research", use_container_width=True):
            st.session_state.phase = 0
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Human Reviews Itinerary
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    td = st.session_state.trip_details
    st.markdown(f"## 🗺️ Itinerary Ready: {td['duration']}-Day {td['destination']} Trip")

    st.markdown("""
    <div class="hitl-box">
        <h3>👤 Human-in-the-Loop — Review Itinerary</h3>
        <p>The <strong>Planner Agent</strong> has crafted your itinerary. Review it carefully.
        Add any final adjustments before the Booking Coordinator prepares your action plan.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        tab1, tab2 = st.tabs(["🗺️ Itinerary", "🔍 Research (Reference)"])
        with tab1:
            st.markdown('<span class="agent-badge badge-planner">🗺️ Planner Agent Output</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)
        with tab2:
            st.markdown('<span class="agent-badge badge-researcher">🔍 Research Reference</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown("### ✏️ Final Adjustments")
        feedback_2 = st.text_area(
            "Any changes before booking?",
            height=250,
            placeholder=(
                "Examples:\n"
                "- Swap Day 3 and Day 5\n"
                "- Add airport transfer notes\n"
                "- Include travel insurance tips\n"
                "- Book business class flights\n"
                "- Add gluten-free dining options"
            ),
            key="feedback_itinerary"
        )

        st.markdown("---")
        if st.button("✅ Approve & Get Booking Plan", use_container_width=True, type="primary"):
            with st.spinner("📋 Booking Coordinator is preparing your action plan..."):
                try:
                    result = run_booking_task(
                        api_key=api_key,
                        itinerary_output=st.session_state.plan_output,
                        destination=td["destination"],
                        origin=td["origin"],
                        travel_dates=td["travel_dates"],
                        travelers=td["travelers"],
                        budget=td["budget"],
                        human_feedback=feedback_2,
                    )
                    st.session_state.booking_output = result
                    st.session_state.phase = 3
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        if st.button("⬅️ Back to Research", use_container_width=True):
            st.session_state.phase = 1
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Final Booking Plan
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 3:
    td = st.session_state.trip_details

    st.markdown("""
    <div style="background:linear-gradient(135deg,#276749,#38a169);color:white;
         padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;text-align:center">
        <h2 style="margin:0">🎉 Your AI Travel Plan is Ready!</h2>
        <p style="margin:0.3rem 0 0;opacity:0.9">All 3 agents have completed their work. Review, copy or download below.</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary strip
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("🛫", "Trip", f"{td['origin']} → {td['destination']}"),
        ("📅", "Dates", td['travel_dates']),
        ("👥", "Travelers", f"{td['travelers']} person(s)"),
        ("💰", "Budget", td['budget'].split("(")[0].strip()),
    ]
    for col, (icon, label, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                 padding:1rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,0.06)">
                <div style="font-size:1.5rem">{icon}</div>
                <div style="font-size:0.75rem;color:#718096;font-weight:500">{label}</div>
                <div style="font-size:0.85rem;font-weight:700;color:#2d3748">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # All outputs in tabs
    tab1, tab2, tab3 = st.tabs(["📋 Booking Plan", "🗺️ Itinerary", "🔍 Research"])

    with tab1:
        st.markdown('<span class="agent-badge badge-coordinator">📋 Booking Coordinator Output</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.booking_output}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<span class="agent-badge badge-planner">🗺️ Planner Agent Output</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent Output</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)

    # Download
    st.markdown("---")
    full_report = f"""
# ✈️ AI Travel Plan: {td['origin']} → {td['destination']}
## Trip Details
- Dates: {td['travel_dates']}
- Duration: {td['duration']} days
- Travelers: {td['travelers']}
- Budget: {td['budget']}
- Preferences: {td['preferences']}

---
# 🔍 RESEARCH REPORT
{st.session_state.research_output}

---
# 🗺️ DAY-BY-DAY ITINERARY
{st.session_state.plan_output}

---
# 📋 BOOKING ACTION PLAN
{st.session_state.booking_output}

---
Generated by AI Travel Assistant (CrewAI + GPT-4o-mini)
"""

    col_dl1, col_dl2, col_dl3 = st.columns([1, 2, 1])
    with col_dl2:
        st.download_button(
            label="📥 Download Complete Travel Plan (.txt)",
            data=full_report,
            file_name=f"travel_plan_{td['destination'].replace(', ','_').replace(' ','_')}.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Plan Another Trip", use_container_width=True):
            for key in ["phase", "research_output", "plan_output", "booking_output", "trip_details"]:
                st.session_state[key] = 0 if key == "phase" else ({} if key == "trip_details" else "")
            st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#a0aec0;font-size:0.8rem;padding:0.5rem">
    Built with ❤️ using <strong>CrewAI</strong> + <strong>Streamlit</strong> + <strong>GPT-4o-mini</strong> &nbsp;|&nbsp;
    Human-in-the-Loop at every step
</div>
""", unsafe_allow_html=True)

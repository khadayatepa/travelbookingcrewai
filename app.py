"""
✈️ AI Travel Booking Assistant — Full LangChain Edition
All 6 LangChain features active:
  LCEL Chains | Memory | Tools | RAG | ReAct Agent | Streaming
"""

import streamlit as st
from travel_crew import (
    run_research_task, run_planning_task, run_booking_task, clear_all_memory
)

st.set_page_config(
    page_title="AI Travel Booking Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-header{background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);color:white;
  padding:2.5rem 2rem;border-radius:16px;margin-bottom:2rem;text-align:center;
  box-shadow:0 8px 32px rgba(0,0,0,0.3);}
.main-header h1{font-family:'Playfair Display',serif;font-size:2.6rem;margin:0;}
.main-header p{color:#a8d8ea;font-size:.95rem;margin-top:.5rem;font-weight:300;}
.hitl-box{background:#fffbeb;border:2px solid #f6ad55;border-radius:12px;padding:1.5rem;margin:1.5rem 0;}
.hitl-box h3{color:#c05621;margin-top:0;}
.feature-pill{display:inline-block;padding:2px 10px;border-radius:20px;
  font-size:.72rem;font-weight:600;margin:2px;background:#e9d8fd;color:#553c9a;}
.agent-badge{display:inline-block;padding:3px 10px;border-radius:20px;
  font-size:.75rem;font-weight:600;margin-bottom:.5rem;}
.badge-researcher{background:#bee3f8;color:#2b6cb0;}
.badge-planner{background:#c6f6d5;color:#276749;}
.badge-coordinator{background:#fed7e2;color:#97266d;}
.output-box{background:#1a202c;color:#e2e8f0;padding:1.5rem;border-radius:10px;
  font-family:'Courier New',monospace;font-size:.85rem;line-height:1.6;
  max-height:500px;overflow-y:auto;white-space:pre-wrap;word-break:break-word;}
.feature-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;}
.feat-item{background:#f0fff4;border:1px solid #9ae6b4;border-radius:6px;
  padding:4px 8px;font-size:.72rem;color:#276749;}
</style>
""", unsafe_allow_html=True)


# ── Session State ──────────────────────────────────────────────────────────
for k, v in {"phase":0,"research_output":"","plan_output":"",
              "booking_output":"","trip_details":{}}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Keys")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    st.success("✅ Live search: DuckDuckGo + Wikipedia (free, no key needed)")

    st.markdown("---")
    st.markdown("### 🦜 LangChain Features Active")
    features = [
        ("🔗", "LCEL Chains", "Planner + Coordinator"),
        ("🧠", "Memory", "Each agent remembers context"),
        ("🔍", "DuckDuckGo + Wiki", "Free search, no key needed"),
        ("📚", "RAG / FAISS", "Vector retrieval between agents"),
        ("🤖", "ReAct Agent", "Research reasons + acts"),
        ("⚡", "Streaming", "Token-by-token output"),
    ]
    for icon, name, desc in features:
        active = True
        if False:  # all tools always active
            pass
        color = "#f0fff4" if active else "#f7fafc"
        border = "#9ae6b4" if active else "#e2e8f0"
        badge = "✅" if active else "⬜"
        st.markdown(f"""
        <div style="background:{color};border:1px solid {border};border-radius:8px;
             padding:6px 10px;margin:4px 0;font-size:.8rem;">
            <span style="font-size:.9rem">{icon}</span>
            <strong style="color:#2d3748"> {name}</strong> {badge}<br>
            <span style="color:#718096;font-size:.72rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🤖 Agent Workflow")
    steps = [
        ("🔍","Research Agent (ReAct)","#4a90e2","Tool use + autonomous search"),
        ("👤","YOU Review","#f6ad55","Approve or edit research"),
        ("🗺️","Planner Agent (LCEL)","#4a90e2","LCEL chain + memory + RAG"),
        ("👤","YOU Review","#f6ad55","Approve or edit itinerary"),
        ("📋","Coordinator (LCEL)","#4a90e2","LCEL + memory + RAG"),
        ("✅","Done!","#38a169","Download plan"),
    ]
    for icon, name, color, desc in steps:
        bg = "#fffbeb" if "YOU" in name else "#f0f7ff"
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;margin:4px 0;padding:5px 8px;
             background:{bg};border-radius:8px;border-left:3px solid {color}">
            <span style="font-size:1rem;margin-right:7px">{icon}</span>
            <div>
                <div style="font-weight:600;font-size:.8rem;color:{color}">{name}</div>
                <div style="font-size:.7rem;color:#718096">{desc}</div>
            </div>
        </div>""", unsafe_allow_html=True)

    if st.session_state.phase > 0:
        st.markdown("---")
        if st.button("🔄 New Trip (clears memory)", use_container_width=True):
            clear_all_memory()
            for k in ["phase","research_output","plan_output","booking_output","trip_details"]:
                st.session_state[k] = 0 if k=="phase" else ({} if k=="trip_details" else "")
            st.rerun()


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>✈️ AI Travel Booking Assistant</h1>
    <p>Full LangChain &nbsp;|&nbsp; LCEL &bull; Memory &bull; Tools &bull; RAG &bull; ReAct &bull; Streaming &nbsp;&times;&nbsp; Human-in-the-Loop</p>
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

    st.info("🔍 **ReAct Agent** — uses DuckDuckGo + Wikipedia to search for real travel info. No API key needed.")

    st.markdown("---")
    _, col_btn, _ = st.columns([1, 2, 1])
    with col_btn:
        if not api_key:
            st.warning("⚠️ Enter your OpenAI API key in the sidebar.")
        else:
            if st.button("🚀 Launch AI Research", use_container_width=True, type="primary"):
                if not destination or not origin:
                    st.error("Please fill in origin and destination!")
                else:
                    st.session_state.trip_details = dict(
                        origin=origin, destination=destination,
                        travel_dates=travel_dates, duration=int(duration),
                        travelers=int(travelers), budget=budget, preferences=preferences
                    )
                    st.session_state.phase = 1
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 1 — Research Agent Running + Review
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 1:
    td = st.session_state.trip_details

    if not st.session_state.research_output:
        st.markdown(f"## 🔍 Research Agent: {td['origin']} → {td['destination']}")

        st.markdown("""
            <div style="background:#ebf8ff;border:1px solid #90cdf4;border-radius:8px;padding:1rem;margin-bottom:1rem">
                <strong style="color:#2b6cb0">🤖 ReAct Agent running</strong><br>
                <span style="font-size:.85rem;color:#2c5282">
                Searching DuckDuckGo + Wikipedia → reasoning → combining results. May take 30-60 seconds.</span>
            </div>""", unsafe_allow_html=True)

        with st.spinner("🔍 Research Agent working..."):
            try:
                result = run_research_task(
                    api_key=api_key,
                    **td
                )
                st.session_state.research_output = result
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.session_state.phase = 0
    else:
        st.markdown(f"## 🔍 Research Complete: {td['origin']} → {td['destination']}")
        st.markdown("""<div class="hitl-box">
            <h3>👤 Human-in-the-Loop — Review Research</h3>
            <p>The <strong>ReAct Research Agent</strong> has completed autonomous web research.
            Review findings, add corrections, then approve to generate your itinerary.</p>
            <div>
            <span class="feature-pill">🔍 ReAct Agent</span>
            <span class="feature-pill">🌐 Live Search</span>
            <span class="feature-pill">📚 Stored in FAISS</span>
            </div></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent — ReAct + DuckDuckGo + Wikipedia</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("### ✏️ Your Feedback")
            fb1 = st.text_area("Corrections or preferences:", height=220,
                placeholder="- Skip Hilton, prefer boutique\n- No water sports\n- Add cooking class\n- Budget is $3000 total\n- Focus on north Bali",
                key="fb1")
            st.markdown("---")
            if st.button("✅ Approve & Build Itinerary", use_container_width=True, type="primary"):
                st.session_state.phase = 2
                st.rerun()
            if st.button("🔄 Re-run Research", use_container_width=True):
                st.session_state.research_output = ""
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 2 — Planner Agent (LCEL + Memory + RAG + Streaming)
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    td = st.session_state.trip_details

    if not st.session_state.plan_output:
        st.markdown(f"## 🗺️ Planner Agent: Building your {td['duration']}-day itinerary")
        st.markdown("""
        <div style="background:#f0fff4;border:1px solid #9ae6b4;border-radius:8px;padding:1rem;margin-bottom:1rem">
            <strong style="color:#276749">🔗 LCEL Chain + Memory + RAG + Streaming</strong><br>
            <span style="font-size:.85rem;color:#2f855a">
            Retrieving research from vector store → building itinerary with conversation memory → streaming output token by token.</span>
        </div>""", unsafe_allow_html=True)

        fb1 = st.session_state.get("fb1", "")
        st.markdown("**Streaming output:**")
        try:
            result = run_planning_task(
                api_key=api_key,
                research_output=st.session_state.research_output,
                destination=td["destination"],
                duration=td["duration"],
                travelers=td["travelers"],
                human_feedback=fb1,
                preferences=td["preferences"],
            )
            st.session_state.plan_output = result
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.phase = 1
    else:
        st.markdown(f"## 🗺️ Itinerary Ready: {td['duration']}-Day {td['destination']} Trip")
        st.markdown("""<div class="hitl-box">
            <h3>👤 Human-in-the-Loop — Review Itinerary</h3>
            <p>The <strong>Planner Agent</strong> used LCEL chains, conversation memory, and RAG retrieval.
            Review the itinerary and add any final adjustments.</p>
            <div>
            <span class="feature-pill">🔗 LCEL Chain</span>
            <span class="feature-pill">🧠 Memory Active</span>
            <span class="feature-pill">📚 RAG Retrieved</span>
            <span class="feature-pill">⚡ Streamed</span>
            </div></div>""", unsafe_allow_html=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            t1, t2 = st.tabs(["🗺️ Itinerary", "🔍 Research"])
            with t1:
                st.markdown('<span class="agent-badge badge-planner">🗺️ Planner — LCEL + Memory + RAG</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)
            with t2:
                st.markdown('<span class="agent-badge badge-researcher">🔍 Research Reference</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)
        with col2:
            st.markdown("### ✏️ Final Adjustments")
            fb2 = st.text_area("Changes before booking?", height=220,
                placeholder="- Swap Day 3 and 5\n- Add airport transfer\n- Book business class\n- Gluten-free options",
                key="fb2")
            st.markdown("---")
            if st.button("✅ Approve & Get Booking Plan", use_container_width=True, type="primary"):
                st.session_state.phase = 3
                st.rerun()
            if st.button("⬅️ Back to Research", use_container_width=True):
                st.session_state.phase = 1
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# PHASE 3 — Coordinator Agent (LCEL + Memory + RAG + Streaming)
# ══════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 3:
    td = st.session_state.trip_details

    if not st.session_state.booking_output:
        st.markdown("## 📋 Coordinator Agent: Building your booking plan")
        st.markdown("""
        <div style="background:#fff5f7;border:1px solid #feb2c1;border-radius:8px;padding:1rem;margin-bottom:1rem">
            <strong style="color:#97266d">🔗 LCEL Chain + Memory + RAG + Streaming</strong><br>
            <span style="font-size:.85rem;color:#702459">
            Retrieving all stored context (research + itinerary) from FAISS → using full conversation memory → generating booking checklist.</span>
        </div>""", unsafe_allow_html=True)

        fb2 = st.session_state.get("fb2", "")
        st.markdown("**Streaming output:**")
        try:
            result = run_booking_task(
                api_key=api_key,
                itinerary_output=st.session_state.plan_output,
                destination=td["destination"],
                origin=td["origin"],
                travel_dates=td["travel_dates"],
                travelers=td["travelers"],
                budget=td["budget"],
                human_feedback=fb2,
            )
            st.session_state.booking_output = result
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.session_state.phase = 2
    else:
        # ── Final output ───────────────────────────────────────────
        st.markdown("""<div style="background:linear-gradient(135deg,#276749,#38a169);color:white;
            padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;text-align:center">
            <h2 style="margin:0">🎉 Your AI Travel Plan is Ready!</h2>
            <p style="margin:.3rem 0 0;opacity:.9">All 6 LangChain features used. Review and download below.</p>
        </div>""", unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        for col, (icon, label, val) in zip([c1,c2,c3,c4],[
            ("🛫","Trip",f"{td['origin']} → {td['destination']}"),
            ("📅","Dates",td['travel_dates']),
            ("👥","Travelers",f"{td['travelers']} person(s)"),
            ("💰","Budget",td['budget'].split("(")[0].strip()),
        ]):
            with col:
                st.markdown(f"""<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;
                    padding:1rem;text-align:center;box-shadow:0 2px 6px rgba(0,0,0,.06)">
                    <div style="font-size:1.4rem">{icon}</div>
                    <div style="font-size:.75rem;color:#718096;font-weight:500">{label}</div>
                    <div style="font-size:.82rem;font-weight:700;color:#2d3748">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2, t3 = st.tabs(["📋 Booking Plan", "🗺️ Itinerary", "🔍 Research"])
        with t1:
            st.markdown('<span class="agent-badge badge-coordinator">📋 Coordinator — LCEL + Memory + RAG</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.booking_output}</div>', unsafe_allow_html=True)
        with t2:
            st.markdown('<span class="agent-badge badge-planner">🗺️ Planner</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.plan_output}</div>', unsafe_allow_html=True)
        with t3:
            st.markdown('<span class="agent-badge badge-researcher">🔍 Research Agent</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="output-box">{st.session_state.research_output}</div>', unsafe_allow_html=True)

        st.markdown("---")
        full_report = f"""✈️ AI TRAVEL PLAN — Full LangChain Edition
{'='*60}
{td['origin']} → {td['destination']} | {td['travel_dates']}
{td['travelers']} traveler(s) | {td['budget']}
Preferences: {td['preferences']}

LangChain features used: LCEL Chains, Memory, Tools (DuckDuckGo+Wikipedia),
RAG (FAISS), ReAct Agent, Streaming

{'='*60}
RESEARCH REPORT (ReAct Agent + Live Search)
{'='*60}
{st.session_state.research_output}

{'='*60}
DAY-BY-DAY ITINERARY (LCEL + Memory + RAG)
{'='*60}
{st.session_state.plan_output}

{'='*60}
BOOKING ACTION PLAN (LCEL + Memory + RAG)
{'='*60}
{st.session_state.booking_output}
"""
        _, col_dl, _ = st.columns([1,2,1])
        with col_dl:
            st.download_button(
                "📥 Download Complete Travel Plan",
                data=full_report,
                file_name=f"travel_{td['destination'].replace(', ','_').replace(' ','_')}.txt",
                mime="text/plain",
                use_container_width=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Plan Another Trip", use_container_width=True):
                clear_all_memory()
                for k in ["phase","research_output","plan_output","booking_output","trip_details"]:
                    st.session_state[k] = 0 if k=="phase" else ({} if k=="trip_details" else "")
                st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center;color:#a0aec0;font-size:.8rem;padding:.5rem">
    🦜 <strong>LangChain</strong> LCEL &bull; Memory &bull; DuckDuckGo+Wiki Tools &bull; RAG &bull; ReAct &bull; Streaming
    &nbsp;+&nbsp; 👤 <strong>Human-in-the-Loop</strong> &nbsp;+&nbsp; 🎈 <strong>Streamlit</strong>
</div>""", unsafe_allow_html=True)

# ✈️ AI Travel Booking Assistant
### CrewAI + Human-in-the-Loop + GPT-4o-mini + Streamlit

---

## 🧠 How It Works

This app uses **3 CrewAI agents** with **2 Human-in-the-Loop checkpoints**:

```
[User Input]
     ↓
[🔍 Research Agent]  ← GPT-4o-mini
     ↓
[👤 YOU REVIEW]      ← Add/Edit/Approve
     ↓
[🗺️ Planner Agent]  ← GPT-4o-mini
     ↓
[👤 YOU REVIEW]      ← Add/Edit/Approve
     ↓
[📋 Coordinator Agent] ← GPT-4o-mini
     ↓
[✅ Download Plan]
```

**Human-in-the-Loop** means YOU stay in control — the AI suggests, you decide.

---

## 📁 File Structure

```
├── app.py              # Streamlit UI + session state + HITL flow
├── travel_crew.py      # CrewAI Agents & Tasks
├── requirements.txt    # Python dependencies
└── README.md
```

---

## 🚀 Deploy to Streamlit Cloud (3 steps)

### Step 1: Push to GitHub
```bash
git init
git add .
git commit -m "AI Travel Booking Assistant"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repo
4. Set **Main file**: `app.py`
5. Click **Deploy**

### Step 3: Enter Your API Key
- In the running app → sidebar → paste your **OpenAI API key**
- Key is only stored in browser session (never persisted)

---

## 🔑 API Keys

| Key | Required | Where to get |
|-----|----------|--------------|
| `OPENAI_API_KEY` | ✅ Yes | [platform.openai.com](https://platform.openai.com) |
| `SERPER_API_KEY` | ❌ Optional | [serper.dev](https://serper.dev) — enables live web search |

**Note**: The OpenAI key is entered in the UI sidebar. For Serper, add it as a Streamlit Cloud secret:
- Go to App Settings → Secrets → add `SERPER_API_KEY = "your-key"`

---

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🤖 The 3 Agents

### 🔍 Research Agent — `Senior Travel Researcher`
- Finds flight options with realistic costs
- Suggests 3 hotel tiers (budget/mid/luxury)
- Lists top 10 activities & local cuisine
- Provides full budget breakdown

### 🗺️ Planner Agent — `Expert Itinerary Planner`
- Creates day-by-day schedule (morning/afternoon/evening)
- Includes meal recommendations per day
- Adds daily budget estimates
- Provides packing list & travel hacks

### 📋 Coordinator Agent — `Travel Booking Coordinator`
- Prioritized booking checklist with platforms
- Budget summary table
- Pre-departure checklist
- Emergency contacts & local phrases

---

## 💰 Cost Estimate

With GPT-4o-mini:
- ~$0.05–0.15 per complete trip plan (3 agent runs)
- Very affordable for demos and prototypes

---

## 🛠️ Customization Ideas

1. **Add real flight search** — integrate Amadeus or Skyscanner API
2. **Add hotel search** — Booking.com or Hotels.com API  
3. **Enable Serper** — let Research Agent search the web live
4. **Add PDF export** — use `reportlab` to generate a PDF travel booklet
5. **Add email** — send the final plan via SMTP
6. **Multi-destination** — extend for multi-city trips

---

Built with ❤️ using CrewAI + Streamlit + GPT-4o-mini

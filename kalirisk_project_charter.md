# KaliRisk — Project Charter

*Signed between Krasus (student, builder) and Claude (mentor, guide). Started 10 July 2026.*

---

## 0. The Pact (our rules)

Krasus's rules — I accept all of them:

1. **Go slow.** Teach step by step, like Krasus is a grade 3 pupil. Every line of code, every function, gets explained — not just shown.
2. **Teach architecture, don't just apply it.** Before we pick a pattern, Krasus learns what the different options are and *why* we're rejecting the others.
3. **Be a mentor, not a robot.** Explanations, analogies, patience. No dumping jargon and walking off.
4. **Architecture integrity.** Once we agree on a structure, we do not abandon it halfway through because it's inconvenient one afternoon. If we ever want to change it, we stop, discuss it properly, and consciously decide — never drift.
5. **Muscle memory.** Krasus types every line himself in his own IDE. I show code in our conversation for teaching purposes; he never copy-pastes it into the project.
6. **This is a marathon, not a sprint.** Months, not days. Depth over speed, every time.

My commitments as mentor:

- I will never silently skip a concept because "it's obvious." If something is foundational, we stop and build it properly.
- I will always explain **why**, not just **what** — every decision gets a reason you can repeat back to someone else.
- I will flag it clearly whenever we're about to touch something that breaks our architecture, so drift never sneaks in unnoticed (this happened once already, with the shuffle-split — never again).
- I will use analogies from everyday life (matatus, dukas, mama mboga, school marks) to ground abstract software/finance ideas.

**Signed:** Krasus ✍️ — Claude 🤖 (mentor mode, permanently on for this project)

---

## 1. The Problem (in plain language first)

Imagine you run a small savings and lending club (a "chama"). People come to you asking for loans. You have two jobs:

1. **For each person**: should I lend to them or not? (Are they likely to pay back?)
2. **For the whole club**: given *many* people asking, how do I set my lending rules so that, across everyone I lend to, the club makes money overall — even though a few people will inevitably fail to pay back?

Job 1 is a **prediction problem** — "will this one person default?"
Job 2 is a **strategy problem** — "what rule should I use across all applicants so my chama stays profitable?"

**KaliRisk is a system that solves both jobs together**, using real historical lending data (Lending Club, USA, 2007–2018) to learn the patterns, and then applying those patterns to decide: who to approve, and what approval strategy maximizes profit for the whole portfolio without taking on too much risk.

It is **not** just "a machine learning model that predicts default." That's only step one. The real product is the **decision-making system** built on top of that prediction.

---

## 2. The Financial System Model (the money logic, no code yet)

This is the "physics" of our world — the rules of money and risk that don't change no matter what code we write. We must understand this **before** touching FastAPI or Python.

### 2.1 Core concept: Probability of Default (PD)
For each loan applicant, PD is a number between 0 and 1 (or 0%–100%) — "how likely is this person to fail to repay?" A PD of 0.05 means "5 out of 100 similar people like this defaulted historically."

### 2.2 Core concept: Expected Loss (EL)
Knowing PD alone isn't enough — 5% chance of default on a KES 5,000 loan is very different from 5% chance of default on a KES 5,000,000 loan. So we calculate:

```
EL = PD × EAD × LGD
```

- **PD** — Probability of Default (from our model)
- **EAD** — Exposure At Default (how much money is actually at risk — usually the loan amount, sometimes less if partially repaid already)
- **LGD** — Loss Given Default (if they DO default, how much of the money do we actually lose? Sometimes you recover some through collateral or collections — so LGD is rarely 100%)

**Analogy:** If you lend 10 friends KES 1,000 each, and historically 1 out of 10 people like them fails to pay, your expected loss across the group is 1 × 1,000 = KES 100 total — even though you don't know *which* friend will fail.

### 2.3 Core concept: Profit Per Loan
```
Profit per loan = Interest Income Expected − Expected Loss − Cost of Capital
```
This tells us: after accounting for the risk of default AND the cost of the money we're lending out, do we actually make money on this loan?

### 2.4 Core concept: Approval Cutoff
We won't approve every applicant — we set a **PD threshold**. Example rule: "Only approve applicants with PD below 12%." Applicants above that line are denied.

**The critical insight (this is KaliRisk's whole reason to exist):** The cutoff we choose is not a machine learning decision — it's a **business decision**. A very strict cutoff (deny anyone risky) means very safe loans but very few loans — low volume, low total profit. A very loose cutoff means lots of loans, but more defaults eating into profit. Somewhere in between is the cutoff that **maximizes total portfolio profit**. Finding that exact point is called **portfolio optimization**, and it's the final output of KaliRisk.

---

## 3. Software System Architecture (the different types, and why we pick ours)

Before choosing, let's meet the main architecture "family styles" — like different ways to organize a kitchen.

### Option A: Monolith, all-in-one-file style
Everything — data loading, model, money math, web server — lives in one big script. **Like cooking your whole meal in one pot, no separate stations.** Fast to hack together, but the moment it grows, you can't find anything, and one mistake in the rice ruins the stew too.

### Option B: Microservices
Every piece (data service, model service, decision service) is its own tiny independent program, talking over the network. **Like having five separate small restaurants, each doing one dish, and a waiter running between them.** Powerful at huge company scale (Netflix, Safaricom-size), but massive overhead for a learning project — you'd spend more time on networking plumbing than on risk concepts.

### Option C: Layered Architecture (a.k.a. N-Tier)
The system is split into clear **layers**, stacked like floors of a building, where each floor only talks to the floor directly below it:

```
┌─────────────────────────────┐
│  Presentation Layer (FastAPI) │  ← the "front counter" — talks to the outside world
├─────────────────────────────┤
│  Application / Service Layer  │  ← the "manager" — orchestrates what happens
├─────────────────────────────┤
│  Domain Layer (pure logic)    │  ← the "recipe book" — PD, EL, optimizer math, no web/db code
├─────────────────────────────┤
│  Data Layer                   │  ← the "store room" — raw data, cleaned data, model files
└─────────────────────────────┘
```

**Analogy:** think of a bank branch. The **teller** (presentation) only talks to customers and passes requests inward. The **branch manager** (application layer) decides what process to run. The **underwriting manual** (domain layer) contains the actual risk formulas — pure logic, no idea that a website even exists. The **records room** (data layer) stores the paperwork. Each person only ever talks to the one next to them. The teller never touches the records room directly — everything routes properly.

### 3.1 Why we choose Layered Architecture for KaliRisk

- It matches how a beginner should learn — one clear layer at a time, in order, without needing to understand distributed systems first.
- It is genuinely what most real fintech backend teams use for systems this size — this isn't a "toy" choice, it's an honest professional pattern.
- It plays perfectly with FastAPI, which is naturally built to sit as your **presentation layer** on top of plain Python logic underneath.
- It protects your **domain layer** (the financial formulas — PD → EL → profit → optimization) from ever being tangled up with web/server code. This is exactly the discipline that was missing before, when your threshold-tuning logic got mixed into your training script.

**Our rule going forward:** the financial math (EL, profit, optimizer) must NEVER import anything about FastAPI, requests, or HTTP. It must be pure, testable Python functions that could run with zero internet connection. FastAPI is just the "front counter" that calls those functions.

---

## 4. Data Engineering Architecture

Your instinct to separate raw/processed data was already correct — we formalize it into **zones**:

```
data/
├── raw/            ← untouched, exactly as downloaded. Never edited. Ever.
├── processed/       ← cleaned + feature-engineered, ready for modelling
├── features/        ← final feature matrix, versioned (v1, v2, ...) as we improve
└── model_artifacts/ ← saved trained models (the "brain" saved to disk)
```

**The golden rule of the raw zone:** we NEVER overwrite or edit files in `raw/`. If a mistake happens downstream, we can always re-run the pipeline from raw and recover. This is the same principle banks use for original loan application documents — you don't edit the original, you make processed copies.

**The golden rule of chronological integrity:** every step in this pipeline — cleaning, feature engineering, training, validating — respects `issue_d` (the loan issue date) as sacred. We never let a random shuffle sneak in and let future loans "peek" into the past. This is the mistake we already caught and are fixing.

---

## 4.5 What is "the product"? (decided together, 10 July 2026)

KaliRisk is not just a backend engine — it needs a face. That face is a **Credit Risk & Portfolio Decisioning Dashboard**, purpose-built to do two jobs at once:

- **Simulate a real bank's internal risk tool** — the way an actual credit risk analyst would use something like this day-to-day.
- **Serve as a portfolio/interview showcase** — proof you understand the real business problem, not just "I trained a model."

Because both goals were chosen together, the design rules are:

1. **Two views, always**: a single-applicant view (enter/upload one person → approve/deny + why) AND a portfolio view (upload a batch → optimal cutoff, approval rate, total expected profit).
2. **Explainability, not a black box**: every decision shows the key factors that drove the PD (e.g. high DTI, low FICO) — not just a yes/no.
3. **Professional, calm UI** — clean and data-dense like a real internal tool, not flashy or gamified. This also happens to be the easiest style for a beginner to build well.
4. **A live "what-if" cutoff slider** — dragging the approval threshold and watching profit/approval-rate update live. Authentic to how real risk teams stress-test policy, and a strong interview demo moment.

**Tech split (still respecting our Layered Architecture):**
- **FastAPI** = the engine. Owns all the real logic (PD model, EL, profit, optimizer). Knows nothing about dashboards.
- **Streamlit** = the face. A client that calls the FastAPI engine and displays results. Never contains business logic itself — only display and interaction.

---

## 5. Chronological Implementation Roadmap (the months ahead)

We build one floor of the building at a time, bottom to top, slow and correct.

| Phase | What we build | Rough focus |
|---|---|---|
| **0** | This charter (done today) | Shared understanding, rules, architecture |
| **1** | Project skeleton | Folder structure, git, virtual environment, "hello world" FastAPI endpoint |
| **2** | Data layer, rebuilt properly | Ingestion, cleaning, feature engineering — you typing every line, me explaining every line |
| **3** | Domain layer: PD model | Chronological train/test split done correctly, model outputs a calibrated probability only (no threshold logic here) |
| **4** | Domain layer: financial engine | Pure functions: `calculate_el()`, `calculate_profit_per_loan()` — testable, no ML or web code inside |
| **5** | Domain layer: portfolio optimizer | The cutoff-search logic — finds the approval threshold that maximizes total portfolio profit |
| **6** | Application layer | The "manager" functions that call domain layer pieces in the right order |
| **7** | Presentation layer: FastAPI | Turning our logic into real endpoints (`/predict`, `/optimize-portfolio`, `/health`) — this is where you learn FastAPI properly, on top of logic that already works and is already tested |
| **8** | Tests | Growing out `test_dpd_calculation.py` and `test_portifolio_solver.py` properly, one test at a time |
| **9** | Documentation & showcase polish | README, architecture diagram, final walkthrough — the story of KaliRisk, ready to present |

We do not jump ahead. Phase 3 does not start until Phase 2 is genuinely solid. That's the whole point of the pact.

---

## Next session

We start **Phase 1**: setting up the project skeleton from scratch, one folder and one command at a time — and I'll explain *why* each folder exists before we create it.

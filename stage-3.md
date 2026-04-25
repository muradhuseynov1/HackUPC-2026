# Phase 3: Interact

## 3.1 Mission & Goal

Build an intelligent interface that bridges the gap between complex industrial telemetry and human understanding. The goal is to transform the raw, timestamped data produced in Phase 2 into **actionable intelligence** through natural language interaction.

The interface must not just "chat"; it must serve as a **diagnostic window** into the Digital Twin. AI is the star of this phase — your system must reason over the data, not invent it.

---

## 3.2 The Integration Bridge (Connecting Phase 2 to Phase 3)

**Crucial Instruction:** Your Phase 3 interface is a consumer of your Phase 2 "Historian." It must be able to query the data produced in the previous phase to generate its responses.

* **The Relationship:** The AI does not "know" the printer state; it must **retrieve** it.
* **The Data Flow:**
    1. **The User** submits a natural language query.
    2. **The Interface** translates that query into a search/query for the Phase 2 database.
    3. **The Interface** provides the retrieved telemetry to the AI as context.
    4. **The AI** generates a response grounded *only* in that specific data.

The query layer should support lookups by timestamp range, component, scenario or run identifier, and current state. This keeps the interface aligned with the Phase 2 historian instead of hard-coding responses.

---

## 3.3 Reasoning Ladder (Design Patterns)

You must choose one of the following three architectural patterns to implement your interface. This choice determines how the AI accesses the "Truth" (the Phase 2 data).

| Pattern | Requirement Level | Core Logic | Architectural Impact |
| :--- | :--- | :--- | :--- |
| **A: Simple Context Injection** | **Mandatory Baseline** | The current state is injected directly into the AI's system prompt. | **Static Prompting:** The AI only knows the "now." It is excellent for "What is the status?" but cannot answer questions about the past. |
| **B: Contextual RAG** | **Advanced** | The system searches the Phase 2 Historian to find relevant context before answering. | **Retrieval Logic:** Requires a mechanism to query the database based on the user's question (e.g., "What happened at 2 PM?"). |
| **C: Agentic Diagnosis** | **Advanced (Highest Tier)** | The AI uses "Tools" to autonomously investigate the data through multi-step reasoning. | **Agentic Loop:** The AI doesn't just read; it *searches, compares, and concludes* (e.g., "I see a heat spike $\rightarrow$ I will check the fan logs $\rightarrow$ I conclude the fan failed"). |

---

## 3.4 Developer's Strategy: How to Approach the AI Layer

Your choice of pattern will fundamentally change how you build your interface.

**1. The "Safety First" Approach (Pattern A):**
If you are focused on ensuring the math and the simulation are solid, start here. You simply take the latest snapshot from your Phase 2 engine and append it to the AI's prompt. This is the fastest way to get a functional, grounded chatbot.

**2. The "Intelligence" Approach (Pattern B):**
If you want your AI to be a true historian, implement **Contextual RAG**. You will need to build a bridge that allows the user's question to trigger a database query, retrieving the specific historical context required to answer it.

**3. The "Expert" Approach (Pattern C):**
If you want to provide the wow factor to the competition, implement **Agentic Diagnosis**. This requires a "Reasoning Loop" (such as a ReAct pattern). The AI must be able to say: *"I don't have the answer yet; let me use the 'Query_Database' tool to check the temperature logs from the last hour."* This is the highest level of Digital Twin sophistication.

---

## 3.5 The Grounding Protocol (The Rule of Truth)

**The core principle of this phase:** your AI must reason over the simulation data, not invent it. This is what separates a genuine Digital Co-Pilot from a generic chatbot.

**The Rule:** The AI must act solely as a **Reasoning Layer** over the data provided by the Phase 2 Historian. It must not use its own training knowledge to answer questions about the printer's state. Every response must be traceable to a specific timestamp or telemetry point in your database.

The method of achieving this grounding depends on your chosen pattern:

* **For Pattern A (Simple Context Injection):** Include the current "State Report" directly within the AI's system prompt. The AI is grounded because its only available context is the data you have provided.
* **For Pattern B (Contextual RAG):** Build a retrieval mechanism that searches the Phase 2 Historian to find the relevant context before the AI generates a response.
* **For Pattern C (Agentic Diagnosis):** The AI uses a specific "Query Tool" to fetch the necessary data from the Phase 2 Historian to support its multi-step reasoning.

Grounding accuracy is a key evaluation criterion — an AI that reasons well over real data is far more impressive than one that invents plausible-sounding answers.

---

## 3.6 The Data Contract (Inputs & Outputs)

### 3.6.1 Expected Inputs (The Query)

The interface must accept:

* **Natural Language:** Text-based queries (e.g., via a chat window).
* **Optional:** Voice-to-text input.

Typical questions should include the current state, the history of a component, trend changes over time, and comparisons between simulation runs.

### 3.6.2 Expected Outputs (The Response)

Every response generated by the AI must include:

* **Grounded Text:** A clear, plain-language explanation.
* **Evidence Citation:** A reference to the specific data point or timestamp used (e.g., *"Based on the telemetry at 14:05:02..."*).
* **Severity Indicator:** A categorization of the information (e.g., `INFO`, `WARNING`, `CRITICAL`).

If the answer depends on a specific run or scenario, the response must cite that run identifier as well as the supporting timestamp.

---

## 3.7 Minimum Requirements

To successfully complete this phase, your interface must meet these requirements:

1. **Implement** Level 1 Reasoning (Data Retrieval).
2. **Adhere** to the Grounding Protocol (No hallucinations).
3. **Provide** evidence citations for every answer.

---

## 3.8 Go Further

The minimum gives you a working, grounded AI interface. These ideas can make it genuinely impressive. They are starting points — feel free to take them further or invent your own.

**1. Proactive Alerting**
Don't wait for the operator to ask. Monitor the historian in the background and fire an alert the moment a component crosses a danger threshold — before the operator even notices. Turn a reactive chatbot into a proactive partner.

* *Example:* "Alert: The printhead temperature has exceeded the safety threshold for 3 consecutive cycles. Recommend inspection."

**2. Voice & Hands-Free Interface**
What if an operator on the factory floor could talk to the machine while their hands are busy? Build a voice-to-voice interface — spoken question in, spoken answer out — and watch the demo come alive.

**3. Root-Cause Diagnosis**
Go beyond "component X is at CRITICAL." Trace the chain of events in the historian, explain *why* it got there, and present a clear story: "The heating elements degraded over 12 hours, raising thermal stress above threshold, which then accelerated nozzle plate clogging." That's the difference between a dashboard and a co-pilot.

**4. Action Paths**
Tell the operator not just what is wrong, but what to do: estimated repair time, the impact on uptime if ignored, and a priority ranking across all active failures. Give them a decision, not just a status.

**5. Autonomous Collaborator**
The highest tier: a co-pilot that remembers past sessions, learns the operator's habits, and proactively surfaces insights they didn't know to ask for. It identifies repetitive queries and automates them. It prepares reports before you request them. This is the machine that truly *communicates*.

---

## 3.9 Deliverables & Evaluation

### What to deliver

* **A data-retrieval interface** — the AI queries the Phase 2 historian to answer questions; it does not answer from memory or hard-coded logic.
* **Grounding protocol compliance** — every response is traceable to a specific data point; no hallucinations.
* **Evidence citations** — every answer includes an explicit reference (timestamp, component, run ID) to the data that supports it.

### How it will be evaluated

Teams will be evaluated based on four primary pillars. A team can score high by being exceptionally reliable (Pillar 1) or by being exceptionally advanced (Pillars 3 & 4).

| Pillar | Criterion | Description |
| :--- | :--- | :--- |
| **1. Reliability** | **Grounding Accuracy** | Does the AI stay strictly within the bounds of the Phase 1/2 data? Zero tolerance for hallucinations. |
| | **Communication Clarity** | How well does the AI translate complex telemetry into clear, actionable, human-readable insights? |
| **2. Intelligence** | **Reasoning Depth** | Does the AI move beyond simple data retrieval into genuine diagnostic reasoning and root-cause analysis? |
| **3. Autonomy** | **Proactive Intelligence** | Does the system alert the operator *before* they ask, or proactively surface relevant insights? |
| | **Collaborative Memory** | Does the agent demonstrate persistent memory and skill automation across sessions? |
| **4. Versatility** | **Interaction Modality** | How seamless and natural is the interaction — voice, visual dashboards, multi-modal? |

---
[Back to Main Brief](./hackathon.md)

# 🏆 Hackathon Brief: When AI meets reality
## 1. The Vision

The future of manufacturing is not just about more powerful machines; it is about **smarter collaboration between humans and machines.** 

As industrial hardware like the **HP Metal Jet S100** reaches unprecedented levels of complexity, the gap between machine capability and human intuition grows wider. An operator can no longer rely on manual monitoring or reactive maintenance to manage a machine of this sophistication. They need more than a dashboard; they need a **partner.**

**Your mission is to build that partner.**

You are not just building a simulation; you are building a **Digital Co-Pilot.** You are creating an intelligent, living entity that understands the physics of the machine, predicts its future, and communicates its needs in a way that empowers the human operator. You are bridging the gap between high-fidelity industrial physics and intuitive human intelligence.

---

## 2. The Challenge: From Physics to Intelligence

To bring this Digital Co-Pilot to life, you must solve three fundamental engineering challenges:

* **The Brain (Modeling):** You must translate the physical reality of metal jetting—wear, thermal stress, and component degradation—into a robust mathematical model.
* **The Life (Simulation):** You must breathe life into that model, creating a "Digital Twin" that evolves, ages, and reacts to the world in a continuous, time-advancing simulation.
* **The Voice (Interaction):** You must build the bridge. You will create an intelligent interface that transforms raw, complex telemetry into proactive, conversational intelligence—turning a machine that "runs" into a machine that "communicates."

---

## 3. Roadmap & Scope
The challenge is structured into three sequential (and cumulative) phases:

| Phase | Name | Focus | Documentation |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Model | Modeling (The Brain) | [View Phase 1 Details](./stage-1.md) |
| **Phase 2** | Simulate | Digital Twin (The Simulation) | [View Phase 2 Details](./stage-2.md) |
| **Phase 3** | Interact | AI / Voice Interface (The Interaction) | [View Phase 3 Details](./stage-3.md) |

---

## 4. Technical Constraints & Rules
To ensure a level playing field, all teams must adhere to the following constraints:

* **Technology Stack:** There is no mandatory technology stack. Teams are free to use any programming language (Python, C++, Rust, etc.), frameworks, or cloud platforms.
* **Data Generation (Crucial):** **No pre-made telemetry or historical data files will be provided.** Teams are responsible for implementing the logic that generates the synthetic telemetry (the sequence of states) required to drive their simulation.
* **AI & LLM Components:** AI and LLM components are **encouraged and will be highly valued** by the judges. When used, they must reason over the data produced by your Phase 1 and Phase 2 models — not over the model's own training knowledge. Every AI-generated answer must be traceable to a specific data point in your simulation.
* **Scope:** The focus is on the *Digital Twin* (the software representation). Integration with actual physical HP hardware is out of scope.

---

## 5. Submission Requirements
To be eligible for prizes, teams must submit a complete package containing:

1.  **Working Demo:** A live demonstration covering at least **Phase 1 + Phase 2** (Phase 3 is a significant bonus). Each phase has specific deliverables:
    * **Phase 1 (Model):** At least 3 degradation models coded, each using at least one input driver, plus a clear explanation of how the degradation logic was implemented.
    * **Phase 2 (Simulate):** A time series visualization showing component health evolving over simulated time, a degradation model explanation, and a failure analysis identifying when and why each component fails.
    * **Phase 3 (Interact, bonus):** A data-grounded conversational interface with explicit evidence citations for every response and no hallucinations.
2.  **Documentation & Delivery Package:**
    * **Architecture Slide Deck:** A presentation explaining your mathematical models, your simulation loop, and your AI implementation. Architecture & application design diagrams are highly encouraged.
    * **Technical Report:** A detailed report describing your modeling approach, simulation design, and AI implementation. This should include a discussion of the challenges you faced and how you overcame them.
    * **Phase 3 Bonus Demo:** If you implement Phase 3 (the AI/Voice Interface), include a demo of the conversational interface in action, along with a detailed explanation of how it works and how it is grounded in your simulation data.
3.  **GitHub Repository:** A clean, well-documented repository including a `README.md` with setup instructions.
4.  **Walkthrough:** A short walkthrough demonstrating the functionality and the "intelligence" of your Digital Twin.
5.  **Code Quality:** Your code should be clean, modular, and well-documented.

---

## 6. Pre-Demo Self-Check
Use this checklist before the live demo so the team can verify the project is complete and grounded.

### Phase 1: Model

* [ ] At least one component per subsystem is modeled (Recoating, Printing, Thermal).
* [ ] Each modeled component uses at least one input driver.
* [ ] At least two failure models are implemented.
* [ ] Component outputs include health index, operational status, and metrics.
* [ ] The team can explain how the degradation logic was designed and implemented.

### Phase 2: Simulate

* [ ] The simulation loop advances time correctly and consistently.
* [ ] The Phase 1 engine is called at every time step.
* [ ] Every record is saved to a historian with a timestamp and driver values.
* [ ] Runs or scenarios can be identified separately.
* [ ] A time series visualization shows component health evolving over time.
* [ ] The team can identify when and why each component fails.

### Phase 3: Interact

* [ ] The interface reads from the Phase 2 historian.
* [ ] Responses are grounded in the simulation data, not hard-coded or hallucinated.
* [ ] Every answer includes an explicit citation or timestamp reference.
* [ ] Any advanced reasoning remains traceable to the underlying logs.

### Demo Readiness

* [ ] The team can explain how the three phases connect.
* [ ] The demo can be reproduced end to end.
* [ ] The team can show where any AI answer comes from in the data.
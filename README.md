Decision System Balancing learning and safety

1. PROBLEM WE STARTED WITH

The goal was to build a system that decides:

“What action should we take for a given user at a given time?”

We were not just predicting behaviour — we were trying to optimise decisions over time.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHAT THIS PROJECT DEMONSTRATES

- Decision systems vs prediction systems  
- Exploration vs exploitation trade-offs  
- Dynamic user behaviour  
- System design with constraints  
- Long-term vs short-term optimisation  
- Real-world ML challenges (latency, scaling)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. WHAT THE ML MODEL ACTUALLY DID

We trained a supervised model (classification model, random forest classifier) to estimate:

P(response = 1 | user_type, prev_action, prev_response, action)

This means:
- The model takes a state + action as input  
- And predicts the probability of user response  

Key Insight:
The model predicts the outcome of an action, not the user’s next action.

What the model does NOT do:
- It does not choose actions  
- It does not optimise anything  
- It does not think long-term  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3. TURNING MODEL INTO A DECISION SYSTEM

Instead of asking:
“What will the user do?”

We ask:
“What will happen IF I take each possible action?”

Mechanism:
- Try all actions  
- Score each action using the model  
- Pick the action with the highest expected reward  

Key Insight:
A predictive model becomes a decision system when you evaluate all possible actions and select the best one.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4. GREEDY POLICY (PROJECT 1)

System:
Always pick the highest probability action

Limitation:
- No exploration  
- Repeats the same action  
- Gets stuck in local optimum  

Key Insight:
Greedy systems maximise short-term reward, not long-term value.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5. ADDING EXPLORATION (EPSILON-GREEDY)

With probability ε → explore (random action)  
With probability (1 - ε) → exploit (best action)

Why exploration is needed:
- The model may be wrong  
- Better actions may exist  
- The environment may change  

Key Insight:
Exploration allows the system to discover better strategies that greedy policies would never try.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6. EPSILON DECAY

Start high → gradually reduce

Interpretation:
- Early → learning phase  
- Later → optimization phase  

Key Insight:
Exploration should decrease as the system becomes more confident.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

7. ADDING CONSTRAINTS (SAFETY LAYER)

Problem:
Without constraints → repetitive and spammy actions

Solution:
- Limit repeated actions  
- Override model decisions
- Constraint of 3 same actions, after the same action 3rd time, block the same next action

Example:
offer → offer → offer → blocked → forced different action  

Key Insight:
Exploration enables learning; constraints ensure safe behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

8. SYSTEM ARCHITECTURE

State → Model → Agent → Constraints → Final Action

Key Insight:
Real-world systems separate prediction, decision-making, and safety into different layers.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

9. EXPERIMENT DESIGN

We compared:
1. Rule-based system  
2. Greedy ML system  
3. AI system (exploration + constraints)

Key Insight:
To understand improvements, change only one component at a time, I kept the dataset and the number of days the same.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

10. INITIAL RESULT (STATIC ENVIRONMENT)

Observation:
Greedy > AI > Rule-based

Reason:
- Preferences fixed  
- Environment simple  
- Greedy is already optimal  

Key Insight:
Exploration can hurt performance in stable environments.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

11. INTRODUCING DYNAMIC ENVIRONMENT

Change:
User preferences shift over time

Example:
Day 1–5 → offer works  
Day 6–20 → video works  

Key Insight:
Real-world user behaviour is dynamic, not static.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

12. RESULT IN DYNAMIC ENVIRONMENT

Observation:
AI > Greedy > Rule-based

Reason:
- Greedy stuck in past  
- AI adapts via exploration  

Key Insight:
Exploration becomes critical when the environment changes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

13. IMPORTANCE OF TIME HORIZON

Observation:
- Short horizon → greedy wins  
- Long horizon → AI wins  

Key Insight:
Exploration only pays off when there is enough time for learning to compound.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

14. EXPLORATION TRADE-OFF

Observation:
- Early → AI worse  
- Later → AI better  

Key Insight:
Exploration sacrifices short-term reward to improve long-term outcomes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

15. VISUALIZATION INSIGHTS

LTV Curve:
AI overtakes greedy after ~10 days  
→ Learning takes time  

Exploration Rate (~6%):
→ Even a small exploration is enough  

Action Distribution:
Early → diverse  
Later → converges  
→ System learns and stabilises  

Constraint Rate (~20%):
→ Safety actively shapes decisions  

AI vs Greedy Gap:
Widens after ~day 7  
→ Adaptation drives advantage  

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

16. PERFORMANCE BOTTLENECK

Problem:
Simulation slow due to repeated model calls

Root Cause:
Repeated feature encoding inside loops

Key Insight:
Inference efficiency is often more critical than model training.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

17. PREDICTION VS DECISION

Prediction:
“What will happen?”

Decision:
“What should I do?”

Key Insight:
Prediction provides information; decision systems use that information to act.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

18. WHY NOT PREDICT NEXT ACTION?

Alternative:
P(action | state)

Problem:
- Learns past behaviour  
- Not optimal behavior  

Key Insight:
Data reflects what was done, not what should be done.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

19. COUNTERFACTUAL THINKING

System evaluates:
“What happens if I take action A, B, or C?”

Key Insight:
Decision systems require evaluating multiple possible futures.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

20. WHY CONSTRAINTS ARE NECESSARY

Even with learning:
- The system may exploit bad patterns  
- May harm user experience  

Key Insight:
Optimisation without constraints can lead to harmful outcomes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

21. FINAL SYSTEM UNDERSTANDING

State → Evaluate actions → Explore/Exploit → Apply constraints → Act → Learn

Final Insight:
A real decision system balances learning, optimisation, and safety.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

META INSIGHT

Machine learning is not just about models.
It is about building systems that make better decisions over time.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY OBSERVATIONS

- AI system underperforms initially due to exploration  
- Surpasses greedy after ~10 days  
- Explore = 6%, Exploit = 94%  
- Even a small exploration is enough to discover better strategies  

Insight:
Even with only ~6% exploration, the system adapts to shifting reward patterns.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEHAVIORAL INSIGHTS

- Early → diverse actions  
- Later → converges to dominant strategy (offers shown) 
- Notifications decrease over time  

Insight:
System learns and stabilises toward optimal behaviour.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CONSTRAINT INSIGHT

Constraint trigger rate ≈ 20%

Insight:
Constraints actively shape behaviour without hurting performance significantly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TEMPORAL INSIGHT

- AI vs Greedy gap increases after ~day 7  
- Crossover happens around ~day 10  

Insight:
Adaptation to changing preferences drives long-term gains.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FINAL INSIGHT

In the early phase, the greedy system performs better because it exploits known high-reward actions.

However, as user preferences shift, the greedy system becomes stuck in outdated patterns.

The epsilon-greedy system, despite exploring only ~6% of the time, discovers new strategies and adapts.

This leads to a crossover point around day 10, after which AI consistently outperforms greedy.

Constraints, triggered ~20% of the time, ensure safe behaviour without harming performance.

Overall:
In dynamic environments, even limited exploration combined with safety constraints improves long-term outcomes.



This system is NOT full reinforcement learning.

It is closer to a contextual bandit:

- Single-step optimization
- No state transition learning
- No Bellman updates
- No online learning

Exploration reduces bias but does not fully solve the counterfactual problem.





# Theory: risk-aware CBS and delay-tolerant ADG execution

This document states the model behind the implementation and proves the properties
claimed in the README: optimality and bounded suboptimality of the risk-aware
("probabilistic") CBS, the trade-off between human risk and plan cost, and
deadlock-freedom and completeness of the execution layer under bounded delays. Every
statement points to the code that implements it and to the test that checks it
empirically.

## 1. Model

**Grid and plans.** $G=(V,E)$ is the 4-connected graph of free MAPF cells
(`params/mapf_<map>.yaml`). Time is discrete, $t=0,1,\dots$. Each action (wait or move
to a neighbour) takes one step. Agent $i$ has start $s_i$ and goal $g_i$. A path
$\pi_i = (\pi_i(0)=s_i,\dots,\pi_i(T_i)=g_i)$ is followed by the agent staying at
$g_i$ forever, so $\pi_i(t)=g_i$ for $t\ge T_i$.

**Conflicts.** Two paths conflict if they have

* a *vertex* conflict: $\pi_i(t)=\pi_j(t)$,
* a *swap* conflict: $\pi_i(t)=\pi_j(t+1) \wedge \pi_i(t+1)=\pi_j(t)$, or
* a *$k$-robustness* conflict (Atzmon et al., 2018): $\pi_i(t)=\pi_j(t+d)$ for some
  $1\le d\le k$.

With $k=0$ this is classical MAPF, which allows *following* (entering a cell as its
occupant leaves) and rotations. The coordinator uses $k=1$ (§4).

**Reservations.** A reservation is a timed path $\rho$ of a robot that is not being
re-planned. It occupies $\rho(t)$ at time $t$ and $\rho(\text{end})$ forever
afterwards. Planned agents must be conflict-free with all reservations.

**Human risk.** $r(t,v)\in[0,1]$ is the predicted probability that a human occupies
cell $v$ at step $t$ (`prediction.py`). Layers beyond the horizon $H$ repeat the last
layer, a static prior. The cost of a path and of a joint plan $\Pi$ are

$$c_\lambda(\pi_i)=\sum_{t=1}^{T_i}\bigl(1+\lambda\,r(t,\pi_i(t))\bigr),\qquad
J_\lambda(\Pi)=\sum_i c_\lambda(\pi_i)=\mathrm{SoC}(\Pi)+\lambda R(\Pi),$$

where $\mathrm{SoC}$ is the classical sum of costs and $R(\Pi)=\sum_i\sum_{t\le T_i} r(t,\pi_i(t))$
is the plan's **risk**. By linearity of expectation, $R(\Pi)$ is the *expected number of
robot–human co-occupancy events* along the plan. By the union bound,
$P(\text{any encounter})\le R(\Pi)$. The costs are stored in integer fixed point
($S=1000$ units per step, `cost_scale`), so each step's cost is rounded by at most
$1/(2S)$.

## 2. Risk-aware CBS

`risk_cbs.hpp` (C++, used by `remroc_mapf_solver`) and `risk_cbs.py` (Python
reference) implement CBS (Sharon et al., 2015) with the costs above, $k$-robustness,
reservations and focal search.

* **Low level.** A* on the time-expanded graph (state $(t,v)$) under the node's
  constraints and the reservations. The heuristic is $h(v)=S\cdot d_{\text{BFS}}(v,g_i)$.
* **Splitting.** A vertex or robustness conflict "$i$ at $v$ at $t_1$ / $j$ at $v$ at
  $t_2$" creates the children $\{(i,v,t_1)\}$ and $\{(j,v,t_2)\}$. A swap conflict creates
  two edge constraints.
* **High level.** Best-first on $J_\lambda$, or focal search with factor $w\ge1$.

**Lemma 1 (sufficient search horizon).** Let $T_s=\max(T_c,T_\rho,H)$, where $T_c$ is
the largest constrained time and $T_\rho$ the last time a reservation moves. For every
optimal path there is one of equal cost that reaches its goal by $T_s+k+|V|+1$.

*Proof.* After $T_s$, all costs and constraints are time-invariant, and the only goal
restriction is arrival after $\max(T_c,T_\rho+k)\le T_s+k$. From its state at $T_s+k$
the agent then solves a static shortest-path problem with positive step costs. Any
optimal continuation is a simple path with at most $|V|-1$ moves, and waits cannot help
in a static environment. $\square$

The low-level search is therefore finite, and truncating it at that bound (`max_time`
in the code) loses no optimal solution.

**Lemma 2 (optimal low level).** The heuristic $h$ is admissible and consistent: every
action costs at least $S$, and $|h(u)-h(v)|\le S$ for neighbours. A* with a consistent
heuristic returns a minimum-cost path under the node's constraints. $\square$

**Theorem 1 (optimality and completeness, $w=1$).** If the instance, together with its
reservations, has a solution, risk-aware $k$-robust CBS returns a conflict-free joint
plan minimising $J_\lambda$.

*Proof.* (i) *Soundness of splitting.* Every valid solution violates at most one of the
two constraints of a split: it cannot contain both "$i$ at $v$ at $t_1$" and "$j$ at $v$
at $t_2$", because that is exactly the conflict. Hence every solution consistent with a
node is consistent with at least one of its children. (ii) *Lower bound.* By Lemma 2,
the cost of a node is a lower bound on the cost of every solution consistent with its
constraints. (iii) Best-first expansion therefore pops a conflict-free node only when
no open node has a lower cost, so the returned plan is optimal. (iv) *Termination.*
Step costs are integers of at least $S$, so a solvable instance has finitely many
distinct constraint-tree nodes of cost at most $C^\*$. The low level is finite by
Lemma 1. $\square$

For unsolvable instances CBS need not terminate, which is why every call carries a
time limit (`time_limit`) and the result reports `timeout` or `no_solution`. The start
state itself can become constrained by a robustness split. The low level then returns
"no path" for that branch; this was a real bug, found by
`test_risk_cbs.cpp::testRobustness`.

**Proposition 2 (bounded suboptimality, focal search).** With $w\ge1$ the high level
expands, among open nodes with $J\le w\cdot\mathrm{LB}$ (where
$\mathrm{LB}=\min_{\text{OPEN}}J$), the node with the fewest conflicting agent pairs.
The returned plan satisfies $J_\lambda(\Pi)\le w\,J_\lambda^\*$.

*Proof.* The minimum-cost open node lower-bounds the optimum (Theorem 1 (ii)), so
$\mathrm{LB}\le J^\*$. The returned node satisfies $J\le w\cdot\mathrm{LB}$. $\square$

This matters in practice. Once the risk map is dense (prediction blended with the
prior), costs differ by fractions of a step and optimal CBS explodes. On the 5-robot
experiment instances, $w=1$ times out after 10 s with more than $10^4$ expanded nodes,
while $w=1.1$ solves them in about 20 ms (`results/planner/planner.md`). Tests:
`test_focal_search_respects_suboptimality_bound`, and the C++/Python cross-validation
`test_cpp_and_python_agree_on_optimal_objective`, in which both independent
implementations must find the same optimal objective.

## 3. Risk vs. optimality

Let $\Pi^\*$ minimise $\mathrm{SoC}$ (that is, $\lambda=0$) and $\Pi_\lambda$ minimise
$J_\lambda$ over the same feasible set.

**Proposition 3 (trade-off).**

1. $0\le \mathrm{SoC}(\Pi_\lambda)-\mathrm{SoC}(\Pi^\*) \le \lambda\bigl(R(\Pi^\*)-R(\Pi_\lambda)\bigr)\le\lambda R(\Pi^\*)$.
2. For any plan $\Pi$ (in particular a risk-minimal one $\Pi^R$):
   $R(\Pi_\lambda)\le R(\Pi)+\bigl(\mathrm{SoC}(\Pi)-\mathrm{SoC}(\Pi_\lambda)\bigr)/\lambda$.
   Hence $R(\Pi_\lambda)\to R_{\min}$ as $\lambda\to\infty$.
3. If $\lambda_1<\lambda_2$, then $R(\Pi_{\lambda_2})\le R(\Pi_{\lambda_1})$ and
   $\mathrm{SoC}(\Pi_{\lambda_2})\ge \mathrm{SoC}(\Pi_{\lambda_1})$.

*Proof.* (1) and (2) rearrange $J_\lambda(\Pi_\lambda)\le J_\lambda(\Pi)$ for
$\Pi=\Pi^\*$ and $\Pi=\Pi^R$. For (3), add
$J_{\lambda_1}(\Pi_{\lambda_1})\le J_{\lambda_1}(\Pi_{\lambda_2})$ and
$J_{\lambda_2}(\Pi_{\lambda_2})\le J_{\lambda_2}(\Pi_{\lambda_1})$ to get
$(\lambda_2-\lambda_1)(R(\Pi_{\lambda_2})-R(\Pi_{\lambda_1}))\le0$. The SoC statement
follows by substituting back. $\square$

So $\lambda$ is an exchange rate: one extra step is spent only if it removes at least
$1/\lambda$ expected encounters. For example, $\lambda=4$ accepts a 2-step detour to
avoid a cell that a human occupies with probability at least 0.5. With a
$w$-suboptimal solver, (1) weakens to
$\mathrm{SoC}(\Pi_\lambda)-\mathrm{SoC}^\*\le (w-1)\mathrm{SoC}^\*+\lambda\bigl(wR(\Pi^\*)-R(\Pi_\lambda)\bigr)$,
and fixed-point rounding adds at most $\sum_i T_i/(2S)$.

Proposition 3 is about **plans under the predicted risk**. At execution time the
realised benefit also depends on prediction quality. With walking humans and roughly
3 s MAPF steps, tracking-based forecasts beat an all-zero forecast for only a few steps
(`results/prediction/brier.md`). This is why the risk map blends the prediction into a
learnt static prior, and why the measured execution-level trade-off
(`results/sim_lambda`) is noisy and only approximately monotone. Test:
`test_risk_optimality_tradeoff_is_monotone`.

Waiting at the goal after arrival does not count towards $c_\lambda$ (standard SoC
convention). Exposure of robots parked at their goals is therefore not optimised.

## 4. Execution: Action Dependency Graph

`adg.py` implements the ADG of Hönig et al. (2019). Each robot's path is compressed
into its move actions $a=(i,\text{src},\text{dst},t_a)$, with waits dropped.

* **Type-1 edges** link consecutive actions of the same robot.
* **Type-2 edges** run $a\to b$ whenever $a$ (robot $i$) leaves cell $v$ at $t_a$ and $b$
  (robot $j\ne i$) enters $v$ at $t_b\ge t_a$.

Execution rule: a robot may start action $b$ only after all its predecessors are
*completed*, meaning the robot is in $\text{dst}$ and has left $\text{src}$.

**Lemma 4 (acyclicity).** If the plan is conflict-free and 1-robust, every ADG edge
strictly increases plan time, so the ADG is acyclic.

*Proof.* Type-1 edges increase time by construction. For a type-2 edge, $i$ is in $v$ at
$t_a$ and $j$ is in $v$ at $t_b+1$. 1-robustness forbids $|t_a-(t_b+1)|\le1$, and
$t_b\ge t_a$, so $t_b+1\ge t_a+2$, that is, $t_b>t_a$. $\square$

For $k=0$ this fails. Four robots rotating in a 2×2 block form a valid classical plan
whose ADG is a cycle, so executing it would deadlock at once (`CyclicADGError`, test
`test_rotation_plan_gives_cyclic_adg`). This is why the coordinator plans with $k=1$.

**Theorem 5 (safety for any timing).** Execute an ADG built from a conflict-free plan
with any action durations and delays. Then no two robots are ever in the same cell,
and no two robots swap cells.

*Proof.* In a conflict-free plan the occupation intervals of each cell $v$ by different
robots are disjoint, and hence totally ordered. For consecutive occupants $i$ then $j$,
$j$'s action entering $v$ depends on $i$'s action leaving $v$. So $j$ starts entering
only after $i$ has completely left. A robot that occupies $v$ at the start and never
leaves cannot have a successor, because the plan is conflict-free. A swap would need
each robot's entering action to depend on the other's leaving action, which is a
2-cycle excluded by Lemma 4. $\square$

**Theorem 6 (deadlock-freedom and completeness under bounded delays).** Suppose the ADG
is acyclic and every *enabled* action completes within bounded time. Formally, a
dispatched action finishes within $\delta+\tau+D$, where $\delta$ is dispatch latency
(one coordinator period), $\tau$ the nominal duration, and $D$ an upper bound on any
stall or on blocking by humans. Then every robot reaches its goal by time
$L\cdot(\delta+\tau+D)$, where $L$ is the number of actions on the longest path of the
ADG, and no deadlock occurs.

*Proof.* Suppose some action never completes. Pick the first such action $b$ in a
topological order. All its predecessors complete, so $b$ becomes enabled and, by
assumption, completes. Contradiction. For the bound, use induction along a topological
order. An action whose predecessors have all completed by $T$ is dispatched by
$T+\delta$ and completes by $T+\delta+\tau+D$. $\square$

The bound is linear in the dependency depth $L\le\sum_i|\pi_i|$, not in the number of
robots times the delay. A delay of one robot propagates only along ADG paths. The
dispatcher additionally never sends a robot into a cell that another robot physically
occupies (`guard_radius_frac`). In a consistent execution such an occupant is always
about to leave (it is completing an action, or it is a robot that was re-planned while
mid-move), so the guard adds at most one action duration per wait and preserves
Theorem 6.

**What the assumptions mean on real robots.** "Bounded delay" is violated if a human
blocks a cell forever. The ADG then waits, which is safe but not live. This is exactly
what local repair (§5) addresses: a robot blocked for `blocked_timeout` seconds is
re-planned around the obstruction. Theorems 5 and 6 are properties of the
coordination layer. The local planner (nav2/MPPI) must still follow the grid
approximately. The stress test in `EXPERIMENTS.md` removes robot–robot avoidance
from the local layer entirely and heavily delays robots. The ADG variants stay
collision-free and complete, while the iterative-MAPF baseline collides in 60–80% of
the episodes.

## 5. Local plan repair

Trigger: a robot is *blocked* (dispatched, stopped, no progress for `blocked_timeout`),
or its next `risk_lookahead` cells have predicted risk above `risk_trigger`. The repair
(`coordination.py::repair`) proceeds as follows.

1. Choose a set $S$ of robots to re-plan, escalating through: the robot; the robot plus
   the idle robots coupled to it by ADG edges; all idle robots. Robots in the middle of
   a move are never re-planned.
2. Compute the **earliest-start (ES) schedule** of the remaining actions of the other
   robots, keeping only dependencies among themselves. Their entry into cells
   currently held by $S$ is delayed by `repair_slack`.
3. Plan $S$ with the risk-aware CBS using the ES paths as reservations.
4. Validate the joint plan (conflict-free, 1-robust) and rebuild the ADG from it. Reject
   the repair if it would revoke actions already dispatched to a robot that keeps its
   plan. A risk-triggered repair is accepted only if it lowers $J_\lambda$ of $S$ by at
   least `min_improvement` steps (hysteresis against chasing prediction noise).

**Lemma 7 (ES schedules are valid reservations).** For any set $Q$ of robots and any
progress state reachable under the ADG, the ES schedule of $Q$ is conflict-free and
1-robust.

*Proof.* Let $i,j\in Q$ visit a cell $v$ consecutively in the original plan. The ADG
edge from $i$'s leaving action to $j$'s entering action is kept, since both robots are
in $Q$. So $ES(b)\ge ES(a)+1$: $i$ is in $v$ until $ES(a)$, and $j$ arrives at
$ES(b)+1\ge ES(a)+2$. That is the 1-robust separation. Dropping actions of robots
outside $Q$ only removes constraints on them. Completed actions impose nothing, and
the progress state is consistent with the dependencies. Swaps are excluded as in
Theorem 5. $\square$

(Test: `test_earliest_start_schedule_is_conflict_free`, on random partial executions.)

**Theorem 8 (repairs keep the guarantees).** Every installed plan is a validated
conflict-free, 1-robust joint plan that starts at the robots' current cells. Its ADG is
acyclic by Lemma 4, so Theorems 5 and 6 hold between repairs. Dispatched actions of
robots that keep their plan remain enabled, so no robot is inside a cell that the new
plan assigns to another robot. With a finite repair budget (`max_repairs`, default
200; the cooldown also limits each robot to one repair per `repair_cooldown` seconds),
only finitely many plan switches occur. After the last one, Theorem 6 applies, so the
fleet completes under bounded delays. $\square$

## 6. Summary of assumptions and limitations

* Grid abstraction: the local planner must stay close to cell centroids. REMROC's nav2
  stack only approximately does so, and the paper's plan–execution gap is exactly this.
  The guarantees concern the coordination layer.
* Humans are observed globally, as by infrastructure sensors; the ROS node reads them
  from the Gazebo world. With partial observability the tracker simply loses them after
  `forget_s`, and the static prior remains.
* Risk costs model *expected co-occupancy*, not collision probability under
  interaction. Humans are assumed not to react to robots, like REMROC's scripted
  actors.
* Optimal risk-aware CBS is only practical for sparse risk. The system uses focal
  search with $w=1.1$, so plans are within 10% of the optimum of $J_\lambda$.
* Completeness needs bounded delays. Indefinite human blockage is handled by repair,
  heuristically and without a guarantee.

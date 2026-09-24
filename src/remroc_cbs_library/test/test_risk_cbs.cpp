// Copyright (c) 2026 Collins Masimba
// SPDX-License-Identifier: Apache-2.0
//
// Dependency-free unit tests for risk_cbs.hpp (returns non-zero on failure).

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "remroc_cbs_library/risk_cbs.hpp"

namespace rc = remroc_cbs_library::risk_cbs;

static int g_failures = 0;

#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::cerr << __FILE__ << ":" << __LINE__ << " CHECK failed: " #cond << std::endl; \
      ++g_failures; \
    } \
  } while (0)

static rc::Cell at(const std::vector<rc::Cell> & p, size_t t) {return t < p.size() ? p[t] : p.back();}

// Independent validation of a joint solution (vertex + swap conflicts, reservations, grid).
static bool validJointPlan(
  const rc::GridMap & g, const std::vector<rc::Agent> & agents,
  const std::vector<std::vector<rc::Cell>> & paths,
  const std::vector<rc::Reservation> & res = {})
{
  size_t T = 0;
  for (const auto & p : paths) {T = std::max(T, p.size());}
  for (const auto & r : res) {T = std::max(T, r.path.size());}
  for (size_t i = 0; i < paths.size(); ++i) {
    if (paths[i].empty() || paths[i].front() != agents[i].start || paths[i].back() != agents[i].goal) {return false;}
    for (size_t t = 0; t < paths[i].size(); ++t) {
      if (!g.isFree(paths[i][t].x, paths[i][t].y)) {return false;}
      if (t > 0 && std::abs(paths[i][t].x - paths[i][t - 1].x) + std::abs(paths[i][t].y - paths[i][t - 1].y) > 1) {return false;}
    }
  }
  std::vector<std::vector<rc::Cell>> all = paths;
  for (const auto & r : res) {all.push_back(r.path);}
  for (size_t t = 0; t <= T; ++t) {
    for (size_t i = 0; i < all.size(); ++i) {
      for (size_t j = i + 1; j < all.size(); ++j) {
        if (j >= paths.size() && i >= paths.size()) {continue;}   // reservation vs reservation
        if (at(all[i], t) == at(all[j], t)) {return false;}
        if (at(all[i], t) != at(all[i], t + 1) && at(all[i], t) == at(all[j], t + 1) &&
          at(all[i], t + 1) == at(all[j], t)) {return false;}
      }
    }
  }
  return true;
}

static void testSingleAgentStraightLine()
{
  rc::GridMap g(5, 1);
  std::vector<rc::Agent> a{{{0, 0}, {4, 0}}};
  auto r = rc::solve(g, a, rc::RiskMap(), {}, rc::Options());
  CHECK(r.success);
  CHECK(r.sum_of_steps == 4);
  CHECK(r.objective_scaled == 4000);
  CHECK(validJointPlan(g, a, r.paths));
}

// Two agents swapping ends of a corridor with a single side pocket: one must yield.
static void testCorridorSwapWithPocket()
{
  rc::GridMap g(5, 2);
  for (int x = 0; x < 5; ++x) {if (x != 2) {g.setObstacle(x, 1);}}
  std::vector<rc::Agent> a{{{0, 0}, {4, 0}}, {{4, 0}, {0, 0}}};
  auto r = rc::solve(g, a, rc::RiskMap(), {}, rc::Options());
  CHECK(r.success);
  CHECK(validJointPlan(g, a, r.paths));
  // Optimal: one agent dodges into the pocket (2 extra moves) and, because the dodger
  // occupies the pocket entrance at t=2, one of them also waits once: 4 + 4 + 2 + 1 = 11.
  CHECK(r.sum_of_steps == 11);
}

static void testNoSolutionIsReported()
{
  rc::GridMap g(3, 1);   // pure corridor, swap impossible
  std::vector<rc::Agent> a{{{0, 0}, {2, 0}}, {{2, 0}, {0, 0}}};
  rc::Options o;
  o.time_limit_s = 2.0;
  auto r = rc::solve(g, a, rc::RiskMap(), {}, o);
  CHECK(!r.success);
}

static void testInvalidInputs()
{
  rc::GridMap g(3, 3);
  g.setObstacle(1, 1);
  CHECK(rc::solve(g, {{{1, 1}, {0, 0}}}, rc::RiskMap(), {}, rc::Options()).status == "invalid_start");
  CHECK(rc::solve(g, {{{0, 0}, {2, 2}}, {{2, 0}, {2, 2}}}, rc::RiskMap(), {}, rc::Options()).status == "duplicate_goal");
}

// Risk makes the agent take a longer route around a predicted human.
static void testRiskDetour()
{
  // 5x3 open grid, agent goes (0,1)->(4,1). The middle row is risky at every time.
  rc::GridMap g(5, 3);
  std::vector<float> data(5 * 3, 0.0F);
  data[1 * 5 + 2] = 1.0F;   // cell (2,1)
  rc::RiskMap risk(5, 3, 1, data);
  std::vector<rc::Agent> a{{{0, 1}, {4, 1}}};

  rc::Options plain;
  auto r0 = rc::solve(g, a, risk, {}, plain);
  CHECK(r0.success && r0.sum_of_steps == 4);

  rc::Options aware;
  aware.risk_weight = 5.0;
  auto r1 = rc::solve(g, a, risk, {}, aware);
  CHECK(r1.success);
  CHECK(r1.sum_of_steps == 6);           // detour through row 0 or 2
  CHECK(r1.total_risk == 0.0);
  CHECK(r1.objective_scaled == 6000);

  // Small lambda: cheaper to accept the risk (4 + 0.5 < 6).
  aware.risk_weight = 0.5;
  auto r2 = rc::solve(g, a, risk, {}, aware);
  CHECK(r2.success && r2.sum_of_steps == 4);
  CHECK(r2.objective_scaled == 4500);
}

// Time-varying risk: waiting for the human to pass is optimal.
static void testWaitForHumanToPass()
{
  rc::GridMap g(3, 1);
  std::vector<float> data(3 * 1 * 4, 0.0F);
  // human in the middle cell for t = 0..2, gone at t = 3
  for (int t = 0; t < 3; ++t) {data[t * 3 + 1] = 1.0F;}
  rc::RiskMap risk(3, 1, 4, data);
  rc::Options o;
  o.risk_weight = 10.0;
  auto r = rc::solve(g, {{{0, 0}, {2, 0}}}, risk, {}, o);
  CHECK(r.success);
  CHECK(r.sum_of_steps == 4);             // wait twice, then go (arrive at middle at t=3)
  CHECK(r.paths[0][3].x == 1);
  CHECK(r.total_risk == 0.0);
}

// Reservations are respected as hard constraints and the solution is conflict free.
static void testReservations()
{
  rc::GridMap g(5, 2);
  for (int x = 0; x < 5; ++x) {if (x != 2) {g.setObstacle(x, 1);}}
  // A reserved robot traverses the corridor from right to left, then parks at (0,0)... but
  // our agent starts at (0,0); instead park it at (1,0) to force the agent to use the pocket.
  rc::Reservation res;
  res.path = {{4, 0}, {3, 0}, {2, 0}, {1, 0}};
  std::vector<rc::Agent> a{{{0, 0}, {4, 0}}};
  auto r = rc::solve(g, a, rc::RiskMap(), {res}, rc::Options());
  CHECK(!r.success);   // (1,0) is parked forever, the agent is walled in

  res.path = {{4, 0}, {3, 0}, {2, 0}, {2, 1}};   // reserved robot parks in the pocket
  r = rc::solve(g, a, rc::RiskMap(), {res}, rc::Options());
  CHECK(r.success);
  CHECK(validJointPlan(g, a, r.paths, {res}));

  // Goal visited by a reserved robot later: the agent must arrive after it passed.
  rc::GridMap open(4, 1);
  rc::Reservation pass;
  pass.path = {{3, 0}, {2, 0}, {3, 0}};   // leaves (3,0), comes back and parks... goal blocked
  r = rc::solve(open, {{{0, 0}, {2, 0}}}, rc::RiskMap(), {pass}, rc::Options());
  CHECK(r.success);
  CHECK(validJointPlan(open, {{{0, 0}, {2, 0}}}, r.paths, {pass}));
  CHECK(r.paths[0].size() >= 3);
}

// Randomised: every returned plan is valid and lambda = 0 cost equals plain CBS cost.
static void testRandomInstancesValid()
{
  std::srand(7);
  int solved = 0;
  for (int it = 0; it < 60; ++it) {
    const int W = 6 + std::rand() % 4;
    const int H = 4 + std::rand() % 3;
    rc::GridMap g(W, H);
    for (int k = 0; k < W * H / 6; ++k) {g.setObstacle(std::rand() % W, std::rand() % H);}
    std::vector<rc::Cell> free;
    for (int y = 0; y < H; ++y) {for (int x = 0; x < W; ++x) {if (g.isFree(x, y)) {free.push_back({x, y});}}}
    const int n = 3 + std::rand() % 2;
    if (static_cast<int>(free.size()) < 2 * n) {continue;}
    for (size_t i = free.size() - 1; i > 0; --i) {std::swap(free[i], free[std::rand() % (i + 1)]);}
    std::vector<rc::Agent> agents;
    for (int i = 0; i < n; ++i) {agents.push_back({free[i], free[n + i]});}
    const int T = 8;
    std::vector<float> data(static_cast<size_t>(T) * W * H);
    for (auto & v : data) {v = (std::rand() % 100 < 15) ? static_cast<float>(std::rand() % 100) / 100.0F : 0.0F;}
    rc::RiskMap risk(W, H, T, data);
    rc::Options o;
    o.risk_weight = static_cast<double>(std::rand() % 4);
    o.time_limit_s = 2.0;
    auto r = rc::solve(g, agents, risk, {}, o);
    if (r.success) {
      ++solved;
      CHECK(validJointPlan(g, agents, r.paths));
      // risk-weighted objective is never below the risk-free optimum of the same instance
      rc::Options o0 = o;
      o0.risk_weight = 0.0;
      auto r0 = rc::solve(g, agents, risk, {}, o0);
      if (r0.success) {
        CHECK(r.objective_scaled >= r0.objective_scaled);
        CHECK(r.sum_of_steps >= r0.sum_of_steps);
        CHECK(r.total_risk <= r0.total_risk + 1e-6);
      }
    }
  }
  CHECK(solved > 30);
}

// k = 1 forbids following and rotations (needed for an acyclic ADG).
static void testRobustness()
{
  // Two agents in a row both moving right: k=0 moves them together, k=1 staggers them.
  rc::GridMap line(3, 1);
  std::vector<rc::Agent> a{{{0, 0}, {1, 0}}, {{1, 0}, {2, 0}}};
  auto r0 = rc::solve(line, a, rc::RiskMap(), {}, rc::Options());
  CHECK(r0.success && r0.sum_of_steps == 2);
  rc::Options k1;
  k1.robustness = 1;
  auto r1 = rc::solve(line, a, rc::RiskMap(), {}, k1);
  CHECK(r1.success && r1.sum_of_steps == 3);
  CHECK(r1.paths[0][1].x == 0);   // the rear agent waits one step

  // Four agents rotating in a closed 2x2 block: feasible only without robustness.
  rc::GridMap box(2, 2);
  std::vector<rc::Agent> rot{{{0, 0}, {1, 0}}, {{1, 0}, {1, 1}}, {{1, 1}, {0, 1}}, {{0, 1}, {0, 0}}};
  auto rr0 = rc::solve(box, rot, rc::RiskMap(), {}, rc::Options());
  CHECK(rr0.success && rr0.sum_of_steps == 4);
  k1.time_limit_s = 0.5;
  auto rr1 = rc::solve(box, rot, rc::RiskMap(), {}, k1);
  CHECK(!rr1.success);

  // Robust planning against a reservation: must not enter a cell right after the
  // reserved robot leaves it.
  rc::GridMap open(4, 1);
  rc::Reservation res;
  res.path = {{2, 0}, {3, 0}};   // leaves (2,0) at t=1 and parks at (3,0)
  k1.time_limit_s = 0.0;
  auto rr = rc::solve(open, {{{1, 0}, {2, 0}}}, rc::RiskMap(), {res}, k1);
  CHECK(rr.success);
  CHECK(rr.paths[0].size() == 3);   // wait at t=1, enter at t=2
}

int main()
{
  testRobustness();
  testSingleAgentStraightLine();
  testCorridorSwapWithPocket();
  testNoSolutionIsReported();
  testInvalidInputs();
  testRiskDetour();
  testWaitForHumanToPass();
  testReservations();
  testRandomInstancesValid();
  if (g_failures == 0) {
    std::cout << "risk_cbs tests: all passed" << std::endl;
    return 0;
  }
  std::cout << "risk_cbs tests: " << g_failures << " failure(s)" << std::endl;
  return 1;
}

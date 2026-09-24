// Copyright (c) 2026 Collins Masimba
// SPDX-License-Identifier: Apache-2.0
//
// Risk-aware ("probabilistic") Conflict-Based Search.
//
// This header is self-contained (standard library only, no boost / yaml-cpp),
// so it can be unit-tested and used from a command line tool without ROS.
//
// Model
// -----
// * 4-connected grid, unit-time actions {wait, up, down, left, right}.
// * A time-indexed risk map r(t, x, y) in [0, 1] gives the predicted probability
//   that a human occupies cell (x, y) at MAPF step t. Layers beyond the horizon
//   clamp to the last layer (which a coordinator typically sets to a static prior).
// * The cost of arriving in cell v at time t is 1 + lambda * r(t, v). Costs are
//   stored in integer fixed point (cost_scale units per step) so that the search
//   is exact and reproducible across implementations.
// * Reservations are timed paths of robots that are *not* re-planned (used for
//   local plan repair). A reserved robot occupies path[t] at time t and parks at
//   path.back() forever. They act as hard constraints for every planned agent.
// * k-robustness (Atzmon et al., 2018): with robustness k >= 1 no agent may occupy a
//   cell within k steps of another agent. k = 1 forbids "following" and hence also
//   rotation cycles, which makes every Action Dependency Graph built from the plan
//   acyclic (see docs/THEORY.md). k = 0 is classical CBS.
//
// With suboptimality w = 1, CBS is optimal w.r.t. the sum over agents of the (risk-weighted) path costs,
// because the low level (A* on the time-expanded graph with an admissible and
// consistent heuristic) is optimal for non-negative edge costs; see
// docs/THEORY.md for the full argument.

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <tuple>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace remroc_cbs_library
{
namespace risk_cbs
{

using Cost = std::int64_t;
constexpr Cost kInfCost = std::numeric_limits<Cost>::max() / 4;

struct Cell
{
  int x = 0;
  int y = 0;
  bool operator==(const Cell & o) const {return x == o.x && y == o.y;}
  bool operator!=(const Cell & o) const {return !(*this == o);}
};

class GridMap
{
public:
  GridMap() = default;
  GridMap(int dimx, int dimy)
  : dimx_(dimx), dimy_(dimy), blocked_(static_cast<size_t>(dimx) * dimy, 0) {}

  int dimx() const {return dimx_;}
  int dimy() const {return dimy_;}
  int numCells() const {return dimx_ * dimy_;}
  bool inBounds(int x, int y) const {return x >= 0 && y >= 0 && x < dimx_ && y < dimy_;}
  int index(int x, int y) const {return y * dimx_ + x;}
  int index(const Cell & c) const {return index(c.x, c.y);}
  Cell cell(int idx) const {return Cell{idx % dimx_, idx / dimx_};}
  void setObstacle(int x, int y)
  {
    if (inBounds(x, y)) {blocked_[index(x, y)] = 1;}
  }
  bool isFree(int x, int y) const {return inBounds(x, y) && !blocked_[index(x, y)];}
  int numFreeCells() const
  {
    return static_cast<int>(std::count(blocked_.begin(), blocked_.end(), 0));
  }

private:
  int dimx_ = 0;
  int dimy_ = 0;
  std::vector<std::uint8_t> blocked_;
};

// Time-indexed human occupancy probability. data index: (t * dimy + y) * dimx + x.
class RiskMap
{
public:
  RiskMap() = default;
  RiskMap(int dimx, int dimy, int horizon, std::vector<float> data)
  : dimx_(dimx), dimy_(dimy), horizon_(horizon), data_(std::move(data))
  {
    if (dimx_ <= 0 || dimy_ <= 0 || horizon_ <= 0 ||
      data_.size() != static_cast<size_t>(dimx_) * dimy_ * horizon_)
    {
      dimx_ = dimy_ = horizon_ = 0;
      data_.clear();
    }
  }
  bool empty() const {return data_.empty();}
  int horizon() const {return horizon_;}
  int dimx() const {return dimx_;}
  int dimy() const {return dimy_;}
  float at(int t, int x, int y) const
  {
    if (data_.empty()) {return 0.0F;}
    t = std::max(0, std::min(t, horizon_ - 1));
    const float v = data_[(static_cast<size_t>(t) * dimy_ + y) * dimx_ + x];
    return std::max(0.0F, std::min(1.0F, v));
  }

private:
  int dimx_ = 0;
  int dimy_ = 0;
  int horizon_ = 0;
  std::vector<float> data_;
};

struct Agent
{
  Cell start;
  Cell goal;
};

struct Reservation
{
  // path[t] is occupied at time t; path.back() stays occupied for all later t.
  std::vector<Cell> path;
};

struct Options
{
  double risk_weight = 0.0;           // lambda
  Cost cost_scale = 1000;             // fixed-point units per unit-time step
  double time_limit_s = 0.0;          // <= 0: unlimited
  int robustness = 0;                 // k (0 = classical CBS)
  // w >= 1: bounded-suboptimal high level (focal search, as in ECBS/BCBS): among the
  // constraint-tree nodes with cost <= w * (minimum open cost) the one with the fewest
  // conflicting agent pairs is expanded. The solution cost is <= w * optimal. w = 1 is
  // optimal CBS (ties broken by fewest conflicts).
  double suboptimality = 1.0;
  std::size_t max_high_level_nodes = 200000;
};

struct Result
{
  bool success = false;
  std::string status = "not_run";
  std::vector<std::vector<Cell>> paths;   // per agent, index = time, ends at goal arrival
  Cost objective_scaled = 0;              // sum of (scaled) path costs
  double objective = 0.0;                 // objective_scaled / cost_scale
  int sum_of_steps = 0;                   // classical sum of costs (unit steps)
  int makespan = 0;
  double total_risk = 0.0;                // sum over agents/time of r(t, path[t]), t >= 1
  std::size_t high_level_expanded = 0;
  std::size_t low_level_expanded = 0;
  double runtime_s = 0.0;
};

namespace detail
{

inline std::uint64_t vertexKey(int t, int cell)
{
  return (static_cast<std::uint64_t>(t) << 32) | static_cast<std::uint32_t>(cell);
}

// requires cell indices < 2^20 and t < 2^24
inline std::uint64_t edgeKey(int t, int from, int to)
{
  return (static_cast<std::uint64_t>(t) << 40) |
         (static_cast<std::uint64_t>(from) << 20) | static_cast<std::uint64_t>(to);
}

struct Constraints
{
  std::unordered_set<std::uint64_t> vertex;
  std::unordered_set<std::uint64_t> edge;
  // Cached for the goal test and the search horizon.
  std::unordered_map<int, int> last_vertex_time_per_cell;
  int max_time = 0;

  void addVertex(int t, int cell)
  {
    vertex.insert(vertexKey(t, cell));
    auto it = last_vertex_time_per_cell.find(cell);
    if (it == last_vertex_time_per_cell.end()) {
      last_vertex_time_per_cell[cell] = t;
    } else {
      it->second = std::max(it->second, t);
    }
    max_time = std::max(max_time, t);
  }
  void addEdge(int t, int from, int to)
  {
    edge.insert(edgeKey(t, from, to));
    max_time = std::max(max_time, t + 1);
  }
  int lastVertexTime(int cell) const
  {
    auto it = last_vertex_time_per_cell.find(cell);
    return it == last_vertex_time_per_cell.end() ? -1 : it->second;
  }
};

class ReservationTable
{
public:
  ReservationTable(const GridMap & grid, const std::vector<Reservation> & reservations, int k)
  : k_(std::max(0, k)), parked_from_(grid.numCells(), std::numeric_limits<int>::max()),
    last_dynamic_(grid.numCells(), -1)
  {
    for (const auto & r : reservations) {
      if (r.path.empty()) {continue;}
      const int len = static_cast<int>(r.path.size());
      for (int t = 0; t < len; ++t) {
        const int c = grid.index(r.path[t]);
        if (t < len - 1) {
          vertex_.insert(vertexKey(t, c));
          last_dynamic_[c] = std::max(last_dynamic_[c], t);
          const int n = grid.index(r.path[t + 1]);
          if (n != c) {edge_.insert(edgeKey(t, c, n));}
        } else {
          parked_from_[c] = std::min(parked_from_[c], t);
        }
      }
      horizon_ = std::max(horizon_, len - 1);
    }
  }

  // Occupied by a reserved robot at any time in [t - k, t + k].
  bool vertexBlocked(int t, int cell) const
  {
    if (t + k_ >= parked_from_[cell]) {return true;}
    for (int tt = std::max(0, t - k_); tt <= t + k_; ++tt) {
      if (vertex_.count(vertexKey(tt, cell)) > 0) {return true;}
    }
    return false;
  }
  // A planned agent moving from -> to during [t, t+1] swaps with a reserved robot
  // moving to -> from during the same interval.
  bool edgeBlocked(int t, int from, int to) const
  {
    return from != to && edge_.count(edgeKey(t, to, from)) > 0;
  }
  bool parkedForever(int cell) const {return parked_from_[cell] != std::numeric_limits<int>::max();}
  int lastDynamic(int cell) const {return last_dynamic_[cell];}
  int horizon() const {return horizon_;}

private:
  int k_;
  std::unordered_set<std::uint64_t> vertex_;
  std::unordered_set<std::uint64_t> edge_;
  std::vector<int> parked_from_;
  std::vector<int> last_dynamic_;
  int horizon_ = 0;
};

class Deadline
{
public:
  explicit Deadline(double seconds)
  : enabled_(seconds > 0.0),
    end_(std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(seconds > 0.0 ? seconds : 0.0))) {}
  bool expired() const {return enabled_ && std::chrono::steady_clock::now() >= end_;}

private:
  bool enabled_;
  std::chrono::steady_clock::time_point end_;
};

struct TimeoutError {};

// Static shortest-path distances to a goal (BFS on the obstacle map).
inline std::vector<int> bfsDistances(const GridMap & grid, const Cell & goal)
{
  std::vector<int> dist(grid.numCells(), -1);
  if (!grid.isFree(goal.x, goal.y)) {return dist;}
  std::deque<int> q;
  dist[grid.index(goal)] = 0;
  q.push_back(grid.index(goal));
  const int dx[4] = {1, -1, 0, 0};
  const int dy[4] = {0, 0, 1, -1};
  while (!q.empty()) {
    const int c = q.front();
    q.pop_front();
    const Cell cc = grid.cell(c);
    for (int k = 0; k < 4; ++k) {
      const int nx = cc.x + dx[k];
      const int ny = cc.y + dy[k];
      if (!grid.isFree(nx, ny)) {continue;}
      const int n = grid.index(nx, ny);
      if (dist[n] >= 0) {continue;}
      dist[n] = dist[c] + 1;
      q.push_back(n);
    }
  }
  return dist;
}

class Solver
{
public:
  Solver(
    const GridMap & grid, const std::vector<Agent> & agents, const RiskMap & risk,
    const std::vector<Reservation> & reservations, const Options & options)
  : grid_(grid), agents_(agents), risk_(risk),
    reservations_(grid, reservations, options.robustness),
    options_(options), deadline_(options.time_limit_s)
  {
    heuristics_.reserve(agents_.size());
    for (const auto & a : agents_) {
      heuristics_.push_back(bfsDistances(grid_, a.goal));
    }
    static_horizon_ = std::max(reservations_.horizon(), risk_.empty() ? 0 : risk_.horizon());
  }

  Cost stepCost(int t_arrive, int cell) const
  {
    if (risk_.empty() || options_.risk_weight <= 0.0) {return options_.cost_scale;}
    const Cell c = grid_.cell(cell);
    const double r = risk_.at(t_arrive, c.x, c.y);
    return options_.cost_scale +
           static_cast<Cost>(std::llround(
             static_cast<double>(options_.cost_scale) * options_.risk_weight * r));
  }

  // Low-level: optimal single-agent path under constraints and reservations.
  // Returns false if no path exists within the (provably sufficient) time bound.
  bool lowLevel(std::size_t agent, const Constraints & cons, std::vector<Cell> & path, Cost & cost)
  {
    const Cell start = agents_[agent].start;
    const Cell goal = agents_[agent].goal;
    const int start_idx = grid_.index(start);
    const int goal_idx = grid_.index(goal);
    const std::vector<int> & h = heuristics_[agent];
    if (h[start_idx] < 0) {return false;}
    if (reservations_.parkedForever(goal_idx)) {return false;}
    // Robustness conflicts can constrain the start state itself (agent must not be at
    // its start at t = 0): that branch of the constraint tree is infeasible.
    if (cons.vertex.count(vertexKey(0, start_idx))) {return false;}

    const int last_res = reservations_.lastDynamic(goal_idx);
    const int last_goal_block = std::max(
      cons.lastVertexTime(goal_idx), last_res < 0 ? -1 : last_res + std::max(0, options_.robustness));
    // After max(constraints, reservations, risk horizon) the world is static, so an
    // optimal path needs at most |free cells| further steps (cf. docs/THEORY.md).
    const int max_time = std::max(cons.max_time, static_horizon_) + std::max(0, options_.robustness) +
      grid_.numFreeCells() + 1;

    struct Node
    {
      int t;
      int cell;
      Cost g;
      int parent;
    };
    struct OpenEntry
    {
      Cost f;
      Cost g;
      int node;
    };
    struct OpenCmp
    {
      bool operator()(const OpenEntry & a, const OpenEntry & b) const
      {
        if (a.f != b.f) {return a.f > b.f;}
        if (a.g != b.g) {return a.g < b.g;}   // prefer deeper nodes on ties
        return a.node > b.node;
      }
    };

    std::vector<Node> nodes;
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenCmp> open;
    std::unordered_map<std::uint64_t, Cost> best_g;

    nodes.push_back(Node{0, start_idx, 0, -1});
    best_g[vertexKey(0, start_idx)] = 0;
    open.push(OpenEntry{options_.cost_scale * h[start_idx], 0, 0});

    const int dx[5] = {0, 1, -1, 0, 0};
    const int dy[5] = {0, 0, 0, 1, -1};
    std::size_t local_expansions = 0;

    while (!open.empty()) {
      const OpenEntry top = open.top();
      open.pop();
      const Node cur = nodes[top.node];
      auto it = best_g.find(vertexKey(cur.t, cur.cell));
      if (it != best_g.end() && it->second < cur.g) {continue;}   // stale entry

      ++low_level_expanded_;
      if ((++local_expansions & 1023U) == 0U && deadline_.expired()) {throw TimeoutError{};}

      if (cur.cell == goal_idx && cur.t > last_goal_block) {
        path.clear();
        for (int n = top.node; n >= 0; n = nodes[n].parent) {
          path.push_back(grid_.cell(nodes[n].cell));
        }
        std::reverse(path.begin(), path.end());
        cost = cur.g;
        return true;
      }
      if (cur.t >= max_time) {continue;}

      const Cell cc = grid_.cell(cur.cell);
      const int nt = cur.t + 1;
      for (int k = 0; k < 5; ++k) {
        const int nx = cc.x + dx[k];
        const int ny = cc.y + dy[k];
        if (!grid_.isFree(nx, ny)) {continue;}
        const int n = grid_.index(nx, ny);
        if (h[n] < 0) {continue;}
        if (cons.vertex.count(vertexKey(nt, n))) {continue;}
        if (cons.edge.count(edgeKey(cur.t, cur.cell, n))) {continue;}
        if (reservations_.vertexBlocked(nt, n)) {continue;}
        if (reservations_.edgeBlocked(cur.t, cur.cell, n)) {continue;}
        const Cost g = cur.g + stepCost(nt, n);
        const std::uint64_t key = vertexKey(nt, n);
        auto bit = best_g.find(key);
        if (bit != best_g.end() && bit->second <= g) {continue;}
        best_g[key] = g;
        nodes.push_back(Node{nt, n, g, top.node});
        open.push(OpenEntry{g + options_.cost_scale * h[n], g, static_cast<int>(nodes.size()) - 1});
      }
    }
    return false;
  }

  Result solve()
  {
    const auto t0 = std::chrono::steady_clock::now();
    Result res;
    try {
      res = solveImpl();
    } catch (const TimeoutError &) {
      res = Result{};
      res.status = "timeout";
    }
    res.high_level_expanded = high_level_expanded_;
    res.low_level_expanded = low_level_expanded_;
    res.runtime_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return res;
  }

private:
  struct HighLevelNode
  {
    std::vector<std::vector<Cell>> paths;
    std::vector<Cost> costs;
    std::vector<Constraints> constraints;
    Cost total = 0;
    int conflicts = 0;
  };

  // vertex == true: agent a1 must not be in cell c1 at t1, or agent a2 not in c1 at t2
  // (t1 == t2 for classical vertex conflicts, |t1 - t2| <= k for robustness conflicts).
  // vertex == false: swap conflict on edge c1 -> c2 during [t1, t1 + 1].
  struct Conflict
  {
    bool vertex = true;
    int t1 = 0;
    int t2 = 0;
    std::size_t a1 = 0;
    std::size_t a2 = 0;
    int c1 = 0;
    int c2 = 0;
  };

  static const Cell & posAt(const std::vector<Cell> & p, int t)
  {
    return t < static_cast<int>(p.size()) ? p[t] : p.back();
  }

  bool firstConflict(const std::vector<std::vector<Cell>> & paths, Conflict & out) const
  {
    int max_t = 0;
    for (const auto & p : paths) {max_t = std::max(max_t, static_cast<int>(p.size()) - 1);}
    const int k = std::max(0, options_.robustness);
    for (int t = 0; t <= max_t; ++t) {
      for (std::size_t i = 0; i < paths.size(); ++i) {
        for (std::size_t j = i + 1; j < paths.size(); ++j) {
          if (posAt(paths[i], t) == posAt(paths[j], t)) {
            out = Conflict{true, t, t, i, j, grid_.index(posAt(paths[i], t)), 0};
            return true;
          }
        }
      }
      // robustness conflicts: one agent in a cell that the other occupied up to k steps earlier
      for (int d = 1; d <= k; ++d) {
        if (t + d > max_t + k) {break;}
        for (std::size_t i = 0; i < paths.size(); ++i) {
          for (std::size_t j = 0; j < paths.size(); ++j) {
            if (i == j) {continue;}
            if (posAt(paths[i], t) == posAt(paths[j], t + d)) {
              out = Conflict{true, t, t + d, i, j, grid_.index(posAt(paths[i], t)), 0};
              return true;
            }
          }
        }
      }
      if (t == max_t) {break;}
      for (std::size_t i = 0; i < paths.size(); ++i) {
        const Cell & ia = posAt(paths[i], t);
        const Cell & ib = posAt(paths[i], t + 1);
        if (ia == ib) {continue;}
        for (std::size_t j = i + 1; j < paths.size(); ++j) {
          if (ia == posAt(paths[j], t + 1) && ib == posAt(paths[j], t)) {
            out = Conflict{false, t, t, i, j, grid_.index(ia), grid_.index(ib)};
            return true;
          }
        }
      }
    }
    return false;
  }

  // Number of agent pairs with at least one (vertex, robustness or swap) conflict.
  int countConflictingPairs(const std::vector<std::vector<Cell>> & paths) const
  {
    const int k = std::max(0, options_.robustness);
    int count = 0;
    for (std::size_t i = 0; i < paths.size(); ++i) {
      for (std::size_t j = i + 1; j < paths.size(); ++j) {
        const int T = static_cast<int>(std::max(paths[i].size(), paths[j].size())) + k;
        bool hit = false;
        for (int t = 0; t <= T && !hit; ++t) {
          for (int d = -k; d <= k && !hit; ++d) {
            if (t + d < 0) {continue;}
            hit = posAt(paths[i], t) == posAt(paths[j], t + d);
          }
          if (!hit && posAt(paths[i], t) != posAt(paths[i], t + 1)) {
            hit = posAt(paths[i], t) == posAt(paths[j], t + 1) && posAt(paths[i], t + 1) == posAt(paths[j], t);
          }
        }
        count += hit ? 1 : 0;
      }
    }
    return count;
  }

  Result failure(const std::string & status) const
  {
    Result r;
    r.status = status;
    return r;
  }

  Result solveImpl()
  {
    if (grid_.numCells() >= (1 << 20)) {return failure("grid_too_large");}
    // Input validation. Duplicated starts/goals would make CBS search forever.
    std::unordered_set<int> starts;
    std::unordered_set<int> goals;
    for (const auto & a : agents_) {
      if (!grid_.isFree(a.start.x, a.start.y)) {return failure("invalid_start");}
      if (!grid_.isFree(a.goal.x, a.goal.y)) {return failure("invalid_goal");}
      if (!starts.insert(grid_.index(a.start)).second) {return failure("duplicate_start");}
      if (!goals.insert(grid_.index(a.goal)).second) {return failure("duplicate_goal");}
      if (reservations_.vertexBlocked(0, grid_.index(a.start))) {return failure("start_reserved");}
    }

    std::vector<std::unique_ptr<HighLevelNode>> nodes;
    const double w = std::max(1.0, options_.suboptimality);
    std::set<std::pair<Cost, std::size_t>> open;                  // (cost, id)
    std::set<std::tuple<int, Cost, std::size_t>> focal;           // (conflicts, cost, id)
    Cost bound = 0;
    auto bound_of = [w](Cost c) {
        return static_cast<Cost>(std::floor(w * static_cast<double>(c) + 1e-9));
      };
    auto push = [&](std::unique_ptr<HighLevelNode> n) {
        const std::size_t id = nodes.size();
        open.emplace(n->total, id);
        if (n->total <= bound) {focal.emplace(n->conflicts, n->total, id);}
        nodes.push_back(std::move(n));
      };

    auto root = std::make_unique<HighLevelNode>();
    root->paths.resize(agents_.size());
    root->costs.resize(agents_.size(), 0);
    root->constraints.resize(agents_.size());
    for (std::size_t i = 0; i < agents_.size(); ++i) {
      if (!lowLevel(i, root->constraints[i], root->paths[i], root->costs[i])) {
        return failure("no_single_agent_path");
      }
      root->total += root->costs[i];
    }
    root->conflicts = countConflictingPairs(root->paths);
    bound = bound_of(root->total);
    push(std::move(root));

    while (!open.empty()) {
      if (deadline_.expired()) {return failure("timeout");}
      if (high_level_expanded_ >= options_.max_high_level_nodes) {return failure("node_limit");}
      // the minimum open cost never decreases (children cost >= parent cost), so the focal
      // bound only grows: add the nodes that newly qualify
      const Cost new_bound = bound_of(open.begin()->first);
      if (new_bound > bound) {
        for (auto it = open.upper_bound({bound, std::numeric_limits<std::size_t>::max()});
          it != open.end() && it->first <= new_bound; ++it)
        {
          focal.emplace(nodes[it->second]->conflicts, it->first, it->second);
        }
        bound = new_bound;
      }
      const std::size_t id = std::get<2>(*focal.begin());
      focal.erase(focal.begin());
      open.erase({nodes[id]->total, id});
      ++high_level_expanded_;
      const HighLevelNode & P = *nodes[id];

      Conflict c;
      if (!firstConflict(P.paths, c)) {
        return makeResult(P);
      }

      for (int side = 0; side < 2; ++side) {
        const std::size_t agent = side == 0 ? c.a1 : c.a2;
        auto child = std::make_unique<HighLevelNode>(P);
        if (c.vertex) {
          child->constraints[agent].addVertex(side == 0 ? c.t1 : c.t2, c.c1);
        } else if (side == 0) {
          child->constraints[agent].addEdge(c.t1, c.c1, c.c2);
        } else {
          child->constraints[agent].addEdge(c.t1, c.c2, c.c1);
        }
        Cost new_cost = 0;
        if (!lowLevel(agent, child->constraints[agent], child->paths[agent], new_cost)) {continue;}
        child->total += new_cost - child->costs[agent];
        child->costs[agent] = new_cost;
        child->conflicts = countConflictingPairs(child->paths);
        push(std::move(child));
      }
    }
    return failure("no_solution");
  }

  Result makeResult(const HighLevelNode & n) const
  {
    Result r;
    r.success = true;
    r.status = "success";
    r.paths = n.paths;
    r.objective_scaled = n.total;
    r.objective = static_cast<double>(n.total) / static_cast<double>(options_.cost_scale);
    for (const auto & p : n.paths) {
      const int steps = static_cast<int>(p.size()) - 1;
      r.sum_of_steps += steps;
      r.makespan = std::max(r.makespan, steps);
      for (int t = 1; t <= steps; ++t) {r.total_risk += risk_.at(t, p[t].x, p[t].y);}
    }
    return r;
  }

  const GridMap & grid_;
  const std::vector<Agent> & agents_;
  const RiskMap & risk_;
  ReservationTable reservations_;
  Options options_;
  Deadline deadline_;
  std::vector<std::vector<int>> heuristics_;
  int static_horizon_ = 0;
  std::size_t high_level_expanded_ = 0;
  std::size_t low_level_expanded_ = 0;
};

}  // namespace detail

inline Result solve(
  const GridMap & grid, const std::vector<Agent> & agents, const RiskMap & risk,
  const std::vector<Reservation> & reservations, const Options & options)
{
  detail::Solver solver(grid, agents, risk, reservations, options);
  return solver.solve();
}

}  // namespace risk_cbs
}  // namespace remroc_cbs_library

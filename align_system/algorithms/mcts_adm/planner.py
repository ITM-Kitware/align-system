from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

from .types import Action, Observation, ToolSpec
from .llm import CriticLLM, PlanCandidate, ProposerLLM


@dataclass
class Node:
    parent: Optional["Node"]
    # Instead of prior_action, store the macro that this edge represents:
    plan_actions: List[Action] = field(default_factory=list)

    children: List["Node"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    unexpanded: List[PlanCandidate] = field(default_factory=list)
    rationale: str = ""       # proposer's rationale for this candidate
    critic_reason: str = ""   # critic's explanation of its score

    @property
    def value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

class MCTSPlanner:
    """
    Tiny-budget MCTS:
    - each expansion calls proposer once for K candidates at that node
    - critic scores a short proposed future (model-based rollout)
    """

    def __init__(
        self,
        proposer: ProposerLLM,
        critic: CriticLLM,
        expansions: int = 5,
        proposals_per_expand: int = 4,
        rollout_horizon: int = 4,
        uct_c: float = 1.4,
        alpha: float = 0.0,
        seed: int = 0,
    ):
        self.proposer = proposer
        self.critic = critic
        self.expansions = expansions
        self.proposals_per_expand = proposals_per_expand
        self.rollout_horizon = rollout_horizon
        self.uct_c = uct_c
        random.seed(seed)

    def _uct_select(self, node: Node) -> Node:
        assert node.children
        logN = math.log(node.visits + 1)

        def uct(n: Node) -> float:
            if n.visits == 0:
                return float("inf")
            return n.value + self.uct_c * math.sqrt(logN / n.visits)

        return max(node.children, key=uct)

    def _reconstruct_actions(self, node: Node) -> List[Action]:
        chunks: List[List[Action]] = []
        cur = node
        while cur and cur.parent is not None:
            chunks.append(cur.plan_actions)
            cur = cur.parent
        chunks.reverse()
        out: List[Action] = []
        for ch in chunks:
            out.extend(ch)
        return out

    def plan(
        self,
        task: str,
        tasks_to_consider: str,
        obs: Observation,
        tools: List[ToolSpec],
        action_history: List[Action],
    ) -> List[Action]:
        root = Node(parent=None, plan_actions= None)

        for _ in range(self.expansions):
            # 1) selection
            node = root
            while node.children and not node.unexpanded:
                node = self._uct_select(node)

            prefix = self._reconstruct_actions(node)

            # 2) expansion
            if not node.unexpanded:
                node.unexpanded = self.proposer.propose(
                    task=task,
                    obs=obs,
                    tools=tools,
                    action_history=action_history + prefix,
                    k=self.proposals_per_expand,
                    diversity_hint="Vary tool choice; include at least one exploration move.",
                )
                print("Proposed paths.")
                print(node.unexpanded)

            if not node.unexpanded:
                continue

            cand = node.unexpanded.pop(0)
            if not cand.actions:
                continue

            child = Node(parent=node, plan_actions=cand.actions, rationale=cand.rationale)
            node.children.append(child)

            # 3) rollout scoring (critic)
            future = (cand.actions or [])[: self.rollout_horizon]
            action_strs = ", ".join(f"{a.tool_name}({a.args})" for a in (cand.actions or []))
            print(f"\n[EXPAND] rationale: {cand.rationale}")
            print(f"[EXPAND] actions:   {action_strs}")
            score, critic_reason = self.critic.score(
                task=tasks_to_consider,
                obs=obs,
                tools=tools,
                action_history=action_history + prefix,
                proposed_future=future,
                #rationale=cand.rationale,
            )
            child.critic_reason = critic_reason
            print(f"[SCORE]  {score:.3f}")
            if score is None:
                score = 0.0  # or `continue` to skip backprop for this node
            # 4) backprop
            cur = child
            while cur is not None:
                cur.visits += 1
                cur.value_sum += score
                cur = cur.parent

        if not root.children:
            return []
        self.print_tree(root, action_history)
        best = max(root.children, key=lambda n: n.value)
        return list(best.plan_actions)
    
    def print_tree(self, root: Node, action_history: List[Action]) -> None:

        # Collect all visited values to normalise colour thresholds
        all_values = [n.value for n in self._all_nodes(root) if n.visits > 0 and n.parent is not None]
        lo = min(all_values) if all_values else 0.0
        hi = max(all_values) if all_values else 1.0
        mid = (lo + hi) / 2

        def _colour(node: Node) -> str:
            if node.visits == 0:
                return "\033[90m"   # grey
            if hi == lo:
                return "\033[92m"   # all equal → just green
            if node.value >= mid + (hi - mid) * 0.5:
                return "\033[92m"   # green  — top third
            elif node.value >= mid:
                return "\033[93m"   # yellow — middle
            else:
                return "\033[91m"   # red    — bottom third

        def _node_label(node: Node) -> str:
            if node.parent is None:
                return "[ROOT]"
            action_strs = [f"{a.tool_name}({a.args})" if hasattr(a, 'tool_name') else str(a)
                        for a in (node.plan_actions or [])]
            actions_display = ", ".join(action_strs) if action_strs else "(empty)"
            value_str = f"{node.value:.3f}" if node.visits > 0 else "unvisited"
            return f"[v={value_str} | n={node.visits}] → {actions_display}"

        def _draw(node: Node, prefix: str, is_last: bool) -> None:
            connector = "└── " if is_last else "├── "
            colour, reset = _colour(node), "\033[0m"
            print(f"{prefix}{connector}{colour}{_node_label(node)}{reset}")
            child_prefix = prefix + ("    " if is_last else "│   ")
            for i, child in enumerate(node.children):
                _draw(child, child_prefix, i == len(node.children) - 1)

        print("\n" + "═" * 60)
        print("  MCTS TREE  (after expansion)")
        print("═" * 60)
        print(f"\033[97m[ROOT]\033[0m  visits={root.visits}")
        for i, child in enumerate(root.children):
            _draw(child, "", i == len(root.children) - 1)

        # ── Best PATH (not just best child) ──────────────────────
        if root.children:
            path, cur = [], max(root.children, key=lambda n: n.value)
            while cur is not None and cur.parent is not None:
                path.append(cur)
                cur = max(cur.children, key=lambda n: n.value) if cur.children else None

            print("\n" + "─" * 60)
            print("  \033[92m★ Best path\033[0m")
            for depth, n in enumerate(path):
                indent = "  " + "    " * depth
                actions = ", ".join(
                    f"{a.tool_name}({a.args})" if hasattr(a, "tool_name") else str(a)
                    for a in (n.plan_actions or [])
                )
                print(f"{indent}↳ [v={n.value:.3f} | n={n.visits}]  {actions}")
                if n.rationale:
                    print(f"{indent}  proposer: {n.rationale}")
                if n.critic_reason:
                    print(f"{indent}  critic:   {n.critic_reason}")
        print("═" * 60 + "\n")

    def _all_nodes(self, root: Node):
        """Flatten entire tree into a list."""
        result, stack = [], [root]
        while stack:
            n = stack.pop()
            result.append(n)
            stack.extend(n.children)
        return result

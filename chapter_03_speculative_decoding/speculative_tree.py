"""
EAGLE-2 Dynamic Speculative Draft Tree Architecture
Generates 2D tree-attention masks and position IDs for non-autoregressive verification.
Implements dynamic candidate tree expansion, depth and branching validation,
and verification-guided branch pruning.
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

@dataclass
class TreeNode:
    token_id: int
    depth: int
    score: float
    parent_idx: int

def build_tree_attention_mask(nodes: List[TreeNode]) -> List[List[int]]:
    """
    Builds a 2D causal tree attention mask where token i can only attend
    to itself and its direct ancestors in the speculative tree.
    Returns an (N x N) binary mask, or [] if nodes is empty.
    """
    n = len(nodes)
    if n == 0:
        return []

    mask = [[0] * n for _ in range(n)]
    for i in range(n):
        curr = i
        visited = set()
        while curr != -1:
            if curr in visited or curr < -1 or curr >= n:
                raise ValueError(
                    f"Invalid parent pointer or cycle detected at node index {curr} from node {i}"
                )
            visited.add(curr)
            mask[i][curr] = 1
            curr = nodes[curr].parent_idx
    return mask

def build_tree_position_ids(nodes: List[TreeNode], base_position: int = 0) -> List[int]:
    """
    Generates depth-aware position IDs for Rotary Position Embeddings (RoPE).
    Sibling nodes at the same tree depth share identical position IDs,
    ensuring correct relative positional encoding along every root-to-leaf path.
    """
    if base_position < 0:
        raise ValueError(f"base_position must be non-negative, got {base_position}")
    if not nodes:
        return []
    return [base_position + node.depth for node in nodes]

def validate_tree_structure(
    nodes: List[TreeNode],
    max_depth: Optional[int] = None,
    max_branching: Optional[int] = None
) -> bool:
    """
    Validates structural integrity of a speculative draft tree:
    - Empty tree is valid.
    - Exactly one root with parent_idx == -1 and depth == 0 located at index 0.
    - All non-root nodes have 0 <= parent_idx < len(nodes) and parent_idx != node_idx.
    - Strict parent-child depth consistency (depth[child] == depth[parent] + 1).
    - Acyclic directed tree structure.
    - All node depths <= max_depth (if max_depth is specified).
    - No node has more than max_branching children (if max_branching is specified).

    Returns True if valid; raises ValueError with descriptive error if invalid.
    """
    if not nodes:
        return True

    # 1. Exactly one root node
    root_indices = [i for i, n in enumerate(nodes) if n.parent_idx == -1]
    if len(root_indices) != 1:
        raise ValueError(f"Tree must have exactly one root (found {len(root_indices)})")
    root_idx = root_indices[0]
    if root_idx != 0:
        raise ValueError(f"Root node must be at index 0 (found at index {root_idx})")
    if nodes[root_idx].depth != 0:
        raise ValueError(f"Root node depth must be 0 (found {nodes[root_idx].depth})")

    n = len(nodes)
    children_count: Dict[int, int] = {i: 0 for i in range(n)}

    # 2. Check each node
    for i, node in enumerate(nodes):
        if i == root_idx:
            continue
        p = node.parent_idx
        if p < 0 or p >= n:
            raise ValueError(
                f"Node {i} has out-of-bounds parent_idx {p} (valid range: [0, {n - 1}])"
            )
        if p == i:
            raise ValueError(f"Node {i} has self-referencing parent_idx {p}")
        if node.depth != nodes[p].depth + 1:
            raise ValueError(
                f"Node {i} depth ({node.depth}) != parent {p} depth ({nodes[p].depth}) + 1"
            )
        children_count[p] += 1

        # Cycle detection
        curr = p
        visited = {i}
        while curr != -1:
            if curr in visited:
                raise ValueError(f"Cycle detected in speculative tree involving node {curr}")
            visited.add(curr)
            curr = nodes[curr].parent_idx

        if max_depth is not None and node.depth > max_depth:
            raise ValueError(
                f"Node {i} depth ({node.depth}) exceeds maximum allowable depth ({max_depth})"
            )

    if max_branching is not None:
        for p, count in children_count.items():
            if count > max_branching:
                raise ValueError(
                    f"Node {p} branching factor ({count}) exceeds maximum allowable ({max_branching})"
                )

    return True

def expand_tree_candidates(
    root_token: int,
    proposals: Union[Dict[int, List[Tuple[int, float]]], Callable[[int, int], List[Tuple[int, float]]]],
    max_depth: int = 3,
    max_nodes: int = 16,
    score_threshold: float = 0.0
) -> List[TreeNode]:
    """
    Expands a speculative draft tree from a root token using path-confidence scoring:
      score(child) = score(parent) * P(child | parent)
    Explores candidates in breadth-first order up to max_depth and max_nodes budget.
    """
    if max_nodes <= 0:
        return []

    root_node = TreeNode(token_id=root_token, depth=0, score=1.0, parent_idx=-1)
    nodes: List[TreeNode] = [root_node]
    queue: List[int] = [0]

    while queue and len(nodes) < max_nodes:
        parent_idx = queue.pop(0)
        p_node = nodes[parent_idx]

        if p_node.depth >= max_depth:
            continue

        if callable(proposals):
            cands = proposals(p_node.token_id, p_node.depth)
        else:
            cands = proposals.get(p_node.token_id, [])

        for child_token, prob in cands:
            child_score = p_node.score * prob
            if child_score < score_threshold:
                continue
            if len(nodes) >= max_nodes:
                break

            child_node = TreeNode(
                token_id=child_token,
                depth=p_node.depth + 1,
                score=round(child_score, 4),
                parent_idx=parent_idx
            )
            child_idx = len(nodes)
            nodes.append(child_node)
            queue.append(child_idx)

    return nodes

def verify_and_prune_tree(
    nodes: List[TreeNode],
    accepted_node_indices: Union[Set[int], List[int], List[bool]],
    include_root: bool = True
) -> Tuple[List[TreeNode], List[int]]:
    """
    Performs verification pruning on a speculative tree:
    1. Propagates acceptance from root: a node is accepted iff it is in accepted_node_indices
       AND all its ancestors are accepted.
    2. Prunes all rejected branches and downstream descendants.
    3. Identifies the longest accepted root-to-leaf path (breaking ties by highest score).

    Returns:
      (surviving_nodes, longest_accepted_token_sequence)
    """
    if not nodes:
        return [], []

    # Normalize accepted indices into a set
    if isinstance(accepted_node_indices, list):
        if accepted_node_indices and isinstance(accepted_node_indices[0], bool):
            accepted_set = {i for i, acc in enumerate(accepted_node_indices) if acc}
        else:
            accepted_set = set(accepted_node_indices)
    else:
        accepted_set = set(accepted_node_indices)

    # Acceptance propagation along causal tree edges
    valid_accepted_indices: Set[int] = set()
    for i, node in enumerate(nodes):
        if node.parent_idx == -1:
            if include_root or (i in accepted_set):
                valid_accepted_indices.add(i)
        else:
            if (i in accepted_set) and (node.parent_idx in valid_accepted_indices):
                valid_accepted_indices.add(i)

    if not valid_accepted_indices:
        return [], []

    surviving_nodes = [nodes[i] for i in sorted(valid_accepted_indices)]

    # Identify longest accepted path (tie-breaking with path score)
    best_leaf = -1
    best_depth = -1
    best_score = -1.0
    for idx in valid_accepted_indices:
        node = nodes[idx]
        if (node.depth > best_depth) or (node.depth == best_depth and node.score > best_score):
            best_depth = node.depth
            best_score = node.score
            best_leaf = idx

    path_indices: List[int] = []
    curr = best_leaf
    while curr != -1:
        path_indices.append(curr)
        curr = nodes[curr].parent_idx
    path_indices.reverse()

    accepted_tokens = [nodes[idx].token_id for idx in path_indices]
    return surviving_nodes, accepted_tokens

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 3: EAGLE-2 TREE-BASED SPECULATIVE VERIFICATION")
    print("=" * 80)
    
    # 1. Manually defined speculative draft tree
    tree_nodes = [
        TreeNode(token_id=101, depth=0, score=1.00, parent_idx=-1),  # Root
        TreeNode(token_id=202, depth=1, score=0.85, parent_idx=0),   # Branch A
        TreeNode(token_id=203, depth=1, score=0.65, parent_idx=0),   # Branch B
        TreeNode(token_id=305, depth=2, score=0.72, parent_idx=1),   # A -> Child 1
        TreeNode(token_id=306, depth=2, score=0.68, parent_idx=1),   # A -> Child 2
        TreeNode(token_id=307, depth=2, score=0.55, parent_idx=2),   # B -> Child 1
    ]
    
    validate_tree_structure(tree_nodes, max_depth=2, max_branching=2)
    mask = build_tree_attention_mask(tree_nodes)
    pos_ids = build_tree_position_ids(tree_nodes, base_position=128)
    
    print(f"Constructed 2D Speculative Tree Mask ({len(tree_nodes)}x{len(tree_nodes)}):")
    for row in mask:
        print("  ", row)
    print(f"Depth-Aware Position IDs (base=128): {pos_ids}")
    print("\nVerified: Each child attends strictly to its causal prefix path.")

    # 2. Dynamic tree expansion
    proposals = {
        101: [(202, 0.85), (203, 0.65)],
        202: [(305, 0.847), (306, 0.80)],
        203: [(307, 0.846)]
    }
    expanded = expand_tree_candidates(101, proposals, max_depth=2, max_nodes=6)
    print(f"\nDynamically Expanded Tree ({len(expanded)} nodes):")
    for i, n in enumerate(expanded):
        print(f"  Node {i}: token={n.token_id}, depth={n.depth}, score={n.score}, parent={n.parent_idx}")

    # 3. Verification pruning simulation:
    # Nodes 1 and 3 accepted (Branch A survives); Node 2 rejected (Branch B pruned along with Child 5)
    surviving, accepted_seq = verify_and_prune_tree(expanded, accepted_node_indices={1, 3, 5})
    print(f"\nVerification Pruning (accepted candidates = {{1, 3, 5}}):")
    print(f"  Surviving Subtree Nodes: {[n.token_id for n in surviving]}")
    print(f"  Longest Accepted Token Sequence: {accepted_seq}")
    print("=" * 80)

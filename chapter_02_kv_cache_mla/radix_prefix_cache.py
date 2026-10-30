"""
SGLang RadixAttention Prefix Caching Engine
Implements deterministic prefix matching, dynamic tree node splitting,
reference counting, and LRU memory block eviction under GPU pressure.
"""
import time
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

@dataclass
class RadixTreeNode:
    node_id: int
    token_ids: List[int] = field(default_factory=list)
    children: Dict[int, 'RadixTreeNode'] = field(default_factory=dict)
    parent: Optional['RadixTreeNode'] = None
    ref_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    value_block_ids: List[int] = field(default_factory=list)

class RadixAttentionEngine:
    def __init__(self, total_gpu_blocks: int = 1000, block_size: int = 16):
        self.total_gpu_blocks = total_gpu_blocks
        self.block_size = block_size
        self.free_blocks = list(range(total_gpu_blocks))
        self.node_counter = 0
        self.root = RadixTreeNode(node_id=0)
        self.nodes_map: Dict[int, RadixTreeNode] = {0: self.root}
        self.stats = {"cache_hits": 0, "cache_misses": 0, "evicted_nodes": 0}

    def _split_node(self, parent: RadixTreeNode, child: RadixTreeNode, split_idx: int) -> RadixTreeNode:
        """
        Splits `child` node at `split_idx` tokens.
        Creates `split_node` containing child.token_ids[:split_idx],
        and updates `child` to contain child.token_ids[split_idx:].
        Returns the newly created `split_node`.
        """
        if split_idx <= 0 or split_idx >= len(child.token_ids):
            raise ValueError(f"Invalid split_idx {split_idx} for child with {len(child.token_ids)} tokens")

        self.node_counter += 1
        child_toks = child.token_ids
        prefix_toks = child_toks[:split_idx]
        suffix_toks = child_toks[split_idx:]
        
        split_blocks_count = math.ceil(split_idx / self.block_size)
        split_blocks = child.value_block_ids[:split_blocks_count]
        remaining_blocks = child.value_block_ids[split_blocks_count:]
        
        needed_suffix_blocks = math.ceil(len(suffix_toks) / self.block_size)
        if len(remaining_blocks) < needed_suffix_blocks:
            shortfall = needed_suffix_blocks - len(remaining_blocks)
            additional_blocks = self._allocate_blocks(shortfall)
            remaining_blocks = remaining_blocks + additional_blocks
        
        split_node = RadixTreeNode(
            node_id=self.node_counter,
            token_ids=prefix_toks,
            parent=parent,
            ref_count=child.ref_count,
            last_access_time=child.last_access_time,
            value_block_ids=split_blocks
        )
        
        child.token_ids = suffix_toks
        child.value_block_ids = remaining_blocks
        child.parent = split_node
        
        split_node.children[child.token_ids[0]] = child
        parent.children[prefix_toks[0]] = split_node
        self.nodes_map[split_node.node_id] = split_node
        
        return split_node

    def match_prefix(self, token_seq: List[int]) -> Tuple[List[int], List[RadixTreeNode]]:
        curr = self.root
        curr.last_access_time = time.time()
        idx = 0
        matched_blocks = []
        traversed_nodes = [curr]
        while idx < len(token_seq):
            first_tok = token_seq[idx]
            if first_tok not in curr.children:
                break
            child = curr.children[first_tok]
            child_tokens = child.token_ids
            match_len = 0
            while match_len < len(child_tokens) and (idx + match_len) < len(token_seq):
                if child_tokens[match_len] == token_seq[idx + match_len]:
                    match_len += 1
                else:
                    break
            if match_len == len(child_tokens):
                idx += match_len
                matched_blocks.extend(child.value_block_ids)
                curr = child
                curr.last_access_time = time.time()
                traversed_nodes.append(curr)
            elif match_len > 0:
                split_node = self._split_node(curr, child, match_len)
                idx += match_len
                matched_blocks.extend(split_node.value_block_ids)
                curr = split_node
                curr.last_access_time = time.time()
                traversed_nodes.append(curr)
                break
            else:
                break
        if len(matched_blocks) > 0:
            self.stats["cache_hits"] += 1
        else:
            self.stats["cache_misses"] += 1
        return matched_blocks, traversed_nodes

    def insert_sequence(self, token_seq: List[int]) -> List[int]:
        if not token_seq:
            return []
        curr = self.root
        idx = 0
        all_allocated = []
        while idx < len(token_seq):
            first_tok = token_seq[idx]
            if first_tok not in curr.children:
                rem = token_seq[idx:]
                n_blocks = math.ceil(len(rem) / self.block_size)
                new_blocks = self._allocate_blocks(n_blocks)
                self.node_counter += 1
                new_node = RadixTreeNode(
                    node_id=self.node_counter,
                    token_ids=rem,
                    parent=curr,
                    ref_count=1,
                    value_block_ids=new_blocks
                )
                curr.children[first_tok] = new_node
                self.nodes_map[new_node.node_id] = new_node
                all_allocated.extend(new_blocks)
                return all_allocated
            else:
                child = curr.children[first_tok]
                child_toks = child.token_ids
                match_len = 0
                while match_len < len(child_toks) and (idx + match_len) < len(token_seq):
                    if child_toks[match_len] == token_seq[idx + match_len]:
                        match_len += 1
                    else:
                        break
                if match_len == len(child_toks):
                    idx += match_len
                    child.ref_count += 1
                    child.last_access_time = time.time()
                    curr = child
                else:
                    # Node split
                    split_node = self._split_node(curr, child, match_len)
                    split_node.ref_count += 1
                    split_node.last_access_time = time.time()
                    curr = split_node
                    idx += match_len
        return all_allocated

    def dec_ref_count(self, node: RadixTreeNode) -> None:
        if node != self.root and node.ref_count > 0:
            node.ref_count -= 1

    def inc_ref_count(self, node: RadixTreeNode) -> None:
        if node != self.root:
            node.ref_count += 1

    def release_sequence(self, token_seq: List[int]) -> int:
        """Decrements ref_count of all nodes corresponding to token_seq."""
        curr = self.root
        idx = 0
        released = 0
        while idx < len(token_seq):
            first_tok = token_seq[idx]
            if first_tok not in curr.children:
                break
            child = curr.children[first_tok]
            child_tokens = child.token_ids
            match_len = 0
            while match_len < len(child_tokens) and (idx + match_len) < len(token_seq):
                if child_tokens[match_len] == token_seq[idx + match_len]:
                    match_len += 1
                else:
                    break
            if match_len == len(child_tokens):
                idx += match_len
                self.dec_ref_count(child)
                released += 1
                curr = child
            else:
                break
        return released

    def release_traversed_nodes(self, nodes: List[RadixTreeNode]) -> None:
        """Decrements ref_count for an explicit list of traversed nodes."""
        for node in nodes:
            self.dec_ref_count(node)

    def evict_to_target_free_blocks(self, target_free: int) -> int:
        evicted = 0
        while len(self.free_blocks) < target_free:
            if not self._evict_lru_node():
                break
            evicted += 1
        return evicted

    def get_total_cached_tokens(self) -> int:
        return sum(len(n.token_ids) for nid, n in self.nodes_map.items() if nid != self.root.node_id)

    def get_sequence_blocks(self, token_seq: List[int]) -> List[int]:
        blocks, _ = self.match_prefix(token_seq)
        return blocks

    def _allocate_blocks(self, count: int) -> List[int]:
        allocated = []
        while len(allocated) < count:
            if self.free_blocks:
                allocated.append(self.free_blocks.pop(0))
            else:
                self._evict_lru_node()
                if not self.free_blocks:
                    raise RuntimeError("GPU Out of Memory in Radix Cache")
        return allocated

    def _evict_lru_node(self) -> bool:
        candidates = [n for n in self.nodes_map.values() if n.ref_count == 0 and len(n.children) == 0 and n != self.root]
        if not candidates:
            return False
        candidates.sort(key=lambda n: n.last_access_time)
        victim = candidates[0]
        self.free_blocks.extend(victim.value_block_ids)
        victim.value_block_ids.clear()
        if victim.parent:
            for k, v in list(victim.parent.children.items()):
                if v == victim:
                    del victim.parent.children[k]
        del self.nodes_map[victim.node_id]
        self.stats["evicted_nodes"] += 1
        return True

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 2: SGLANG RADIXATTENTION PREFIX CACHING SIMULATOR")
    print("=" * 80)
    engine = RadixAttentionEngine(total_gpu_blocks=500, block_size=16)
    sys_prompt = list(range(100, 300)) # 200 token system prompt
    
    # Request 1 (Initial prompt)
    req1 = sys_prompt + list(range(1000, 1050))
    m1, _ = engine.match_prefix(req1)
    alloc1 = engine.insert_sequence(req1)
    print(f"Req 1: Matched {len(m1)*16} tokens | Allocated {len(alloc1)} blocks")
    
    # Request 2 (Shares system prompt prefix)
    req2 = sys_prompt + list(range(2000, 2080))
    m2, _ = engine.match_prefix(req2)
    alloc2 = engine.insert_sequence(req2)
    print(f"Req 2: Matched {len(m2)*16} tokens (Prefix Reused!) | Allocated {len(alloc2)} new blocks")
    
    print(f"Cache Statistics: Hits={engine.stats['cache_hits']}, Misses={engine.stats['cache_misses']}")
    print("=" * 80)

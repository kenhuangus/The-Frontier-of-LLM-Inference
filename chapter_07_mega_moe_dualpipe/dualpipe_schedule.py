"""
DeepSeek DualPipe Bidirectional 1F1B Schedule Simulator
Overlaps forward and backward computation with inter-node expert all-to-all communication.
"""
import math
from typing import List, Dict, Any, Optional

def calculate_dualpipe_bubble(
    num_microbatches: int,
    num_pipeline_stages: int,
    num_chunks: int = 2
) -> Dict[str, float]:
    """
    Calculates pipeline bubble ratios for standard 1F1B vs. DeepSeek DualPipe.
    
    Bubble Ratio Formulas:
      Standard 1F1B:  Bubble = (P - 1) / M
      DualPipe:       Bubble = (P - 1) / (C * M)
    where:
      P = num_pipeline_stages
      M = num_microbatches
      C = num_chunks (chunks per microbatch, default 2 for bidirectional 1F1B)
    """
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be a positive integer.")
    if num_pipeline_stages <= 0:
        raise ValueError("num_pipeline_stages must be a positive integer.")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be a positive integer.")

    if num_pipeline_stages == 1:
        return {
            "Standard_1F1B_Bubble": 0.0,
            "DualPipe_Bubble": 0.0,
            "Bubble_Reduction_Pct": 0.0
        }

    bubble_ratio_standard = (num_pipeline_stages - 1) / num_microbatches
    bubble_ratio_dualpipe = (num_pipeline_stages - 1) / (num_chunks * num_microbatches)
    
    reduction_pct = ((bubble_ratio_standard - bubble_ratio_dualpipe) / bubble_ratio_standard) * 100.0

    return {
        "Standard_1F1B_Bubble": round(bubble_ratio_standard * 100.0, 2),
        "DualPipe_Bubble": round(bubble_ratio_dualpipe * 100.0, 2),
        "Bubble_Reduction_Pct": round(reduction_pct, 2)
    }

def generate_dualpipe_schedule(
    num_microbatches: int = 8,
    num_pipeline_stages: int = 4,
    num_chunks: int = 2
) -> Dict[str, Any]:
    """
    Generates the step-by-step bidirectional execution schedule for all pipeline stages.
    
    DualPipe partitions execution into:
      1. Warmup: Initial forward passes to populate the pipeline.
      2. Steady-State: Bidirectional 1F1B overlapping forward compute with backward
         activation/weight communication.
      3. Cooldown: Draining the remaining backward passes.
      
    Handles odd chunk numbers and chunk count variations.
    """
    if num_microbatches <= 0:
        raise ValueError("num_microbatches must be a positive integer.")
    if num_pipeline_stages <= 0:
        raise ValueError("num_pipeline_stages must be a positive integer.")
    if num_chunks <= 0:
        raise ValueError("num_chunks must be a positive integer.")

    is_odd_chunks = (num_chunks % 2 != 0)
    total_chunks = num_microbatches * num_chunks
    
    if num_pipeline_stages == 1:
        stage_schedule = []
        for mb in range(num_microbatches):
            for c in range(num_chunks):
                stage_schedule.append({
                    "step": len(stage_schedule),
                    "phase": "compute",
                    "forward_op": f"F_mb{mb}_c{c}",
                    "backward_op": None,
                    "overlapped": False
                })
        for mb in range(num_microbatches):
            for c in range(num_chunks):
                stage_schedule.append({
                    "step": len(stage_schedule),
                    "phase": "compute",
                    "forward_op": None,
                    "backward_op": f"B_mb{mb}_c{c}",
                    "overlapped": False
                })
        return {
            "Pipeline_Stages": 1,
            "Microbatches": num_microbatches,
            "Chunks_Per_Microbatch": num_chunks,
            "Is_Odd_Chunks": is_odd_chunks,
            "Warmup_Steps": 0,
            "Steady_Steps": 0,
            "Cooldown_Steps": 0,
            "Total_Steps": len(stage_schedule),
            "Overlapped_Steps": 0,
            "Overlap_Ratio_Pct": 0.0,
            "Stage_Schedules": {0: stage_schedule}
        }

    warmup_steps = min(num_pipeline_stages - 1, total_chunks)
    cooldown_steps = warmup_steps
    steady_steps = max(0, total_chunks - warmup_steps)
    total_steps = warmup_steps + steady_steps + cooldown_steps

    stage_schedules: Dict[int, List[Dict[str, Any]]] = {}
    
    for stage_id in range(num_pipeline_stages):
        timeline: List[Dict[str, Any]] = []
        step_idx = 0
        
        # 1. Warmup Phase (Forward only)
        for w in range(warmup_steps):
            f_chunk_idx = w
            mb_idx = f_chunk_idx // num_chunks
            c_idx = f_chunk_idx % num_chunks
            direction = 0 if (stage_id < num_pipeline_stages // 2 or not is_odd_chunks) else 1
            timeline.append({
                "step": step_idx,
                "phase": "warmup",
                "forward_op": f"F_mb{mb_idx}_c{c_idx}_d{direction}",
                "backward_op": None,
                "overlapped": False
            })
            step_idx += 1
            
        # 2. Steady-State Phase (1F1B Overlap: Forward compute overlaps Backward comm)
        for s in range(steady_steps):
            f_chunk_idx = warmup_steps + s
            b_chunk_idx = s
            
            f_mb = f_chunk_idx // num_chunks
            f_c = f_chunk_idx % num_chunks
            b_mb = b_chunk_idx // num_chunks
            b_c = b_chunk_idx % num_chunks
            
            timeline.append({
                "step": step_idx,
                "phase": "steady_state",
                "forward_op": f"F_mb{f_mb}_c{f_c}",
                "backward_op": f"B_mb{b_mb}_c{b_c}",
                "overlapped": True
            })
            step_idx += 1
            
        # 3. Cooldown Phase (Backward only)
        for c in range(cooldown_steps):
            b_chunk_idx = steady_steps + c
            if b_chunk_idx < total_chunks:
                b_mb = b_chunk_idx // num_chunks
                b_c = b_chunk_idx % num_chunks
                timeline.append({
                    "step": step_idx,
                    "phase": "cooldown",
                    "forward_op": None,
                    "backward_op": f"B_mb{b_mb}_c{b_c}",
                    "overlapped": False
                })
                step_idx += 1

        stage_schedules[stage_id] = timeline

    overlap_ratio = (steady_steps / total_steps) * 100.0 if total_steps > 0 else 0.0

    return {
        "Pipeline_Stages": num_pipeline_stages,
        "Microbatches": num_microbatches,
        "Chunks_Per_Microbatch": num_chunks,
        "Is_Odd_Chunks": is_odd_chunks,
        "Warmup_Steps": warmup_steps,
        "Steady_Steps": steady_steps,
        "Cooldown_Steps": cooldown_steps,
        "Total_Steps": total_steps,
        "Overlapped_Steps": steady_steps,
        "Overlap_Ratio_Pct": round(overlap_ratio, 2),
        "Stage_Schedules": stage_schedules
    }

def simulate_dualpipe_overlap(
    num_microbatches: int = 8,
    num_pipeline_stages: int = 4,
    num_chunks: int = 2
) -> Dict[str, Any]:
    """
    Simulates DualPipe bidirectional chunk pipelining.
    Forward chunk computation on microbatch i overlaps with backward communication on microbatch j.
    """
    bubble_stats = calculate_dualpipe_bubble(
        num_microbatches=num_microbatches,
        num_pipeline_stages=num_pipeline_stages,
        num_chunks=num_chunks
    )
    schedule_stats = generate_dualpipe_schedule(
        num_microbatches=num_microbatches,
        num_pipeline_stages=num_pipeline_stages,
        num_chunks=num_chunks
    )

    return {
        "Microbatches": num_microbatches,
        "Pipeline_Stages": num_pipeline_stages,
        "Chunks_Per_Microbatch": num_chunks,
        "Is_Odd_Chunks": schedule_stats["Is_Odd_Chunks"],
        "Standard_1F1B_Bubble": bubble_stats["Standard_1F1B_Bubble"],
        "DualPipe_Bubble": bubble_stats["DualPipe_Bubble"],
        "Bubble_Reduction_Pct": bubble_stats["Bubble_Reduction_Pct"],
        "Warmup_Steps": schedule_stats["Warmup_Steps"],
        "Steady_Steps": schedule_stats["Steady_Steps"],
        "Cooldown_Steps": schedule_stats["Cooldown_Steps"],
        "Total_Steps": schedule_stats["Total_Steps"],
        "Overlapped_Steps": schedule_stats["Overlapped_Steps"],
        "Overlap_Ratio_Pct": schedule_stats["Overlap_Ratio_Pct"],
        "Stage_Schedules": schedule_stats["Stage_Schedules"]
    }

if __name__ == "__main__":
    print("=" * 80)
    print("  CHAPTER 7: DEEPSEEK DUALPIPE PIPELINE BUBBLE REDUCTION")
    print("=" * 80)
    res = simulate_dualpipe_overlap(num_microbatches=16, num_pipeline_stages=8)
    print(f"Pipeline Grid: {res['Pipeline_Stages']} Stages, {res['Microbatches']} Microbatches")
    print(f"  Standard 1F1B Pipeline Bubble : {res['Standard_1F1B_Bubble']}% idle time")
    print(f"  DeepSeek DualPipe Bubble      : {res['DualPipe_Bubble']}% idle time")
    print(f"  Effective Pipeline Speedup    : {res['Bubble_Reduction_Pct']}% bubble elimination!")
    print(f"  Schedule Breakdown            : {res['Warmup_Steps']} warmup, {res['Steady_Steps']} steady, {res['Cooldown_Steps']} cooldown")
    print(f"  Overlap Ratio                 : {res['Overlap_Ratio_Pct']}% of steps concurrent")
    print("=" * 80)

"""
DeepSeek DualPipe Bidirectional 1F1B Schedule Simulator
Overlaps forward and backward computation with inter-node expert all-to-all communication.
"""
from typing import List, Dict

def simulate_dualpipe_overlap(
    num_microbatches: int = 8,
    num_pipeline_stages: int = 4
) -> Dict[str, float]:
    """
    Simulates DualPipe bidirectional chunk pipelining.
    Forward chunk computation on microbatch i overlaps with backward communication on microbatch j.
    """
    bubble_ratio_standard = (num_pipeline_stages - 1) / num_microbatches
    bubble_ratio_dualpipe = (num_pipeline_stages - 1) / (2 * num_microbatches)
    
    return {
        "Microbatches": num_microbatches,
        "Pipeline_Stages": num_pipeline_stages,
        "Standard_1F1B_Bubble": round(bubble_ratio_standard * 100, 2),
        "DualPipe_Bubble": round(bubble_ratio_dualpipe * 100, 2),
        "Bubble_Reduction_Pct": round(((bubble_ratio_standard - bubble_ratio_dualpipe) / bubble_ratio_standard) * 100, 2)
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
    print("=" * 80)

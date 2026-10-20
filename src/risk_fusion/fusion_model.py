"""
fusion_model.py
----------------
Combines classifier probability + posture score into
a final risk level: Low / Medium / High
"""


def fuse_risk(prob_aggressive: float, posture: float) -> tuple:
    """
    Args:
        prob_aggressive: float 0-1 from ResNet18 sigmoid
        posture: float 0-1 from posture heuristics

    Returns:
        (risk_level: str, confidence: float)
        risk_level: "Low" | "Medium" | "High"
    """
    # Weighted fusion: classifier is more reliable
    fused = 0.75 * prob_aggressive + 0.25 * posture

    if fused < 0.35:
        risk_level = "Low"
        confidence = 1.0 - fused
    elif fused < 0.65:
        risk_level = "Medium"
        # confidence peaks at 0.5
        confidence = 1.0 - abs(fused - 0.5) * 2
    else:
        risk_level = "High"
        confidence = fused

    return risk_level, round(float(confidence), 4)
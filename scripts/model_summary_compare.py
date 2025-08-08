import sys
sys.path.insert(0, '.')  # Use local ultralytics

from ultralytics import YOLO

def info(model_path, name):
    m = YOLO(model_path)
    print(f"=== {name} ===")
    m.info(verbose=False)
    params = sum(p.numel() for p in m.model.parameters())
    return params

if __name__ == "__main__":
    print("=== MODEL PARAMETER COMPARISON ===\n")
    
    # Compare baseline vs our P2-CA model
    baseline_params = info("yolov8m.yaml", "BASELINE YOLOv8m")
    our_params = info("yolov8m_p2_ca_budget.yaml", "YOLOv8m P2-CA BUDGET")
    
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Baseline YOLOv8m: {baseline_params:,} parameters")
    print(f"YOLOv8m P2-CA: {our_params:,} parameters")
    diff = our_params - baseline_params
    pct = (diff / baseline_params) * 100
    print(f"Difference: {diff:,} parameters ({pct:+.1f}%)")
    
    if abs(pct) <= 5:
        print("✅ Parameter count is within 5% of baseline (budget neutral)")
    else:
        print("❌ Parameter count differs by more than 5% from baseline")
    
    print(f"\n🎯 Added P2 output for small objects with only {pct:.1f}% parameter increase!")
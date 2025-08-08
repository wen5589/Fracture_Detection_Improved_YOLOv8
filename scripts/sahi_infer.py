import argparse
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to best.pt")
    ap.add_argument("--source", required=True, help="image or dir")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--slice_w", type=int, default=640)
    ap.add_argument("--slice_h", type=int, default=640)
    ap.add_argument("--overlap", type=float, default=0.25)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.7)
    ap.add_argument("--save", type=str, default="runs/sahi_pred")
    args = ap.parse_args()

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        confidence_threshold=args.conf,
        device="cuda:0"
    )

    result = get_sliced_prediction(
        args.source,
        detection_model=model,
        slice_height=args.slice_h,
        slice_width=args.slice_w,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        postprocess_match_metric="IOU",
        postprocess_match_threshold=args.iou,
        auto_slice_resolution=False,
        project=args.save
    )
    print(result)

if __name__ == "__main__":
    main()
# Small-object mAP improvement without increasing params

## Train
```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolov8m_p2_ca_budget.yaml \
  data=your_data.yaml imgsz=960 epochs=500 batch=16 \
  hyp=ultralytics/cfg/hyps/hyp_smallobj_ca.yaml device=0 workers=8
```

## Validate
```bash
yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=your_data.yaml imgsz=960 conf=0.001 iou=0.7
```

## SAHI sliced inference (better APs without changing parameters)
```bash
pip install sahi
python scripts/sahi_infer.py \
  --model runs/detect/train/weights/best.pt \
  --source path/to/images_or_dir \
  --imgsz 960 --slice_w 640 --slice_h 640 --overlap 0.25
```

## Compare params & GFLOPs vs baseline
```bash
python scripts/model_summary_compare.py
```

Notes:
- Adjust `nc` in the model YAML to your number of classes if different from 9.
- If parameter count is slightly higher than baseline, narrow head widths (e.g., 448->416, 896->832) or set repeats=1 on head C2f to match exactly.
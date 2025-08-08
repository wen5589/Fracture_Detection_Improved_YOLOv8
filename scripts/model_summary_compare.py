from ultralytics import YOLO

def info(model_path, imgsz=640):
    m = YOLO(model_path)
    print(f"==> {model_path}")
    m.info(verbose=True, imgsz=imgsz)

if __name__ == "__main__":
    # Adjust baseline path if needed
    info("ultralytics/cfg/models/v8/yolov8.yaml", imgsz=640)
    info("ultralytics/cfg/models/v8/yolov8m_p2_ca_budget.yaml", imgsz=640)
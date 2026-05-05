from ultralytics import YOLO

# To jest to kluczowe zabezpieczenie dla systemu Windows:
if __name__ == '__main__':
    # Wczytujemy bazowy model Pose
    model = YOLO('yolov8n-pose.pt')

    # Uruchamiamy trenowanie
    results = model.train(
        data='data.yaml',  # Upewnij się, że nazwa pliku to data.yaml (zgodnie z logami)
        epochs=100,
        imgsz=640,
        device=0,
        name='pitch_keypoints'
    )
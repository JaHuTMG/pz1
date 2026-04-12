import cv2
import numpy as np
import json
from ultralytics import YOLO
from sklearn.cluster import KMeans

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def get_dominant_color(crop):
    h, w = crop.shape[:2]

    # Pobranie tylko kolorow koszulek
    chest = crop[int(h * 0.20):int(h * 0.55), :]

    if chest.size == 0:
        return np.array([0, 0, 0])

    hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)

    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)

    pixels = chest[non_green_mask == 255]

    if len(pixels) < 10:
        return np.array([0, 0, 0])

    if len(pixels) > 300:
        np.random.shuffle(pixels)
        pixels = pixels[:300]

    kmeans = KMeans(n_clusters=1, n_init=1, max_iter=5)
    kmeans.fit(pixels)

    return kmeans.cluster_centers_[0].astype(int)


model = YOLO('yolo11s.pt')
cap = cv2.VideoCapture("test.mp4")

id_mapping = {}  # Nowe ID -> Stare ID
last_positions = {}  # Ostatnie X,Y każdego ID
last_seen_frame = {}  # Numer klatki ostatniego widzenia
last_colors = {}  # Zapamiętany kolor koszulki każdego ID

match_data = []
frame_id = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame_id += 1
    frame_objects = []

    h, w = frame.shape[:2]
    scale_ratio = 1280 / w if w > 1280 else 1.0
    if w > 1280:
        frame = cv2.resize(frame, (1280, int(h * scale_ratio)))

    results = model.track(frame, persist=True, conf=0.1, tracker="tracker_config.yaml")

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        classes = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):
            if cls != 0: continue  # Skupienie sie tylko na zawodnikach

            x1, y1, x2, y2 = box
            original_track_id = int(track_id)
            curr_pos = np.array([int((x1 + x2) / 2), y2])

            # Pobranie koloru
            person_crop = frame[y1:y2, x1:x2]
            current_color = get_dominant_color(person_crop)

            # Odzyskanie zgubionego ID
            if original_track_id in id_mapping:
                track_id = id_mapping[original_track_id]
            else:
                best_match = None
                min_dist = 120  # Dystans szukania zgubionego ID

                for old_id, old_pos in last_positions.items():
                    # Ignorujemy graczy aktualnie widocznych
                    if last_seen_frame.get(old_id, 0) == frame_id: continue
                    # Ignorujemy tych, których nie ma > 2 sekundy (60 klatek)
                    if frame_id - last_seen_frame.get(old_id, 0) > 60: continue

                    # Filtr koloru
                    color_diff = np.linalg.norm(current_color - last_colors.get(old_id, np.array([0, 0, 0])))

                    if color_diff > 45:
                        continue

                    dist = np.linalg.norm(curr_pos - old_pos)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = old_id

                if best_match is not None:
                    id_mapping[original_track_id] = best_match
                    track_id = best_match

            # Aktualizacja pamieci
            last_positions[track_id] = curr_pos
            last_seen_frame[track_id] = frame_id
            if sum(current_color) > 30:
                last_colors[track_id] = current_color

            # Rysowanie i zapis
            b, g, r = last_colors.get(track_id, current_color)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (int(b), int(g), int(r)), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(b), int(g), int(r)),
                        2)

            frame_objects.append({
                "id": int(track_id),
                "type": "person",
                "point": [int(curr_pos[0] / scale_ratio), int(curr_pos[1] / scale_ratio)]
            })

    if frame_objects:
        match_data.append({"frame_id": frame_id, "objects": frame_objects})

    cv2.imshow("Detekcja", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

# EKSPORT DO JSON PO ZAKOŃCZENIU
# print("\n[INFO] Zapisywanie trajektorii do pliku match_data.json...")
# with open('match_data.json', 'w') as f:
#     json.dump(match_data, f, indent=4, cls=NumpyEncoder)
# print("[INFO] Gotowe")
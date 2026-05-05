import cv2
import numpy as np
import json
from ultralytics import YOLO
from sklearn.cluster import KMeans
import pandas as pd
import supervision as sv

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)): return int(obj)
        if isinstance(obj, (np.floating, np.float64)): return float(obj)
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
model.to('cuda')
pitch_model = YOLO('runs/pose/pitch_keypoints-7/weights/best.pt')
pitch_model.to('cuda')

video_path = "test.mp4"

tracker = sv.ByteTrack(track_activation_threshold=0.1, lost_track_buffer=60)
generator = sv.get_video_frames_generator(video_path)

last_colors = {}
match_data = []
ball_positions = []
ALPHA = 0.2


for frame_id, frame in enumerate(generator, start=1):
    frame_objects = []
    current_ball_pos = [None, None]

    h, w = frame.shape[:2]
    scale_ratio = 1280 / w if w > 1280 else 1.0

    if scale_ratio != 1.0:
        proc_frame = cv2.resize(frame, (1280, int(h * scale_ratio)))
    else:
        proc_frame = frame.copy()

    results = model(proc_frame, imgsz=1280, conf=0.05, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    players_det = detections[detections.class_id == 0]
    tracked_players = tracker.update_with_detections(players_det)

    for box, track_id in zip(tracked_players.xyxy, tracked_players.tracker_id):
        x1, y1, x2, y2 = box.astype(int)

        feet_x = int(((x1 + x2) / 2) / scale_ratio)
        feet_y = int(y2 / scale_ratio)

        person_crop = proc_frame[y1:y2, x1:x2]
        current_color = get_dominant_color(person_crop)

        if sum(current_color) > 30:
            if track_id in last_colors:
                smoothed_color = (1 - ALPHA) * np.array(last_colors[track_id]) + ALPHA * current_color
                last_colors[track_id] = smoothed_color.astype(int).tolist()
            else:
                last_colors[track_id] = current_color.tolist()

        color = last_colors.get(track_id, [255, 255, 255])
        b, g, r = color

        cv2.ellipse(proc_frame, center=(int((x1 + x2) / 2), y2), axes=(int((x2 - x1) / 2), int((y2 - y1) * 0.1)),
                    angle=0, startAngle=0, endAngle=360, color=(int(b), int(g), int(r)), thickness=2)
        cv2.putText(proc_frame, f"ID:{track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (int(b), int(g), int(r)), 2)

        frame_objects.append({
            "id": int(track_id),
            "type": "person",
            "point": [feet_x, feet_y],
            "color": color
        })

    balls_det = detections[detections.class_id == 32]

    if len(balls_det) > 0:
        best_ball_idx = np.argmax(balls_det.confidence)
        bx1, by1, bx2, by2 = balls_det.xyxy[best_ball_idx].astype(int)

        ball_x = int((bx1 + bx2) / 2)
        ball_y = int((by1 + by2) / 2)

        real_ball_x = int(ball_x / scale_ratio)
        real_ball_y = int(ball_y / scale_ratio)
        current_ball_pos = [real_ball_x, real_ball_y]

        cv2.circle(proc_frame, (ball_x, ball_y), 6, (0, 0, 255), -1)
        cv2.circle(proc_frame, (ball_x, ball_y), 8, (255, 255, 255), 2)
        cv2.putText(proc_frame, "Pilka", (ball_x - 15, ball_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    ball_positions.append({
        "frame_id": frame_id,
        "ball_x": current_ball_pos[0],
        "ball_y": current_ball_pos[1]
    })

    pitch_results = pitch_model(proc_frame, conf=0.5, verbose=False)[0]

    if pitch_results.keypoints is not None and len(pitch_results.keypoints.xy[0]) > 0:
        # Pobieramy współrzędne punktów
        pitch_keypoints = pitch_results.keypoints.xy[0].cpu().numpy()

        # Iterujemy przez każdy wykryty punkt i rysujemy go na klatce
        for point_idx, pt in enumerate(pitch_keypoints):
            kx, ky = int(pt[0]), int(pt[1])

            # YOLO-Pose czasami zwraca [0, 0] dla punktów, których nie widzi w kadrze.
            # Rysujemy tylko te, które są większe od 0.
            if kx > 0 and ky > 0:
                # Rysowanie wyraźnej magentowej kropki
                cv2.circle(proc_frame, (kx, ky), 6, (255, 0, 255), -1)
                # Opcjonalnie: Numer punktu (przydatne do debugowania i układania mapy 2D)
                cv2.putText(proc_frame, f"P{point_idx}", (kx + 8, ky - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


    if frame_objects:
        match_data.append({"frame_id": frame_id, "objects": frame_objects})

    cv2.imshow("Detekcja", proc_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()


df_ball = pd.DataFrame(ball_positions)

df_ball['ball_x'] = df_ball['ball_x'].interpolate(method='quadratic').bfill().ffill()
df_ball['ball_y'] = df_ball['ball_y'].interpolate(method='quadratic').bfill().ffill()

ball_dict = df_ball.set_index('frame_id').to_dict('index')

for frame_data in match_data:
    fid = frame_data["frame_id"]
    if fid in ball_dict:
        bx = ball_dict[fid]['ball_x']
        by = ball_dict[fid]['ball_y']

        if pd.notna(bx) and pd.notna(by):
            frame_data["objects"].append({
                "id": 999,
                "type": "ball",
                "point": [int(bx), int(by)]
            })

print("[INFO] Zapisywanie danych do pliku match_data.json...")
with open('match_data.json', 'w') as f:
    json.dump(match_data, f, indent=4, cls=NumpyEncoder)
print("[INFO] Gotowe! Piłka dodana do boiska.")
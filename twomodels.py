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


FIELD_POINTS = {
    0: (0.0, 0.0), 1: (0.0, 13.84), 2: (0.0, 24.84), 3: (0.0, 43.16),
    4: (0.0, 54.16), 5: (0.0, 68.0), 6: (5.5, 24.84), 7: (5.5, 43.16),
    8: (11.0, 34.0), 9: (16.5, 13.84), 10: (16.5, 26.69), 11: (16.5, 41.31),
    12: (16.5, 54.16), 13: (52.5, 0.0), 14: (52.5, 24.85), 15: (52.5, 43.15),
    16: (52.5, 68.0), 17: (88.5, 13.84), 18: (88.5, 26.69), 19: (88.5, 41.31),
    20: (88.5, 54.16), 21: (94.0, 34.0), 22: (99.5, 24.84), 23: (99.5, 43.16),
    24: (105.0, 0.0), 25: (105.0, 13.84), 26: (105.0, 24.84), 27: (105.0, 43.16),
    28: (105.0, 54.16), 29: (105.0, 68.0), 30: (43.35, 34.0), 31: (61.65, 34.0)
}


def compute_homography(kp_xy, kp_conf, conf_thresh=0.5):
    image_points = []
    world_points = []

    for idx, (pt, conf) in enumerate(zip(kp_xy, kp_conf)):
        if conf < conf_thresh: continue
        if idx not in FIELD_POINTS: continue
        x, y = pt
        if x <= 0 or y <= 0: continue
        image_points.append([x, y])
        world_points.append(FIELD_POINTS[idx])

    if len(image_points) < 4: return None
    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)
    H, _ = cv2.findHomography(image_points, world_points, cv2.RANSAC, 5.0)
    return H


def project_point(point, H):
    if H is None: return None
    pt = np.array([[point]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, H)
    return transformed[0][0]


def create_radar_background(scale=10):
    w, h = int(105 * scale), int(68 * scale)
    bg = np.full((h, w, 3), (34, 139, 34), dtype=np.uint8)
    color = (255, 255, 255)
    thickness = 2
    cv2.rectangle(bg, (0, 0), (w, h), color, thickness)
    cv2.line(bg, (w // 2, 0), (w // 2, h), color, thickness)
    cv2.circle(bg, (w // 2, h // 2), int(9.15 * scale), color, thickness)
    cv2.circle(bg, (w // 2, h // 2), 2, color, -1)
    cv2.rectangle(bg, (0, int(13.84 * scale)), (int(16.5 * scale), int(54.16 * scale)), color, thickness)
    cv2.rectangle(bg, (0, int(24.84 * scale)), (int(5.5 * scale), int(43.16 * scale)), color, thickness)
    cv2.rectangle(bg, (w - int(16.5 * scale), int(13.84 * scale)), (w, int(54.16 * scale)), color, thickness)
    cv2.rectangle(bg, (w - int(5.5 * scale), int(24.84 * scale)), (w, int(43.16 * scale)), color, thickness)
    return bg


def get_dominant_color(crop):
    h, w = crop.shape[:2]
    chest = crop[int(h * 0.20):int(h * 0.55), :]
    if chest.size == 0: return np.array([0, 0, 0])
    hsv = cv2.cvtColor(chest, cv2.COLOR_BGR2HSV)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    non_green_mask = cv2.bitwise_not(green_mask)
    pixels = chest[non_green_mask == 255]
    if len(pixels) < 10: return np.array([0, 0, 0])
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
spatial_data = []
ball_positions = []
ALPHA = 0.2
homography_matrix = None

# --- BAZA WIEDZY DO PŁYNNOŚCI ---
player_history = {}
player_speeds = {}
player_positions_smooth = {}  # NOWOŚĆ: Pamięć do wygładzania ruchów na radarze
FPS = 25
SPEED_WINDOW = 25
# --------------------------------

RADAR_SCALE = 6
radar_bg = create_radar_background(scale=RADAR_SCALE)

team_kmeans = None
team_colors_centers = None

possession_frames = {0: 0, 1: 0}
team_passes = {0: 0, 1: 0}

last_possessing_team = None
last_touch_team = None
last_touch_player_id = None
last_touch_pos = None
loose_ball_frames = 0

POSSESSION_DISTANCE_THRESHOLD = 2.5
MIN_PASS_DISTANCE = 3.0

for frame_id, frame in enumerate(generator, start=1):
    frame_objects = []
    current_ball_pos = [None, None]

    h, w = frame.shape[:2]
    scale_ratio = 1280 / w if w > 1280 else 1.0

    if scale_ratio != 1.0:
        proc_frame = cv2.resize(frame, (1280, int(h * scale_ratio)))
    else:
        proc_frame = frame.copy()

    current_radar = radar_bg.copy()

    pitch_results = pitch_model(proc_frame, conf=0.5, verbose=False)[0]
    if pitch_results.keypoints is not None and len(pitch_results.keypoints.xy[0]) > 0:
        kp_xy = pitch_results.keypoints.xy[0].cpu().numpy()
        kp_conf = pitch_results.keypoints.conf[0].cpu().numpy()
        homography_matrix = compute_homography(kp_xy, kp_conf)

    results = model(proc_frame, imgsz=1280, conf=0.05, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)

    players_det = detections[detections.class_id == 0]
    tracked_players = tracker.update_with_detections(players_det)

    team_0_positions = []
    team_1_positions = []

    for box, track_id in zip(tracked_players.xyxy, tracked_players.tracker_id):
        x1, y1, x2, y2 = box.astype(int)

        feet_x_proc = int((x1 + x2) / 2)
        feet_y_proc = int(y2)
        field_pos = project_point((feet_x_proc, feet_y_proc), homography_matrix)

        if field_pos is None: continue
        raw_fx, raw_fy = field_pos
        if not (0 <= raw_fx <= 105 and 0 <= raw_fy <= 68): continue

        # --- WYGŁADZANIE PRZESTRZENNE (Eliminacja skoków radaru) ---
        if track_id in player_positions_smooth:
            old_fx, old_fy = player_positions_smooth[track_id]
            # Używamy tylko 25% nowej pozycji i zachowujemy 75% starej, co pożera "skoki"
            fx = 0.25 * raw_fx + 0.75 * old_fx
            fy = 0.25 * raw_fy + 0.75 * old_fy
        else:
            fx, fy = raw_fx, raw_fy

        player_positions_smooth[track_id] = (fx, fy)
        # ------------------------------------------------------------

        feet_x_original = int(feet_x_proc / scale_ratio)
        feet_y_original = int(feet_y_proc / scale_ratio)

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
        draw_color = (int(b), int(g), int(r))  # Domyślny kolor przed przypisaniem

        team_id = None
        if team_kmeans is None and len(last_colors) >= 10:
            colors_array = np.array(list(last_colors.values()))
            team_kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
            team_kmeans.fit(colors_array)
            team_colors_centers = team_kmeans.cluster_centers_

        if team_kmeans is not None:
            team_id = team_kmeans.predict([color])[0]
            if team_id == 0:
                team_0_positions.append((fx, fy))
                # Zastępujemy brudny kolor czystym kolorem drużyny 1
                draw_color = tuple(map(int, team_colors_centers[0]))
            else:
                team_1_positions.append((fx, fy))
                # Zastępujemy brudny kolor czystym kolorem drużyny 2
                draw_color = tuple(map(int, team_colors_centers[1]))

        # --- STABILNE OBLICZANIE PRĘDKOŚCI ---
        if track_id not in player_history:
            player_history[track_id] = []
            player_speeds[track_id] = 0.0

        player_history[track_id].append((fx, fy, frame_id))

        if len(player_history[track_id]) > SPEED_WINDOW:
            player_history[track_id].pop(0)

        if len(player_history[track_id]) >= 10:
            past_points = player_history[track_id][:5]
            avg_past_x = sum(p[0] for p in past_points) / len(past_points)
            avg_past_y = sum(p[1] for p in past_points) / len(past_points)
            past_frame = past_points[len(past_points) // 2][2]

            recent_points = player_history[track_id][-5:]
            avg_curr_x = sum(p[0] for p in recent_points) / len(recent_points)
            avg_curr_y = sum(p[1] for p in recent_points) / len(recent_points)
            curr_frame = recent_points[len(recent_points) // 2][2]

            dist_meters = np.hypot(avg_curr_x - avg_past_x, avg_curr_y - avg_past_y)
            time_seconds = (curr_frame - past_frame) / FPS

            if time_seconds > 0:
                raw_speed_kmh = (dist_meters / time_seconds) * 3.6

                if raw_speed_kmh > 37.0:
                    raw_speed_kmh = player_speeds[track_id]

                if raw_speed_kmh < 2.5:
                    raw_speed_kmh = 0.0

                player_speeds[track_id] = (0.15 * raw_speed_kmh) + (0.85 * player_speeds[track_id])

        speed_kmh = player_speeds.get(track_id, 0.0)
        # --------------------------------------

        # Wideo AR: Używamy czystego koloru (draw_color)
        cv2.ellipse(proc_frame, center=(feet_x_proc, y2), axes=(int((x2 - x1) / 2), int((y2 - y1) * 0.1)),
                    angle=0, startAngle=0, endAngle=360, color=draw_color, thickness=2)

        # Wideo AR: Pigułki
        text_id = f"ID:{track_id}"
        text_speed = f"{speed_kmh:.1f} km/h"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45

        (w_id, h_id), _ = cv2.getTextSize(text_id, font, font_scale, 1)
        (w_sp, h_sp), _ = cv2.getTextSize(text_speed, font, font_scale - 0.05, 1)

        max_width = max(w_id, w_sp)
        padding_h = 4

        box_x1 = x1
        box_y2 = y1 - 5
        box_y1 = box_y2 - (h_id + h_sp + padding_h + 8)
        box_x2 = x1 + max_width + 12

        cv2.rectangle(proc_frame, (box_x1, box_y1), (box_x2, box_y2), (30, 30, 30), -1)
        # Boczny pasek też zyskuje czysty kolor
        cv2.rectangle(proc_frame, (box_x1, box_y1), (box_x1 + 4, box_y2), draw_color, -1)

        cv2.putText(proc_frame, text_id, (box_x1 + 8, box_y1 + h_id + 4), font, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
        cv2.putText(proc_frame, text_speed, (box_x1 + 8, box_y2 - 3), font, font_scale - 0.05, (200, 200, 200), 1,
                    cv2.LINE_AA)

        obj_data = {
            "id": int(track_id),
            "type": "person",
            "team": int(team_id) if team_id is not None else -1,
            "point": [feet_x_original, feet_y_original],
            "color": draw_color,  # Zapisujemy wyczyszczony kolor dla Dashboardu
            "field_position": [float(fx), float(fy)]
        }

        # Radar Live: Używamy czystego koloru + eleganckiej, białej obwódki zamiast czarnej
        rx, ry = int(fx * RADAR_SCALE), int(fy * RADAR_SCALE)
        cv2.circle(current_radar, (rx, ry), 8, draw_color, -1)
        cv2.circle(current_radar, (rx, ry), 8, (255, 255, 255), 1)

        frame_objects.append(obj_data)

    balls_det = detections[detections.class_id == 32]
    if len(balls_det) > 0:
        best_ball_idx = np.argmax(balls_det.confidence)
        bx1, by1, bx2, by2 = balls_det.xyxy[best_ball_idx].astype(int)

        ball_x_proc = int((bx1 + bx2) / 2)
        ball_y_proc = int((by1 + by2) / 2)
        ball_field_pos = project_point((ball_x_proc, ball_y_proc), homography_matrix)

        if ball_field_pos is not None:
            bfx, bfy = ball_field_pos
            if 0 <= bfx <= 105 and 0 <= bfy <= 68:
                real_ball_x = int(ball_x_proc / scale_ratio)
                real_ball_y = int(ball_y_proc / scale_ratio)
                current_ball_pos = [real_ball_x, real_ball_y]

                cv2.circle(proc_frame, (ball_x_proc, ball_y_proc), 6, (0, 0, 255), -1)
                cv2.circle(proc_frame, (ball_x_proc, ball_y_proc), 8, (255, 255, 255), 2)

                rbx, rby = int(bfx * RADAR_SCALE), int(bfy * RADAR_SCALE)
                cv2.circle(current_radar, (rbx, rby), 5, (0, 0, 255), -1)
                cv2.circle(current_radar, (rbx, rby), 6, (255, 255, 255), 1)

                if team_kmeans is not None:
                    min_dist = float('inf')
                    closest_team = None
                    closest_player_id = None

                    for obj in frame_objects:
                        if obj["type"] == "person" and obj["team"] != -1:
                            p_fx, p_fy = obj["field_position"]
                            dist = np.hypot(p_fx - bfx, p_fy - bfy)
                            if dist < min_dist:
                                min_dist = dist
                                closest_team = obj["team"]
                                closest_player_id = obj["id"]

                    if min_dist < POSSESSION_DISTANCE_THRESHOLD and closest_team is not None:
                        current_touch_pos = (bfx, bfy)

                        if last_touch_team == closest_team and last_touch_player_id != closest_player_id:
                            if last_touch_pos is not None:
                                dist_passed = np.hypot(current_touch_pos[0] - last_touch_pos[0],
                                                       current_touch_pos[1] - last_touch_pos[1])
                                if dist_passed > MIN_PASS_DISTANCE or loose_ball_frames > 10:
                                    team_passes[closest_team] += 1

                        last_possessing_team = closest_team
                        last_touch_team = closest_team
                        last_touch_player_id = closest_player_id
                        last_touch_pos = current_touch_pos
                        loose_ball_frames = 0
                    else:
                        loose_ball_frames += 1

    if last_possessing_team is not None:
        possession_frames[last_possessing_team] += 1

    t0_center_x, t0_center_y = None, None
    t1_center_x, t1_center_y = None, None

    if team_kmeans is not None:
        c0 = tuple(map(int, team_colors_centers[0]))
        c1 = tuple(map(int, team_colors_centers[1]))

        total_possession = possession_frames[0] + possession_frames[1]
        pct_0 = int((possession_frames[0] / total_possession) * 100) if total_possession > 0 else 0
        pct_1 = 100 - pct_0 if total_possession > 0 else 0

        # --- NOWOCZESNY PANEL STATYSTYK ---
        overlay = proc_frame.copy()

        cv2.rectangle(overlay, (15, 15), (420, 150), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.8, proc_frame, 0.2, 0, proc_frame)

        cv2.putText(proc_frame, "STATYSTYKI MECZOWE", (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.line(proc_frame, (30, 55), (400, 55), (100, 100, 100), 1)

        if team_0_positions:
            t0_center_x = sum(p[0] for p in team_0_positions) / len(team_0_positions)
            t0_center_y = sum(p[1] for p in team_0_positions) / len(team_0_positions)

            cv2.rectangle(proc_frame, (30, 75), (45, 90), c0, -1)
            cv2.putText(proc_frame, f"Druzyna 1: {pct_0}%   |   Podania: {team_passes[0]}", (60, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)

        if team_1_positions:
            t1_center_x = sum(p[0] for p in team_1_positions) / len(team_1_positions)
            t1_center_y = sum(p[1] for p in team_1_positions) / len(team_1_positions)

            cv2.rectangle(proc_frame, (30, 115), (45, 130), c1, -1)
            cv2.putText(proc_frame, f"Druzyna 2: {pct_1}%   |   Podania: {team_passes[1]}", (60, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)
        # ----------------------------------

    spatial_data.append({
        "frame_id": frame_id,
        "team_0_cm": [t0_center_x, t0_center_y] if t0_center_x else None,
        "team_1_cm": [t1_center_x, t1_center_y] if t1_center_x else None,
        "team_0_positions": team_0_positions,
        "team_1_positions": team_1_positions
    })

    ball_positions.append({
        "frame_id": frame_id,
        "ball_x": current_ball_pos[0],
        "ball_y": current_ball_pos[1]
    })

    if frame_objects: match_data.append({"frame_id": frame_id, "objects": frame_objects})

    cv2.imshow("Detekcja", proc_frame)
    cv2.imshow("Radar", current_radar)

    if cv2.waitKey(1) & 0xFF == ord("q"): break

cv2.destroyAllWindows()

df_ball = pd.DataFrame(ball_positions)
if not df_ball.empty and 'ball_x' in df_ball.columns and 'ball_y' in df_ball.columns:
    df_ball['ball_x'] = df_ball['ball_x'].interpolate(method='quadratic').bfill().ffill()
    df_ball['ball_y'] = df_ball['ball_y'].interpolate(method='quadratic').bfill().ffill()

    ball_dict = df_ball.set_index('frame_id').to_dict('index')

    for frame_data in match_data:
        fid = frame_data["frame_id"]
        if fid in ball_dict:
            bx = ball_dict[fid].get('ball_x')
            by = ball_dict[fid].get('ball_y')

            if pd.notna(bx) and pd.notna(by):
                frame_data["objects"].append({
                    "id": 999,
                    "type": "ball",
                    "point": [int(bx), int(by)]
                })

with open('spatial_data.json', 'w') as f:
    json.dump(spatial_data, f, cls=NumpyEncoder)

with open('match_data.json', 'w') as f:
    json.dump(match_data, f, cls=NumpyEncoder)

# --- ZAPIS KOŃCOWYCH STATYSTYK DLA DASHBOARDU ---
total_poss = possession_frames[0] + possession_frames[1]
final_stats = {
    "Team_1": {
        "possession": int((possession_frames[0] / total_poss) * 100) if total_poss > 0 else 0,
        "passes": team_passes[0]
    },
    "Team_2": {
        "possession": int((possession_frames[1] / total_poss) * 100) if total_poss > 0 else 0,
        "passes": team_passes[1]
    }
}

with open('match_stats.json', 'w') as f:
    json.dump(final_stats, f)
# ---------------------------------------------
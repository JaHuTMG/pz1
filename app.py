import sys
import streamlit as st
import subprocess
from PIL import Image
import json
import cv2
import numpy as np
import pandas as pd

st.set_page_config(page_title="PZ1 - Analiza Meczu", layout="wide", initial_sidebar_state="expanded")

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1857/1857612.png", width=100)  # Ikonka piłki
    st.title("Panel Sterowania")
    st.markdown("Zarządzaj analizą z tego miejsca.")

    st.divider()

    if st.button("🚀 Wygeneruj Raport Końcowy (Heatmapy)", use_container_width=True):
        with st.spinner("Trwa przeliczanie danych i rysowanie map ciepła..."):
            # Uruchamia skrypt Janka
            subprocess.run([sys.executable, "heatmap.py"])
        st.success("Heatmapy zaktualizowane pomyślnie!")

    st.info("Pamiętaj, aby przed wygenerowaniem raportu uruchomić główny skrypt detekcji (wideo).")

st.title("⚽ System Analizy Taktycznej (PZ1)")
st.markdown("Projekt zespołowy - Rzutowanie zawodników na radar 2D oraz analiza statystyczna.")

tab1, tab2, tab3, tab4 = st.tabs(["🎥 Wideo (AR)", "📡 Radar 2D (Live)", "🔥 Heatmapy (Strefy)", "📊 Statystyki Meczu"])

with tab1:
    st.header("Nagranie z meczu")
    st.markdown("Tutaj w przyszłości pojawi się wideo z nałożonymi statystykami AR.")
    st.info("Oczekiwanie na finalny plik wideo z modelu...")

with tab2:
    st.header("Dynamiczny Radar Boiska")

    try:
        with open('match_data.json', 'r') as f:
            match_data = json.load(f)

        FPS = 25  # Zakładamy 25 klatek na sekundę
        total_seconds = len(match_data) // FPS

        time_options = [f"{s // 60:02d}:{s % 60:02d}" for s in range(total_seconds + 1)]

        selected_time = st.select_slider(
            "Wybierz moment meczu",
            options=time_options
        )

        minutes, seconds = map(int, selected_time.split(":"))
        selected_frame = (minutes * 60 + seconds) * FPS

        if selected_frame >= len(match_data):
            selected_frame = len(match_data) - 1
        # --------------------------------

        frame_data = match_data[selected_frame]

        radar_img = np.zeros((400, 600, 3), dtype=np.uint8)
        radar_img[:] = (34, 139, 34)  # Zielone tło boiska
        cv2.rectangle(radar_img, (0, 0), (600, 400), (255, 255, 255), 2)  # Kontur
        cv2.line(radar_img, (300, 0), (300, 400), (255, 255, 255), 2)  # Połowa
        cv2.circle(radar_img, (300, 200), 50, (255, 255, 255), 2)  # Środek

        for obj in frame_data["objects"]:
            x, y = obj["point"]
            rx = int((x / 1280) * 600)
            ry = int((y / 720) * 400)

            if obj["type"] == "person":
                c = obj.get("color", [255, 255, 255])
                color = (int(c[0]), int(c[1]), int(c[2]))

                cv2.circle(radar_img, (rx, ry), 10, color, -1)
                cv2.circle(radar_img, (rx, ry), 10, (255, 255, 255), 1, cv2.LINE_AA)

                text = str(obj["id"])
                font = cv2.FONT_HERSHEY_SIMPLEX
                (text_w, text_h), _ = cv2.getTextSize(text, font, 0.4, 1)
                txt_x = rx - text_w // 2
                txt_y = ry - 14

                cv2.putText(radar_img, text, (txt_x + 1, txt_y + 1), font, 0.4, (20, 20, 20), 1, cv2.LINE_AA)
                cv2.putText(radar_img, text, (txt_x, txt_y), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            elif obj["type"] == "ball":
                cv2.circle(radar_img, (rx, ry), 6, (255, 255, 255), -1)
                cv2.circle(radar_img, (rx, ry), 6, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.circle(radar_img, (rx, ry), 2, (0, 0, 255), -1, cv2.LINE_AA)

        col1, col2, col3 = st.columns([1, 2, 1])  # Tworzymy 3 kolumny, środkowa jest najszersza
        with col2:
            # Wrzucamy obrazek do środkowej kolumny z narzuconą szerokością 700px
            st.image(radar_img, channels="BGR", width=700)

    except FileNotFoundError:
        st.warning("Brak pliku match_data.json. Uruchom detekcję, aby wygenerować dane radaru.")

with tab3:
    st.header("Analiza Przestrzenna Drużyn")
    try:
        heatmap_img = Image.open('team_heatmaps.png')
        st.image(heatmap_img, caption="Heatmapy wygenerowane z danych meczowych.", width=700)
    except FileNotFoundError:
        st.info("Wciśnij przycisk 'Wygeneruj Raport' w panelu bocznym, aby utworzyć heatmapy!")

with tab4:
    st.header("Podsumowanie Taktyczne")
    st.markdown("Kluczowe wskaźniki wydajności (KPI) obu drużyn wygenerowane przez sztuczną inteligencję.")
    st.divider()

    try:
        with open('match_stats.json', 'r') as f:
            stats = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟩 Drużyna 1")
            # Element st.metric tworzy ładną kartę z dużą liczbą
            st.metric(label="Liczba celnych podań", value=stats["Team_1"]["passes"])
            st.write(f"**Posiadanie piłki: {stats['Team_1']['possession']}%**")
            # Pasek postępu (przyjmuje wartości od 0.0 do 1.0)
            st.progress(stats["Team_1"]["possession"] / 100)

        with col2:
            st.subheader("⬜ Drużyna 2")
            st.metric(label="Liczba celnych podań", value=stats["Team_2"]["passes"])
            st.write(f"**Posiadanie piłki: {stats['Team_2']['possession']}%**")
            st.progress(stats["Team_2"]["possession"] / 100)

        st.divider()

        st.markdown("### Szczegółowe zestawienie")

        # Tworzymy DataFrame z danych
        df = pd.DataFrame({
            "Wskaźnik": ["Posiadanie piłki (%)", "Wymienione podania"],
            "Drużyna 1": [stats["Team_1"]["possession"], stats["Team_1"]["passes"]],
            "Drużyna 2": [stats["Team_2"]["possession"], stats["Team_2"]["passes"]]
        })

        st.dataframe(df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.info("Brak pliku statystyk. Pamiętaj, aby uruchomić skrypt detekcji wideo w celu wygenerowania danych!")
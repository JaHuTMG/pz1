import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def draw_pitch(ax):
    pitch_color = '#1e8f24'
    line_color = 'white'

    ax.set_facecolor(pitch_color)

    ax.plot([0, 105, 105, 0, 0], [0, 0, 68, 68, 0], color=line_color, linewidth=2)

    ax.plot([52.5, 52.5], [0, 68], color=line_color, linewidth=2)

    center_circle = plt.Circle((52.5, 34), 9.15, color=line_color, fill=False, linewidth=2)
    ax.add_patch(center_circle)
    ax.plot(52.5, 34, 'o', color=line_color)

    ax.plot([0, 16.5, 16.5, 0], [13.84, 13.84, 54.16, 54.16], color=line_color, linewidth=2)
    ax.plot([105, 88.5, 88.5, 105], [13.84, 13.84, 54.16, 54.16], color=line_color, linewidth=2)

    ax.plot([0, 5.5, 5.5, 0], [24.84, 24.84, 43.16, 43.16], color=line_color, linewidth=2)
    ax.plot([105, 99.5, 99.5, 105], [24.84, 24.84, 43.16, 43.16], color=line_color, linewidth=2)

    ax.set_xlim(-2, 107)
    ax.set_ylim(-2, 70)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.invert_yaxis()

print("Wczytywanie danych...")
try:
    with open('spatial_data.json', 'r') as f:
        spatial_data = json.load(f)
except FileNotFoundError:
    print("Nie znaleziono pliku spatial_data.json! Uruchom najpierw główny skrypt detekcji.")
    exit()

team_0_x, team_0_y = [], []
team_1_x, team_1_y = [], []

for frame in spatial_data:
    for pos in frame.get('team_0_positions', []):
        team_0_x.append(pos[0])
        team_0_y.append(pos[1])

    for pos in frame.get('team_1_positions', []):
        team_1_x.append(pos[0])
        team_1_y.append(pos[1])

if not team_0_x or not team_1_x:
    print("Brak wystarczających danych do wygenerowania mapy.")
    exit()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 14))
fig.canvas.manager.set_window_title('Analiza Przestrzenna - Heatmapy')
fig.suptitle('Mapy Ciepła (Heatmaps) - Ustawienie Drużyn', fontsize=18, fontweight='bold', color='white')
fig.patch.set_facecolor('#222222')

draw_pitch(ax1)
sns.kdeplot(
    x=team_0_x,
    y=team_0_y,
    ax=ax1,
    fill=True,
    cmap="Reds",
    alpha=0.6,
    thresh=0.1,
    levels=10
)
ax1.set_title("Drużyna 1 (Czerwona strefa aktywności)", fontsize=14, color='white', pad=10)

draw_pitch(ax2)
sns.kdeplot(
    x=team_1_x,
    y=team_1_y,
    ax=ax2,
    fill=True,
    cmap="Blues",
    alpha=0.6,
    thresh=0.1,
    levels=10
)
ax2.set_title("Drużyna 2 (Niebieska strefa aktywności)", fontsize=14, color='white', pad=10)

plt.tight_layout()
plt.subplots_adjust(top=0.92)

plt.savefig('team_heatmaps.png', facecolor=fig.get_facecolor(), dpi=300)
print("Zapisano wykres jako 'team_heatmaps.png'. Wyświetlanie okna...")
plt.show()
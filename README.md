## 🚀 Jak uruchomić projekt

Projekt składa się z dwóch głównych modułów: silnika detekcji AI (analiza wideo) oraz interaktywnego Dashboardu w Streamlit (wizualizacja wyników). Aby wszystko działało poprawnie, najpierw wygeneruj dane, a dopiero potem odpal interfejs graficzny.

Na początku upewnij się, że jesteś w głównym folderze projektu i aktywuj wirtualne środowisko:
```bash
venv\Scripts\activate
```
Następnie uruchom główny skrypt analityczny, który przetworzy wideo, rozpozna zawodników i wyliczy statystyki:
```bash
python twomodels.py
```

Ważne: Pozwól skryptowi działać przez dłuższą chwilę. Aby poprawnie zakończyć analizę i wymusić zapis plików danych (match_data.json, spatial_data.json, match_stats.json) na dysku, kliknij w okno z odtwarzanym wideo i wciśnij klawisz q na klawiaturze (unikaj zamykania okna systemowym krzyżykiem).

Gdy dane są gotowe, uruchom panel wizualizacyjny:
```bash
streamlit run app.py
```

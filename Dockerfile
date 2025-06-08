# Базалық образ
FROM python:3.10-slim

# Жұмыс каталогын жасау
WORKDIR /app

# Талап етілетін файлдарды көшіру
COPY requirements.txt .
COPY interface/ ./interface/
COPY models/ ./models/
COPY data/ ./data/

# Пакеттерді орнату
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8501


CMD ["streamlit", "run", "interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

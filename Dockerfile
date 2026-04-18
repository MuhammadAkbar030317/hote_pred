# 1️⃣ Base image
FROM python:3.11-slim

# 2️⃣ Work directory
WORKDIR /App

# 3️⃣ System dependencies (minimal)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4️⃣ Requirements copy
COPY requirements.txt .

# 5️⃣ Install python deps
RUN pip install --no-cache-dir -r requirements.txt

# 6️⃣ Project files copy
COPY . .

# 7️⃣ Expose FastAPI port
EXPOSE 8000

# 8️⃣ Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Gunakan Python versi 3.11.8 yang ringan
FROM python:3.11.8-slim

# Install library tingkat OS untuk OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

# Atur direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin seluruh kode aplikasi Anda (main.py, folder model, dll)
COPY . .

# Perintah menjalankan aplikasi web (Sesuaikan 'main:app' jika nama file Anda bukan main.py)
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "8000"]
import os
import time
import uuid
import gc
import json
from datetime import datetime
from tensorflow.keras import backend as K

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

from utils.validator import validate_video
from utils.predictor import predict_video


app = Flask(__name__)


UPLOAD_FOLDER = os.path.join(app.root_path, "static", "uploads")
ALLOWED_EXTENSIONS = {"mp4"}
# Variabel untuk menunjuk lokasi file JSON penyimpan riwayat
HISTORY_FILE = os.path.join(app.root_path, 'history.json')

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def save_to_history(filename, label, confidence):
    """
    Menyimpan hasil analisis video ke dalam file JSON history.json
    """
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
            
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    except json.JSONDecodeError:
        history = []
        
    # Tambahkan di index 0 agar data terbaru selalu berada di paling atas
    new_entry = {
        "waktu": datetime.now().strftime("%d %b %Y, %H:%M"),
        "nama_file": filename,
        "hasil": label,
        "skor": f"{confidence:.1f}%"
    }
    history.insert(0, new_entry)
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f)


def allowed_file(filename):
    """
    Memeriksa apakah ekstensi file yang diunggah adalah MP4.
    """
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def show_error(title, message, icon="⚠", status_code=400):
    """
    Menampilkan halaman error VerDeep dengan pesan ramah pengguna.
    """
    return (
        render_template(
            "error.html",
            error_title=title,
            error_icon=icon,
            error_message=message,
            error_description=""
        ),
        status_code
    )


def remove_uploaded_file(file_path):
    """
    Menghapus file upload apabila proses validasi atau prediksi gagal.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
    except OSError:
        pass


def get_user_friendly_prediction_error(error):
    """
    Mengonversi exception internal menjadi pesan yang aman dan mudah dipahami.
    """
    raw_message = str(error).lower()

    no_face_markers = (
        "wajah tidak ditemukan",
        "wajah tidak terdeteksi",
        "face not found",
        "no face",
        "no faces",
        "empty output",
        "output shape: (0,",
        "shape=(0,",
        "conv2d.call",
        "convolution operation resulted in an empty output",
    )

    if any(marker in raw_message for marker in no_face_markers):
        return (
            "Wajah tidak terdeteksi pada video yang diunggah. "
            "Silakan gunakan video yang menampilkan wajah dengan jelas, "
            "memiliki pencahayaan cukup, dan wajah tidak tertutup atau "
            "terlalu kecil."
        )

    video_read_markers = (
        "video tidak dapat dibuka",
        "cannot open video",
        "failed to open",
        "gagal membaca frame",
        "tidak memiliki frame",
        "frame tidak valid",
    )

    if any(marker in raw_message for marker in video_read_markers):
        return (
            "Video tidak dapat dibaca dengan baik. "
            "Silakan pastikan file MP4 tidak rusak dan coba unggah kembali."
        )

    return (
        "Video tidak dapat dianalisis karena terjadi kendala saat pemrosesan. "
        "Silakan gunakan video MP4 lain yang menampilkan wajah dengan jelas."
    )


def build_result(prediction):
    """
    Menyatukan keluaran predictor.py untuk result.html.
    """
    if not isinstance(prediction, dict):
        return (
            None,
            [],
            {},
            "Format hasil prediksi model tidak dikenali."
        )

    analyzed_frames = prediction.get("frames", [])

    if not isinstance(analyzed_frames, list):
        analyzed_frames = []

    video_info = prediction.get("video_info", {})

    if not isinstance(video_info, dict):
        video_info = {}

    if prediction.get("error"):
        raw_message = prediction.get(
            "message",
            "Preprocessing video tidak dapat menghasilkan data yang diperlukan."
        )

        friendly_message = get_user_friendly_prediction_error(
            ValueError(raw_message)
        )

        return (
            None,
            analyzed_frames,
            video_info,
            friendly_message
        )

    raw_deepfake_score = prediction.get(
        "probability_deepfake",
        prediction.get("score", 0.0)
    )

    try:
        deepfake_score = float(raw_deepfake_score)
    except (TypeError, ValueError):
        return (
            None,
            [],
            {},
            "Skor prediksi model tidak dapat dibaca."
        )

    deepfake_score = max(0.0, min(deepfake_score, 1.0))

    label = prediction.get("label")

    if label not in ("Real", "Deepfake"):
        label = (
            "Deepfake"
            if deepfake_score >= 0.5
            else "Real"
        )

    result_class = prediction.get("result_class")

    if result_class not in ("real", "deepfake"):
        result_class = (
            "deepfake"
            if label == "Deepfake"
            else "real"
        )

    # Perbaikan bug penulisan perhitungan confidence
    confidence_value = deepfake_score if result_class == "deepfake" else (1.0 - deepfake_score)

    result = {
        "label": label,
        "score": deepfake_score,
        "confidence": confidence_value * 100,
        "result_class": result_class,
        "frame_count": prediction.get(
            "frame_count",
            len(analyzed_frames)
        )
    }

    return result, analyzed_frames, video_info, None


@app.route("/")
def home():
    """
    Halaman beranda VerDeep.
    """
    return render_template("beranda.html")

@app.route('/riwayat')
def riwayat():
    """
    Menampilkan halaman log/riwayat video yang telah diuji.
    """
    history_data = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
        except json.JSONDecodeError:
            history_data = []
            
    return render_template('riwayat.html', riwayat=history_data)

@app.route('/hapus_riwayat/<int:index>', methods=['POST'])
def hapus_riwayat(index):
    """
    Menghapus satu baris data spesifik pada riwayat berdasarkan indeksnya.
    """
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
                
            if 0 <= index < len(history):
                history.pop(index)
                
                with open(HISTORY_FILE, 'w') as f:
                    json.dump(history, f)
        except Exception as e:
            app.logger.error(f"Gagal menghapus riwayat: {e}")
            
    return redirect(url_for('riwayat'))

@app.route('/arsitektur')
def arsitektur():
    return render_template('arsitektur.html')

@app.route('/kontak')
def kontak():
    return render_template('kontak.html')


@app.route("/analisis", methods=["GET", "POST"])
def index():
    """
    Menampilkan halaman upload dan memproses video MP4.
    """
    if request.method == "GET":
        return render_template("index.html")

    if "video" not in request.files:
        return show_error(
            "File Video Tidak Ditemukan",
            "Form upload tidak menerima file video.",
            "📁"
        )

    file = request.files["video"]

    if file.filename == "":
        return show_error(
            "Video Belum Dipilih",
            "Silakan pilih file video terlebih dahulu.",
            "📁"
        )

    if not allowed_file(file.filename):
        return show_error(
            "Format Video Tidak Didukung",
            "Gunakan file video dengan format MP4.",
            "📄"
        )

    original_filename = secure_filename(file.filename)

    if not original_filename:
        return show_error(
            "Nama File Tidak Valid",
            "Nama file video tidak dapat diproses. Silakan ubah nama file lalu coba kembali.",
            "📄"
        )

    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        unique_filename
    )

    try:
        file.save(file_path)

    except Exception:
        return show_error(
            "Upload Gagal",
            "File video tidak dapat disimpan oleh sistem. Silakan coba kembali.",
            "✕",
            500
        )

    is_valid, error_message = validate_video(file_path)

    if not is_valid:
        remove_uploaded_file(file_path)

        return show_error(
            "Video Tidak Dapat Diproses",
            error_message,
            "⚠"
        )

    try:
        start_time = time.perf_counter()

        # Proses AI berjalan di sini
        prediction = predict_video(file_path)

        processing_time = time.perf_counter() - start_time
        
        # --- KODE PEMBERSIH MEMORI (GARBAGE COLLECTOR) ---
        gc.collect()

    except ValueError as error:
        remove_uploaded_file(file_path)
        
        # Bersihkan memori juga jika terjadi error saat prediksi
        gc.collect()

        return show_error(
            "Video Tidak Dapat Diproses",
            get_user_friendly_prediction_error(error),
            "⚠"
        )

    except Exception as error:
        app.logger.exception(
            "Kesalahan internal saat memprediksi video: %s",
            error
        )

        remove_uploaded_file(file_path)
        
        # Bersihkan memori juga jika terjadi error saat prediksi
        K.clear_session()
        gc.collect()

        return show_error(
            "Proses Prediksi Gagal",
            (
                "Terjadi kendala teknis saat menganalisis video. "
                "Silakan coba kembali beberapa saat lagi."
            ),
            "✕",
            500
        )

    result, analyzed_frames, video_info, prediction_error = build_result(
        prediction
    )

    if prediction_error:
        remove_uploaded_file(file_path)

        return show_error(
            "Video Tidak Dapat Diproses",
            prediction_error,
            "⚠"
        )
        
    # --- BAGIAN PENAMBAHAN DATA KE RIWAYAT ---
    # Jika prediksi berhasil dan tidak ada error, simpan data ke JSON history
    label_to_save = result.get('label', 'Unknown')
    # Jika labelnya 'Deepfake', mungkin Anda ingin menyimpannya sebagai 'Palsu (Deepfake)' agar sesuai UI
    if label_to_save == "Deepfake":
        label_to_save = "Palsu (Deepfake)"
    elif label_to_save == "Real":
        label_to_save = "Asli"
        
    confidence_to_save = result.get('confidence', 0.0)
    
    # Panggil fungsi simpan riwayat
    save_to_history(original_filename, label_to_save, confidence_to_save)
    # -----------------------------------------

    return render_template(
        "result.html",
        result=result,
        video_filename=unique_filename,
        video_url=f"uploads/{unique_filename}",
        analyzed_frames=analyzed_frames,
        video_info=video_info,
        processing_time=round(processing_time, 2)
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
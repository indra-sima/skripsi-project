import cv2
from moviepy import VideoFileClip


MAX_WIDTH = 1920
MAX_HEIGHT = 1080
MAX_DURATION = 30


def validate_duration(video_path, max_duration=MAX_DURATION):
    """
    Memastikan durasi video valid dan tidak melebihi batas.
    Return: (True, None) jika valid, atau (False, pesan_error) jika tidak.
    """
    clip = None

    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration

        if duration is None or duration <= 0:
            return False, "Video tidak memiliki durasi yang valid."

        if duration > max_duration:
            return False, (
                f"Durasi video adalah {duration:.1f} detik. "
                f"Batas maksimum adalah {max_duration} detik."
            )

        return True, None

    except Exception:
        return False, (
            "Video tidak dapat dibaca untuk pemeriksaan durasi. "
            "Pastikan file video tidak rusak dan menggunakan format yang didukung."
        )

    finally:
        if clip is not None:
            clip.close()


def validate_resolution(
    video_path,
    max_width=MAX_WIDTH,
    max_height=MAX_HEIGHT
):
    """
    Memastikan resolusi video tidak lebih dari 1920x1080.
    Return: (True, None) jika valid, atau (False, pesan_error) jika tidak.
    """
    cap = cv2.VideoCapture(video_path)

    try:
        if not cap.isOpened():
            return False, (
                "Video tidak dapat dibaca. "
                "Pastikan file video valid dan tidak rusak."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            return False, "Resolusi video tidak dapat dibaca."

        if width > max_width or height > max_height:
            return False, (
                f"Resolusi video adalah {width}×{height}. "
                f"Batas maksimum yang diizinkan adalah "
                f"{max_width}×{max_height}."
            )

        return True, None

    except Exception:
        return False, "Terjadi kesalahan saat memeriksa resolusi video."

    finally:
        cap.release()


def validate_video(video_path):
    """
    Menjalankan semua validasi metadata video:
    1. Video dapat dibaca.
    2. Resolusi maksimal 1920x1080.
    3. Durasi maksimal 60 detik.

    Return: (True, None) jika semua valid,
            (False, pesan_error) jika salah satu validasi gagal.
    """
    is_valid, message = validate_resolution(video_path)

    if not is_valid:
        return False, message

    is_valid, message = validate_duration(video_path)

    if not is_valid:
        return False, message

    return True, None

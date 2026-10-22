import os
import shutil
import uuid

import cv2
import numpy as np
import tensorflow as tf
from mtcnn import MTCNN


IMG_SIZE = 224
SEQ_LEN = 60

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FRAMES_ROOT = os.path.join(
    PROJECT_ROOT,
    "static",
    "uploads",
    "frames"
)

detector = MTCNN()


def _create_frame_directory():
    os.makedirs(FRAMES_ROOT, exist_ok=True)

    analysis_id = uuid.uuid4().hex

    frame_directory = os.path.join(
        FRAMES_ROOT,
        analysis_id
    )

    os.makedirs(
        frame_directory,
        exist_ok=False
    )

    return analysis_id, frame_directory


def prepare_video(video_path):
    """
    Pipeline dibuat sama dengan inferensi Colab:

    Video -> 60 indeks np.linspace -> BGR ke RGB -> MTCNN
    -> wajah confidence tertinggi -> crop -> resize 224x224
    -> simpan JPEG -> TensorFlow decode JPEG -> float32
    -> tensor (1, 60, 224, 224, 3).

    Tidak melakukan normalisasi /255.0.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        raise ValueError(
            "Video tidak dapat dibaca. Pastikan file valid."
        )

    frame_directory = None

    try:
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        fps = float(
            cap.get(cv2.CAP_PROP_FPS) or 0.0
        )

        width = int(
            cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        if total_frames <= 0:
            raise ValueError(
                "Video tidak memiliki frame yang valid."
            )

        duration_seconds = (
            total_frames / fps
            if fps > 0
            else 0.0
        )

        video_info = {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "duration_seconds": round(
                duration_seconds,
                2
            ),
            "width": width,
            "height": height
        }

        indices = np.linspace(
            0,
            max(total_frames - 1, 0),
            SEQ_LEN,
            dtype=int
        )

        analysis_id, frame_directory = (
            _create_frame_directory()
        )

        saved_frame_paths = []
        analyzed_frames = []

        for out_idx, frame_idx in enumerate(
            indices,
            start=1
        ):
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(frame_idx)
            )

            ret, frame = cap.read()

            if not ret:
                continue

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            faces = detector.detect_faces(rgb)

            if len(faces) == 0:
                continue

            face = max(
                faces,
                key=lambda detected_face: (
                    detected_face["confidence"]
                )
            )

            x, y, w, h = face["box"]

            x = max(0, int(x))
            y = max(0, int(y))

            x2 = min(
                rgb.shape[1],
                x + int(w)
            )

            y2 = min(
                rgb.shape[0],
                y + int(h)
            )

            face_crop = rgb[y:y2, x:x2]

            if face_crop.size == 0:
                continue

            face_crop = cv2.resize(
                face_crop,
                (IMG_SIZE, IMG_SIZE)
            )

            filename = f"frame_{out_idx:03d}.jpg"

            output_path = os.path.join(
                frame_directory,
                filename
            )

            saved = cv2.imwrite(
                output_path,
                cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_RGB2BGR
                )
            )

            if not saved:
                raise RuntimeError(
                    f"Gagal menyimpan frame {filename}."
                )

            saved_frame_paths.append(output_path)

            analyzed_frames.append(
                {
                    "url": (
                        f"uploads/frames/"
                        f"{analysis_id}/"
                        f"{filename}"
                    ),
                    "source_frame": int(frame_idx)
                }
            )

        if len(saved_frame_paths) == 0:
            raise ValueError(
                "Tidak ada wajah terdeteksi pada video."
            )

        if len(saved_frame_paths) != SEQ_LEN:
            raise ValueError(
                f"Hanya {len(saved_frame_paths)} dari "
                f"{SEQ_LEN} frame yang berhasil "
                "mendeteksi wajah."
            )

        frames = []

        for path in saved_frame_paths:
            image = tf.io.read_file(path)

            image = tf.image.decode_jpeg(
                image,
                channels=3
            )

            image = tf.image.resize(
                image,
                [IMG_SIZE, IMG_SIZE]
            )

            image = tf.cast(
                image,
                tf.float32
            )

            frames.append(image)

        frames = tf.stack(
            frames,
            axis=0
        )

        model_input = tf.expand_dims(
            frames,
            axis=0
        )

        return (
            model_input,
            analyzed_frames,
            video_info
        )

    except Exception:
        if (
            frame_directory
            and os.path.exists(frame_directory)
        ):
            shutil.rmtree(
                frame_directory,
                ignore_errors=True
            )

        raise

    finally:
        cap.release()
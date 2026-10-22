from utils.preprocess import prepare_video
from rebuild_model import model


THRESHOLD = 0.5


def predict_video(video_path):
    """
    Output model mengikuti Colab:
    nilai output = probabilitas Deepfake.
    """
    data, analyzed_frames, video_info = prepare_video(
        video_path
    )

    n_frames = len(analyzed_frames)
    expected_frames = model.input_shape[1]

    if (
        expected_frames is not None
        and n_frames != expected_frames
    ):
        return {
            "error": True,
            "detected_faces": n_frames,
            "message": (
                f"Hanya {n_frames} dari "
                f"{expected_frames} frame yang berhasil "
                "disiapkan untuk model."
            ),
            "frames": analyzed_frames,
            "video_info": video_info
        }

    probability = float(
        model.predict(data, verbose=0)[0][0]
    )

    label = (
        "Deepfake"
        if probability >= THRESHOLD
        else "Real"
    )

    confidence = (
        probability
        if label == "Deepfake"
        else 1.0 - probability
    )

    return {
        "error": False,
        "label": label,

        # Nilai sama dengan output mentah Colab.
        "probability": probability,
        "probability_deepfake": probability,
        "probability_real": 1.0 - probability,

        # Untuk tampilan keyakinan kelas terpilih.
        "score": confidence,
        "confidence": confidence * 100,

        "result_class": (
            "deepfake"
            if label == "Deepfake"
            else "real"
        ),

        "frames": analyzed_frames,
        "frame_count": n_frames,
        "video_info": video_info
    }
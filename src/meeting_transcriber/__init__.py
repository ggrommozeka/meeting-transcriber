"""meeting-transcriber: fully local meeting transcription pipeline."""

__version__ = "0.1.0"


def transcribe_audio(*args, **kwargs):
    """Transcribe an audio/video input without using its video frames."""

    from .api import transcribe_audio as implementation

    return implementation(*args, **kwargs)


def transcribe_video(*args, **kwargs):
    """Transcribe a video and OCR slides/code shown on screen."""

    from .api import transcribe_video as implementation

    return implementation(*args, **kwargs)


__all__ = ["transcribe_audio", "transcribe_video"]

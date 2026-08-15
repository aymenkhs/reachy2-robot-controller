"""Post-process OpenAI PCM speech into a metallic / cute-robot voice."""

from __future__ import annotations

import numpy as np
from scipy import signal

from config import (
    BITCRUSH_BITS,
    METALLIC_GAIN_DB,
    PITCH_SEMITONES,
    RING_HZ,
    RING_MIX,
    WET,
)


def pcm16_to_float(raw: bytes) -> np.ndarray:
    """Convert little-endian int16 PCM bytes to float32 in [-1, 1]."""
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return (
        np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    )


def float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 audio in [-1, 1] to little-endian int16 PCM bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def _pitch_shift(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Resample pitch-up (slightly faster — fine for a cute robot)."""
    if abs(semitones) < 1e-3:
        return audio
    factor = 2.0 ** (semitones / 12.0)
    n = len(audio)
    x_old = np.arange(n, dtype=np.float64)
    x_new = np.arange(0, n, factor, dtype=np.float64)
    shifted = np.interp(x_new, x_old, audio).astype(np.float32)
    if len(shifted) < n:
        shifted = np.pad(shifted, (0, n - len(shifted)))
    return shifted[:n]


def _ring_mod(audio: np.ndarray, sr: int, hz: float, mix: float) -> np.ndarray:
    t = np.arange(len(audio), dtype=np.float32) / sr
    carrier = np.sin(2.0 * np.pi * hz * t)
    modulated = audio * (0.65 + 0.35 * carrier)
    return (1.0 - mix) * audio + mix * modulated


def _bitcrush(audio: np.ndarray, bits: float) -> np.ndarray:
    levels = max(2.0, 2.0 ** bits)
    return np.round(audio * (levels * 0.5)) / (levels * 0.5)


def _metallic_eq(audio: np.ndarray, sr: int, gain_db: float) -> np.ndarray:
    if abs(gain_db) < 1e-3:
        return audio
    center, q = 3800.0, 1.1
    gain = 10.0 ** (gain_db / 20.0)
    w0 = 2.0 * np.pi * center / sr
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha / gain
    b = np.array(
        [(1.0 + alpha * gain) / a0, (-2.0 * np.cos(w0)) / a0, (1.0 - alpha * gain) / a0]
    )
    a = np.array([1.0, (-2.0 * np.cos(w0)) / a0, (1.0 - alpha / gain) / a0])
    return signal.lfilter(b, a, audio).astype(np.float32)


def cute_robot_fx(
    audio: np.ndarray,
    sr: int,
    *,
    pitch_semitones: float = PITCH_SEMITONES,
    ring_hz: float = RING_HZ,
    ring_mix: float = RING_MIX,
    bitcrush_bits: float = BITCRUSH_BITS,
    metallic_gain_db: float = METALLIC_GAIN_DB,
    wet: float = WET,
) -> np.ndarray:
    """Apply metallic cute-robot FX to float32 mono audio."""
    if abs(wet) < 1e-3 or len(audio) == 0:
        return audio.astype(np.float32)

    dry = audio.astype(np.float32)
    wet_audio = _pitch_shift(dry, sr, pitch_semitones)
    wet_audio = _ring_mod(wet_audio, sr, ring_hz, ring_mix)
    wet_audio = _bitcrush(wet_audio, bitcrush_bits)
    wet_audio = _metallic_eq(wet_audio, sr, metallic_gain_db)
    peak = float(np.max(np.abs(wet_audio)) or 1.0)
    wet_audio = wet_audio / peak * 0.92
    out = (1.0 - wet) * dry + wet * wet_audio
    peak = float(np.max(np.abs(out)) or 1.0)
    if peak > 0.98:
        out *= 0.98 / peak
    return out.astype(np.float32)


def process_pcm16(
    raw: bytes,
    sr: int,
    *,
    pitch_semitones: float = PITCH_SEMITONES,
    ring_hz: float = RING_HZ,
    ring_mix: float = RING_MIX,
    bitcrush_bits: float = BITCRUSH_BITS,
    metallic_gain_db: float = METALLIC_GAIN_DB,
    wet: float = WET,
) -> bytes:
    """Run cute-robot FX on int16 PCM and return int16 PCM bytes."""
    if not raw:
        return b""
    return float_to_pcm16(
        cute_robot_fx(
            pcm16_to_float(raw),
            sr,
            pitch_semitones=pitch_semitones,
            ring_hz=ring_hz,
            ring_mix=ring_mix,
            bitcrush_bits=bitcrush_bits,
            metallic_gain_db=metallic_gain_db,
            wet=wet,
        )
    )

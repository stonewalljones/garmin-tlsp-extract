#!/usr/bin/env python3
"""
Garmin Travelapse & Dashcam Telemetry Extractor
==============================================
Extracts burned-in (or embedded) Timestamps, GPS Coordinates, and Speed
from Garmin Travelapse and dashcam videos, and exports to GPX, CSV, GeoJSON, or KML.

Features:
- Video ingestion via OpenCV or FFmpeg streaming.
- Specialized OCR preprocessing for white/translucent Garmin overlay text.
- Robust multi-pattern parsers for GPS (DD, DDM, DMS), Speed (MPH/KMH), and Timestamps.
- Multi-format exporters: GPX (1.1 standard), CSV, GeoJSON, and KML.
- Preview mode (--preview) to inspect ROI and OCR binarization before full run.
- Embedded subtitle/telemetry track extraction fallback via FFmpeg.
- Temporal deduplication, anomaly filtering, and coordinate validation.
"""

import argparse
import atexit
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

# Ensure terminal echo and state are always restored on exit
def _restore_terminal():
    try:
        subprocess.run(["stty", "sane"], check=False, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    except Exception:
        pass

atexit.register(_restore_terminal)

# Optional 3rd party libraries with graceful fallbacks
try:
    import cv2
    import numpy as np

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    np = None

try:
    import pytesseract
    from PIL import Image

    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False
    try:
        from PIL import Image
    except ImportError:
        Image = None

try:
    import easyocr

    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

HAS_TESSERACT_BIN = shutil.which("tesseract") is not None

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


@dataclass
class TelemetryPoint:
    """Represents a single telemetry waypoint extracted from video."""

    latitude: float
    longitude: float
    timestamp: Optional[datetime] = None
    speed_mph: Optional[float] = None
    speed_kmh: Optional[float] = None
    speed_mps: Optional[float] = None
    frame_number: int = 0
    video_time_sec: float = 0.0
    raw_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timestamp_iso": (
                self.timestamp.isoformat() if self.timestamp else None
            ),
            "speed_mph": round(self.speed_mph, 2) if self.speed_mph is not None else None,
            "speed_kmh": round(self.speed_kmh, 2) if self.speed_kmh is not None else None,
            "speed_mps": round(self.speed_mps, 2) if self.speed_mps is not None else None,
            "frame_number": self.frame_number,
            "video_time_sec": round(self.video_time_sec, 3),
            "raw_text": self.raw_text.strip(),
        }


@dataclass
class DebugFrameRecord:
    """Detailed diagnostic record for every frame processed before/after cleaning."""

    frame_number: int
    video_time_sec: float
    raw_ocr: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    timestamp_iso: Optional[str] = None
    speed_mph: Optional[float] = None
    speed_kmh: Optional[float] = None
    has_coords: bool = False
    has_timestamp: bool = False
    has_speed: bool = False
    status: str = "PENDING"
    drop_reason: str = ""


# ==============================================================================
# Telemetry Text Parsing (GPS, Timestamp, Speed)
# ==============================================================================

class TelemetryParser:
    """Parses raw OCR text strings into structured telemetry data."""

    # Pre-compiled regular expressions for OCR variations

    # Garmin speed pattern — handles OCR corruptions observed in real video:
    #   - Dash/dot/underscore glued between number and unit: "22-MPH", "15.MPH", "84_MPH"
    #   - Letter O confused with zero: "OMPH", "O-MPH"
    #   - Unit fragmented by spaces/dots: "MP H", "M.P.H", "M-P-H"
    #   - No separator at all: "22MPH"
    SPEED_PATTERN = re.compile(
        r"(?i)"
        r"(?<![\d\.])"
        r"(?P<speed>[Oo]|\d{1,3}(?:\.\d+)?)"
        r"[\-\—\–\_\.\s]{0,3}"
        r"(?P<unit>M[\-\.\s]?P[\-\.\s]?H|MPH|KM[\./ ]H|KMH|KPH|KNOTS?|KT)"
        r"(?!\d)",
    )

    # Date/Time patterns commonly used by Garmin:
    # 1. YYYY/MM/DD HH:MM:SS or YYYY-MM-DD HH:MM:SS
    # 2. MM/DD/YYYY HH:MM:SS (AM/PM)
    # 3. DD/MM/YYYY HH:MM:SS
    DATETIME_PATTERNS = [
        (
            re.compile(
                r"\b(?P<year>20\d\d)[/-](?P<month>0?[1-9]|1[0-2])[/-](?P<day>0?[1-9]|[12]\d|3[01])\s+(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)\b"
            ),
            "%Y-%m-%d %H:%M:%S",
        ),
        (
            re.compile(
                r"\b(?P<month>0?[1-9]|1[0-2])[/-](?P<day>0?[1-9]|[12]\d|3[01])[/-](?P<year>20\d\d)\s+(?P<hour>[01]?\d|2[0-3]):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)\s*(?P<ampm>[AP]M)?\b",
                re.IGNORECASE,
            ),
            "%m/%d/%Y %I:%M:%S %p",
        ),
    ]

    # Coordinate Patterns:
    # Pattern A: Degrees and Decimal Minutes (Garmin default, e.g., N 37° 46.123' W 122° 25.456' or N37 46.123 W122 25.456)
    DDM_PATTERN = re.compile(
        r"(?i)(?P<lat_hemi>[NS])\s*[:\s]?\s*(?P<lat_deg>\d{1,2})[°*\s]+(?P<lat_min>\d{1,2}(?:\.\d+)?)\'?\s*[,/|\s]\s*"
        r"(?P<lon_hemi>[EW])\s*[:\s]?\s*(?P<lon_deg>\d{1,3})[°*\s]+(?P<lon_min>\d{1,2}(?:\.\d+)?)\'?",
    )

    # Pattern B: Decimal Degrees with Hemispheres (e.g. 37.77490 N, 122.41940 W or N37.7749 W122.4194)
    DD_HEMI_PATTERN = re.compile(
        r"(?i)(?:(?P<lat_hemi_pre>[NS])\s*)?(?P<lat_val>\d{1,2}\.\d{3,7})\s*(?:(?P<lat_hemi_post>[NS]))?\s*[,/|\s]\s*"
        r"(?:(?P<lon_hemi_pre>[EW])\s*)?(?P<lon_val>\d{1,3}\.\d{3,7})\s*(?:(?P<lon_hemi_post>[EW]))?"
    )

    # Pattern C: Signed Decimal Degrees (e.g. LAT: 37.7749, LON: -122.4194)
    DD_SIGNED_PATTERN = re.compile(
        r"(?i)(?:lat(?:itude)?\s*[:=\s]\s*)?(?P<lat>[+-]?\d{1,2}\.\d{4,7})\s*[,/|\s]\s*"
        r"(?:lon(?:gitude)?\s*[:=\s]\s*)?(?P<lon>[+-]?\d{1,3}\.\d{4,7})"
    )

    @classmethod
    def clean_ocr_text(cls, text: str) -> str:
        """Fixes common OCR confusions in numeric Garmin telemetry."""
        # Replace confusing OCR characters in coordinate-like segments
        replacements = {
            "|": " ",
            "~": "-",
            "—": "-",
            "–": "-",
            "*": " ",
            "'": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "{": " ",
            "}": " ",
        }
        cleaned = text
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)

        # ── Longitude decimal-point normalisation ─────────────────────────────
        # OCR frequently misreads the decimal point in the 3-digit western longitude
        # as a colon, degree sign, or dash, e.g.:
        #   "-106:02727"  →  "-106.02727"   (colon misread)
        #   "-106°02848"  →  "-106.02848"   (degree sign misread)
        #   "-106-02774"  →  "-106.02774"   (dash misread — must be 4-6 decimal digits)
        # These patterns are safe: timestamps use 2-digit fields (HH:MM) so the
        # 3-digit-before-separator constraint avoids false matches in time strings.
        cleaned = re.sub(r'(-1\d{2})[:°](\d{4,6})\b', r'\1.\2', cleaned)
        cleaned = re.sub(r'(-1\d{2})-(\d{4,6})\b', r'\1.\2', cleaned)

        # Collapse repeated spaces
        return re.sub(r"[ \t]+", " ", cleaned).strip()

    @classmethod
    def parse_coordinates(cls, text: str) -> Optional[Tuple[float, float]]:
        """Extracts latitude and longitude in decimal degrees from text."""
        cleaned = cls.clean_ocr_text(text)

        # 1. Try Degrees & Decimal Minutes (DDM: N 37° 46.123' W 122° 25.456')
        m_ddm = cls.DDM_PATTERN.search(cleaned)
        if m_ddm:
            try:
                lat_deg = float(m_ddm.group("lat_deg"))
                lat_min = float(m_ddm.group("lat_min"))
                lat = lat_deg + (lat_min / 60.0)
                if m_ddm.group("lat_hemi").upper() == "S":
                    lat = -lat

                lon_deg = float(m_ddm.group("lon_deg"))
                lon_min = float(m_ddm.group("lon_min"))
                lon = lon_deg + (lon_min / 60.0)
                if m_ddm.group("lon_hemi").upper() == "W":
                    lon = -lon

                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except (ValueError, TypeError):
                pass

        # 2. Try Decimal Degrees with Hemisphere tags
        m_dd_hemi = cls.DD_HEMI_PATTERN.search(cleaned)
        if m_dd_hemi:
            try:
                lat_val = float(m_dd_hemi.group("lat_val"))
                lon_val = float(m_dd_hemi.group("lon_val"))
                lat_h = (
                    m_dd_hemi.group("lat_hemi_pre")
                    or m_dd_hemi.group("lat_hemi_post")
                    or "N"
                ).upper()
                lon_h = (
                    m_dd_hemi.group("lon_hemi_pre")
                    or m_dd_hemi.group("lon_hemi_post")
                    or "E"
                ).upper()

                lat = -lat_val if lat_h == "S" else lat_val
                lon = -lon_val if lon_h == "W" else lon_val

                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except (ValueError, TypeError):
                pass

        # 3. Try Signed Decimal Degrees
        m_dd_signed = cls.DD_SIGNED_PATTERN.search(cleaned)
        if m_dd_signed:
            try:
                lat = float(m_dd_signed.group("lat"))
                lon = float(m_dd_signed.group("lon"))
                if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                    return (lat, lon)
            except (ValueError, TypeError):
                pass

        return None

    @classmethod
    def parse_speed(cls, text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Parses speed and returns (speed_mph, speed_kmh, speed_mps).
        """
        cleaned = cls.clean_ocr_text(text)
        m = cls.SPEED_PATTERN.search(cleaned)
        if not m:
            return (None, None, None)

        try:
            # Normalise speed value: strip embedded junk, handle letter-O → 0
            raw_speed = m.group("speed").strip()
            raw_speed = raw_speed.replace("O", "0").replace("o", "0")
            val = float(raw_speed)

            # Normalise unit: collapse fragmented variants ("M.P.H", "M-P-H", "MP H" → "mph")
            raw_unit = m.group("unit").lower()
            raw_unit = re.sub(r'[\-\.\s]', '', raw_unit)  # strip separators inside unit

            if raw_unit == "mph":
                mph = val
                kmh = mph * 1.609344
            elif raw_unit in ("kmh", "kph", "kmh"):
                kmh = val
                mph = kmh / 1.609344
            elif raw_unit in ("knots", "kt"):
                mph = val * 1.15078
                kmh = val * 1.852
            else:
                return (None, None, None)

            mps = kmh / 3.6
            return (mph, kmh, mps)
        except (ValueError, TypeError):
            return (None, None, None)

    @classmethod
    def parse_timestamp(cls, text: str) -> Optional[datetime]:
        """Parses timestamp into datetime object."""
        cleaned = cls.clean_ocr_text(text)

        # Try regex patterns
        for pattern, _ in cls.DATETIME_PATTERNS:
            m = pattern.search(cleaned)
            if m:
                d = m.groupdict()
                try:
                    year = int(d["year"])
                    month = int(d["month"])
                    day = int(d["day"])
                    hour = int(d["hour"])
                    minute = int(d["minute"])
                    second = int(d["second"])

                    ampm = d.get("ampm")
                    if ampm:
                        ampm = ampm.upper()
                        if ampm == "PM" and hour < 12:
                            hour += 12
                        elif ampm == "AM" and hour == 12:
                            hour = 0

                    return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
                except (ValueError, KeyError):
                    continue

        return None

    @classmethod
    def parse_frame_text(
        cls, text: str, frame_num: int = 0, time_sec: float = 0.0
    ) -> Optional[TelemetryPoint]:
        """Parses a full block of frame OCR text into a TelemetryPoint."""
        coords = cls.parse_coordinates(text)
        if not coords:
            return None

        lat, lon = coords
        mph, kmh, mps = cls.parse_speed(text)
        dt = cls.parse_timestamp(text)

        return TelemetryPoint(
            latitude=lat,
            longitude=lon,
            timestamp=dt,
            speed_mph=mph,
            speed_kmh=kmh,
            speed_mps=mps,
            frame_number=frame_num,
            video_time_sec=time_sec,
            raw_text=text,
        )

    @classmethod
    def parse_frame_debug(
        cls, text: str, frame_num: int = 0, time_sec: float = 0.0
    ) -> Tuple[Optional[TelemetryPoint], DebugFrameRecord]:
        """Parses a frame OCR text, creating both a candidate TelemetryPoint and diagnostic DebugFrameRecord."""
        coords = cls.parse_coordinates(text)
        mph, kmh, mps = cls.parse_speed(text)
        dt = cls.parse_timestamp(text)

        has_coords = coords is not None
        has_dt = dt is not None
        has_speed = mph is not None or kmh is not None

        lat = coords[0] if coords else None
        lon = coords[1] if coords else None
        dt_iso = dt.isoformat() if dt else None

        missing = []
        if not has_coords:
            missing.append("coordinates")
        if not has_dt:
            missing.append("timestamp")
        if not has_speed:
            missing.append("speed")

        clean_text = text.replace("\n", " ").strip()
        if not clean_text:
            status = "NO_TEXT_DETECTED"
            reason = "OCR returned empty string (no text detected in ROI)"
        elif missing:
            status = "DROPPED_INCOMPLETE"
            reason = f"Missing required fields: {', '.join(missing)}"
        else:
            status = "CANDIDATE"
            reason = "Successfully parsed coordinates, timestamp, and speed"

        rec = DebugFrameRecord(
            frame_number=frame_num,
            video_time_sec=time_sec,
            raw_ocr=clean_text,
            latitude=lat,
            longitude=lon,
            timestamp_iso=dt_iso,
            speed_mph=mph,
            speed_kmh=kmh,
            has_coords=has_coords,
            has_timestamp=has_dt,
            has_speed=has_speed,
            status=status,
            drop_reason=reason,
        )

        pt = None
        if has_coords:
            pt = TelemetryPoint(
                latitude=lat,
                longitude=lon,
                timestamp=dt,
                speed_mph=mph,
                speed_kmh=kmh,
                speed_mps=mps,
                frame_number=frame_num,
                video_time_sec=time_sec,
                raw_text=text,
            )

        return (pt, rec)


# ==============================================================================
# Video Frame Extraction & OCR Preprocessing
# ==============================================================================

class VideoProcessor:
    """Handles video frame reading, ROI cropping, image enhancement, and OCR."""

    def __init__(
        self,
        video_path: str,
        roi_bbox: Optional[Tuple[float, float, float, float]] = None,
        ocr_engine: str = "tesseract",
    ):
        """
        roi_bbox: (ymin, ymax, xmin, xmax) normalized coordinates (0.0 to 1.0)
                  Default is bottom Garmin overlay band: (0.94, 0.998, 0.0, 0.85)
        """
        self.video_path = str(video_path)
        # Tight default focus on Garmin bottom telemetry strip
        self.roi_bbox = roi_bbox or (0.94, 0.998, 0.0, 0.85)
        self.ocr_engine = ocr_engine

        if not Path(self.video_path).exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

    @staticmethod
    def crop_roi(
        image_np: Any, roi_bbox: Tuple[float, float, float, float]
    ) -> Any:
        """Crops image numpy array based on normalized bbox (ymin, ymax, xmin, xmax)."""
        h, w = image_np.shape[:2]
        ymin, ymax, xmin, xmax = roi_bbox
        y1 = max(0, int(ymin * h))
        y2 = min(h, int(ymax * h))
        x1 = max(0, int(xmin * w))
        x2 = min(w, int(xmax * w))
        return image_np[y1:y2, x1:x2]

    # Minimum ratio of bright pixels in ROI required to attempt OCR.
    # ROIs below this threshold are blank/dark frames with no telemetry overlay.
    _BRIGHT_PIXEL_THRESHOLD: int = 170
    _BRIGHT_PIXEL_MIN_RATIO: float = 0.02

    @staticmethod
    def _has_overlay_pixels(gray: Any) -> bool:
        """
        Fast pixel density pre-filter (~0.1ms). Returns False when the ROI is a
        blank/dark frame that cannot contain the Garmin telemetry overlay, allowing
        the worker to skip all Tesseract calls entirely.
        """
        if not HAS_OPENCV or gray is None:
            return True  # Can't check — let OCR try anyway
        bright_count = int(np.count_nonzero(gray >= VideoProcessor._BRIGHT_PIXEL_THRESHOLD))
        total = gray.size
        return total > 0 and (bright_count / total) >= VideoProcessor._BRIGHT_PIXEL_MIN_RATIO

    @staticmethod
    def _make_scaled_gray(roi_image: Any) -> Any:
        """Converts ROI to grayscale and upscales to at least 60px tall for Tesseract."""
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()
        h, w = gray.shape[:2]
        if h < 60:
            scale_factor = max(2.0, 70.0 / max(1, h))
            gray = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor,
                              interpolation=cv2.INTER_CUBIC)
        return gray

    @staticmethod
    def _candidate_images(scaled_gray: Any):
        """
        Lazy generator that yields preprocessed candidate images one at a time.
        Candidates are ordered fastest-to-slowest so the caller can stop as soon
        as a successful parse is found, avoiding unnecessary preprocessing work.

        Candidate 1 - brightness threshold (fastest, best for Garmin white-on-dark)
        Candidate 2 - morphological close + Otsu  (catches lower-contrast text)
        Candidate 3 - contrast-stretched grayscale (final fallback)
        """
        pad = lambda img: cv2.copyMakeBorder(img, 15, 15, 15, 15,
                                              cv2.BORDER_CONSTANT, value=255)
        # Candidate 1: direct brightness threshold + invert
        _, bright = cv2.threshold(scaled_gray, 170, 255, cv2.THRESH_BINARY)
        yield pad(cv2.bitwise_not(bright))

        # Candidate 2: morphological close (replaces the slow bilateralFilter) + Otsu + invert
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(scaled_gray, cv2.MORPH_CLOSE, kernel)
        _, otsu = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        yield pad(cv2.bitwise_not(otsu))

        # Candidate 3: contrast-stretched grayscale
        norm = cv2.normalize(scaled_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        yield pad(norm)

    @classmethod
    def run_ocr_on_roi(cls, roi_image: Any) -> str:
        """
        Unified, lazy OCR pipeline:
        1. Fast pixel density pre-filter - skips Tesseract on blank/dark frames entirely.
        2. Generates candidate preprocessed images one at a time (no eager allocation).
        3. Tries PSM 7 then PSM 6 on each candidate, stopping as soon as GPS
           coordinates parse successfully (no PSM 11 - too slow for single-line overlays).
        4. Uses --oem 1 (LSTM only) instead of --oem 3 for consistent, faster calls.

        Thread-safe - called directly by parallel worker threads.
        """
        if not (HAS_PYTESSERACT or HAS_TESSERACT_BIN or HAS_EASYOCR):
            return ""

        if not HAS_OPENCV or roi_image is None or roi_image.size == 0:
            # No OpenCV - fall back to raw PIL OCR
            return cls._tesseract_call(roi_image, "--psm 6")

        scaled_gray = cls._make_scaled_gray(roi_image)

        # ── Pre-filter: skip Tesseract entirely on blank/dark frames ──────────
        if not cls._has_overlay_pixels(scaled_gray):
            return ""

        # ── Lazy per-candidate loop, PSM 7 then PSM 6, stop on first parse ───
        psm_modes = ("--psm 7", "--psm 6")
        best_text = ""

        for candidate_img in cls._candidate_images(scaled_gray):
            for psm in psm_modes:
                text = cls._tesseract_call(candidate_img, psm)
                if text and TelemetryParser.parse_coordinates(text):
                    return text  # ← early exit on first successful parse
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text

        return best_text

    @staticmethod
    def _tesseract_call(img: Any, psm: str) -> str:
        """
        Single Tesseract invocation. Uses --oem 1 (LSTM only) for consistent
        timing. Returns empty string on any error.
        """
        config = f"{psm} --oem 1"
        if HAS_PYTESSERACT:
            try:
                pil_img = (
                    Image.fromarray(img)
                    if (HAS_OPENCV and isinstance(img, np.ndarray))
                    else img
                )
                return pytesseract.image_to_string(pil_img, config=config)
            except Exception:
                return ""
        elif HAS_TESSERACT_BIN:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_name = tmp.name
            try:
                if HAS_OPENCV and isinstance(img, np.ndarray):
                    cv2.imwrite(tmp_name, img)
                elif hasattr(img, "save"):
                    img.save(tmp_name)
                res = subprocess.run(
                    ["tesseract", tmp_name, "stdout"] + psm.split() + ["--oem", "1"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    text=True,
                    timeout=5,
                )
                return res.stdout
            except Exception:
                return ""
            finally:
                try:
                    os.remove(tmp_name)
                except OSError:
                    pass
        return ""

    # ── Legacy shims so existing callers (preview, tests) still work ─────────
    @staticmethod
    def preprocess_for_ocr(roi_image: Any) -> List[Any]:
        """Compatibility shim - returns the raw ROI; run_ocr_on_roi handles preprocessing lazily."""
        return [roi_image]

    @classmethod
    def run_ocr(cls, processed_imgs: Any) -> str:
        """Compatibility shim for callers that still use the old two-step API."""
        roi = processed_imgs[0] if isinstance(processed_imgs, list) else processed_imgs
        return cls.run_ocr_on_roi(roi)

    def extract_frames_opencv(
        self, sample_interval_sec: float = 0.0, every_frame: bool = False
    ) -> Generator[Tuple[int, float, Any], None, None]:
        """Yields (frame_num, timestamp_sec, cropped_roi_image) using OpenCV."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if every_frame or sample_interval_sec <= 0:
            frame_step = 1
        else:
            frame_step = max(1, int(round(fps * sample_interval_sec)))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                sec = frame_idx / fps
                roi = self.crop_roi(frame, self.roi_bbox)
                yield (frame_idx, sec, roi)

            frame_idx += 1

        cap.release()

    def extract_frames_ffmpeg(
        self, sample_interval_sec: float = 0.0, every_frame: bool = False
    ) -> Generator[Tuple[int, float, Any], None, None]:
        """Yields frames via FFmpeg pipe subprocess (fallback when OpenCV is missing)."""
        cmd = ["ffmpeg", "-nostdin", "-i", self.video_path]

        # Probe video fps if possible
        fps = 30.0
        try:
            probe_res = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", self.video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            if probe_res.stdout and "/" in probe_res.stdout:
                num, den = probe_res.stdout.strip().split("/")
                fps = float(num) / float(den)
        except Exception:
            pass

        if not (every_frame or sample_interval_sec <= 0):
            fps_filter = f"fps=1/{sample_interval_sec}" if sample_interval_sec >= 1.0 else f"fps={1.0/sample_interval_sec}"
            cmd.extend(["-vf", fps_filter])

        cmd.extend(["-f", "image2pipe", "-vcodec", "png", "-"])

        pipe = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=10**7,
        )

        import io
        from PIL import Image

        frame_idx = 0
        img_buffer = bytearray()
        # PNG signature
        png_sig = b"\x89PNG\r\n\x1a\n"

        while True:
            chunk = pipe.stdout.read(65536)
            if not chunk:
                break
            img_buffer.extend(chunk)

            # Find PNG images in stream
            while True:
                start = img_buffer.find(png_sig)
                if start == -1:
                    img_buffer.clear()
                    break
                # Find next PNG or end
                next_start = img_buffer.find(png_sig, start + 8)
                if next_start == -1:
                    # Keep remaining buffer for next read
                    img_buffer = img_buffer[start:]
                    break

                png_bytes = img_buffer[start:next_start]
                img_buffer = img_buffer[next_start:]

                try:
                    pil_img = Image.open(io.BytesIO(png_bytes))
                    if HAS_OPENCV:
                        img_np = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                        roi = self.crop_roi(img_np, self.roi_bbox)
                    else:
                        w, h = pil_img.size
                        ymin, ymax, xmin, xmax = self.roi_bbox
                        roi = pil_img.crop((int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)))

                    if every_frame or sample_interval_sec <= 0:
                        sec = frame_idx / fps
                    else:
                        sec = frame_idx * sample_interval_sec

                    yield (frame_idx, sec, roi)
                    frame_idx += 1
                except Exception:
                    pass

        pipe.terminate()


def _process_frame_worker(task: Tuple[int, float, Any]) -> Tuple[Optional[TelemetryPoint], DebugFrameRecord]:
    """Worker function executed in parallel threads to OCR and parse one frame."""
    frame_num, sec, roi = task
    if roi is None:
        rec = DebugFrameRecord(
            frame_number=frame_num,
            video_time_sec=sec,
            raw_ocr="",
            status="NO_IMAGE",
            drop_reason="Failed to extract frame ROI from video stream",
        )
        return (None, rec)
    try:
        # Unified lazy pipeline: pre-filter + lazy candidate generation + PSM 7/6 + oem 1
        ocr_text = VideoProcessor.run_ocr_on_roi(roi)
        return TelemetryParser.parse_frame_debug(ocr_text, frame_num, sec)
    except Exception as e:
        rec = DebugFrameRecord(
            frame_number=frame_num,
            video_time_sec=sec,
            raw_ocr="",
            status="ERROR",
            drop_reason=f"Exception during OCR processing: {e}",
        )
        return (None, rec)


def process_frames_parallel(
    frame_gen: Generator[Tuple[int, float, Any], None, None],
    num_workers: int = 4,
    total_frames: Optional[int] = None,
) -> Tuple[List[TelemetryPoint], List[DebugFrameRecord]]:
    """
    Executes frame preprocessing and OCR across multiple worker threads in parallel.
    Uses a rolling buffer to minimize memory usage while keeping CPU workers fully saturated.
    Returns both candidate TelemetryPoints and comprehensive DebugFrameRecords.
    """
    points: List[TelemetryPoint] = []
    debug_records: List[DebugFrameRecord] = []
    max_queued = max(8, num_workers * 4)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_map: Dict[Any, int] = {}
        pbar = (
            tqdm(total=total_frames, desc=f"Processing Frames ({num_workers} threads)", unit="frame")
            if HAS_TQDM
            else None
        )

        for frame_item in frame_gen:
            fut = executor.submit(_process_frame_worker, frame_item)
            future_map[fut] = frame_item[0]

            # Drain completed tasks when the queue gets full
            while len(future_map) >= max_queued:
                done, _ = concurrent.futures.wait(
                    future_map.keys(), return_when=concurrent.futures.FIRST_COMPLETED
                )
                for f in done:
                    pt, rec = f.result()
                    if pt:
                        points.append(pt)
                    debug_records.append(rec)
                    if pbar is not None:
                        pbar.update(1)
                    del future_map[f]

        # Drain all remaining tasks
        for f in concurrent.futures.as_completed(future_map.keys()):
            pt, rec = f.result()
            if pt:
                points.append(pt)
            debug_records.append(rec)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    # Ensure chronological sort by frame number
    points.sort(key=lambda p: p.frame_number)
    debug_records.sort(key=lambda r: r.frame_number)
    return points, debug_records


# ==============================================================================
# Exporters (GPX, CSV, GeoJSON, KML)
# ==============================================================================

class TelemetryExporter:
    """Exports structured TelemetryPoint list into multiple geospatial formats."""

    @staticmethod
    def to_csv(points: List[TelemetryPoint], output_path: str) -> None:
        """Exports points to a clean CSV spreadsheet."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp_iso",
                    "latitude",
                    "longitude",
                    "speed_mph",
                    "speed_kmh",
                    "speed_mps",
                    "frame_number",
                    "video_time_sec",
                    "raw_ocr",
                ]
            )
            for p in points:
                writer.writerow(
                    [
                        p.timestamp.isoformat() if p.timestamp else "",
                        f"{p.latitude:.6f}",
                        f"{p.longitude:.6f}",
                        f"{p.speed_mph:.2f}" if p.speed_mph is not None else "",
                        f"{p.speed_kmh:.2f}" if p.speed_kmh is not None else "",
                        f"{p.speed_mps:.2f}" if p.speed_mps is not None else "",
                        p.frame_number,
                        f"{p.video_time_sec:.3f}",
                        p.raw_text.replace("\n", " ").strip(),
                    ]
                )

    @staticmethod
    def to_debug_csv(records: List[DebugFrameRecord], output_path: str) -> None:
        """Exports raw frame OCR and parsing/cleaning diagnostics to a CSV for troubleshooting."""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "frame_number",
                    "video_time_sec",
                    "status",
                    "drop_reason",
                    "has_coords",
                    "has_timestamp",
                    "has_speed",
                    "latitude",
                    "longitude",
                    "timestamp_iso",
                    "speed_mph",
                    "speed_kmh",
                    "raw_ocr_text",
                ]
            )
            for r in records:
                writer.writerow(
                    [
                        r.frame_number,
                        f"{r.video_time_sec:.3f}",
                        r.status,
                        r.drop_reason,
                        "YES" if r.has_coords else "NO",
                        "YES" if r.has_timestamp else "NO",
                        "YES" if r.has_speed else "NO",
                        f"{r.latitude:.6f}" if r.latitude is not None else "",
                        f"{r.longitude:.6f}" if r.longitude is not None else "",
                        r.timestamp_iso or "",
                        f"{r.speed_mph:.2f}" if r.speed_mph is not None else "",
                        f"{r.speed_kmh:.2f}" if r.speed_kmh is not None else "",
                        r.raw_ocr.replace("\n", " ").strip(),
                    ]
                )

    @staticmethod
    def to_gpx(points: List[TelemetryPoint], output_path: str, track_name: str = "Garmin Travelapse") -> None:
        """
        Exports points to standard GPX 1.1 with <trk>, <trkseg>, <trkpt>,
        <time>, <speed> (in m/s), and Garmin extensions.
        """
        gpx = ET.Element(
            "gpx",
            version="1.1",
            creator="Garmin Travelapse Telemetry Extractor",
            xmlns="http://www.topografix.com/GPX/1/1",
        )

        metadata = ET.SubElement(gpx, "metadata")
        name_elem = ET.SubElement(metadata, "name")
        name_elem.text = track_name
        time_elem = ET.SubElement(metadata, "time")
        if points and points[0].timestamp:
            time_elem.text = points[0].timestamp.isoformat()
        else:
            time_elem.text = datetime.now(timezone.utc).isoformat()

        trk = ET.SubElement(gpx, "trk")
        trk_name = ET.SubElement(trk, "name")
        trk_name.text = track_name
        trkseg = ET.SubElement(trk, "trkseg")

        for p in points:
            trkpt = ET.SubElement(
                trkseg,
                "trkpt",
                lat=f"{p.latitude:.6f}",
                lon=f"{p.longitude:.6f}",
            )
            if p.timestamp:
                t = ET.SubElement(trkpt, "time")
                t.text = p.timestamp.isoformat()

            if p.speed_mps is not None:
                s = ET.SubElement(trkpt, "speed")
                s.text = f"{p.speed_mps:.2f}"

            # Extensions for extra dashboard telemetry
            ext = ET.SubElement(trkpt, "extensions")
            if p.speed_mph is not None:
                mph_elem = ET.SubElement(ext, "speed_mph")
                mph_elem.text = f"{p.speed_mph:.1f}"
            if p.speed_kmh is not None:
                kmh_elem = ET.SubElement(ext, "speed_kmh")
                kmh_elem.text = f"{p.speed_kmh:.1f}"
            video_sec_elem = ET.SubElement(ext, "video_time_sec")
            video_sec_elem.text = f"{p.video_time_sec:.2f}"

        tree = ET.ElementTree(gpx)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def to_geojson(points: List[TelemetryPoint], output_path: str) -> None:
        """
        Exports points to GeoJSON with both a LineString track and individual Point features.
        Compatible with geojson.io, QGIS, Mapbox, and Leaflet.
        """
        features = []

        # 1. Trajectory LineString
        coordinates = [[p.longitude, p.latitude] for p in points]
        if coordinates:
            line_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates,
                },
                "properties": {
                    "name": "Garmin Travelapse Route",
                    "total_points": len(points),
                    "start_time": points[0].timestamp.isoformat() if points[0].timestamp else None,
                    "end_time": points[-1].timestamp.isoformat() if points[-1].timestamp else None,
                },
            }
            features.append(line_feature)

        # 2. Individual Waypoints
        for p in points:
            pt_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [p.longitude, p.latitude],
                },
                "properties": {
                    "timestamp": p.timestamp.isoformat() if p.timestamp else None,
                    "speed_mph": round(p.speed_mph, 1) if p.speed_mph is not None else None,
                    "speed_kmh": round(p.speed_kmh, 1) if p.speed_kmh is not None else None,
                    "speed_mps": round(p.speed_mps, 2) if p.speed_mps is not None else None,
                    "video_time_sec": round(p.video_time_sec, 2),
                    "frame_number": p.frame_number,
                },
            }
            features.append(pt_feature)

        geojson_data = {
            "type": "FeatureCollection",
            "features": features,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson_data, f, indent=2)

    @staticmethod
    def to_kml(points: List[TelemetryPoint], output_path: str, track_name: str = "Garmin Travelapse") -> None:
        """Exports points to Google Earth KML format with stylized route and waypoint placemarks."""
        kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
        doc = ET.SubElement(kml, "Document")
        doc_name = ET.SubElement(doc, "name")
        doc_name.text = track_name

        # Route Style (Red line)
        style = ET.SubElement(doc, "Style", id="routeStyle")
        line_style = ET.SubElement(style, "LineStyle")
        color = ET.SubElement(line_style, "color")
        color.text = "ff0000ff"  # ABGR: Opaque red
        width = ET.SubElement(line_style, "width")
        width.text = "4"

        # LineString Placemark
        route_pm = ET.SubElement(doc, "Placemark")
        pm_name = ET.SubElement(route_pm, "name")
        pm_name.text = f"{track_name} Path"
        style_url = ET.SubElement(route_pm, "styleUrl")
        style_url.text = "#routeStyle"
        linestring = ET.SubElement(route_pm, "LineString")
        tessellate = ET.SubElement(linestring, "tessellate")
        tessellate.text = "1"
        coords_elem = ET.SubElement(linestring, "coordinates")
        coords_elem.text = " ".join([f"{p.longitude},{p.latitude},0" for p in points])

        # Waypoint Placemarks Folder
        folder = ET.SubElement(doc, "Folder")
        folder_name = ET.SubElement(folder, "name")
        folder_name.text = "Waypoints"

        for p in points:
            pm = ET.SubElement(folder, "Placemark")
            p_name = ET.SubElement(pm, "name")
            p_name.text = p.timestamp.strftime("%H:%M:%S") if p.timestamp else f"T+{p.video_time_sec:.1f}s"

            desc = ET.SubElement(pm, "description")
            desc_lines = [
                f"<b>Time:</b> {p.timestamp.isoformat() if p.timestamp else 'N/A'}",
                f"<b>Coordinates:</b> {p.latitude:.6f}, {p.longitude:.6f}",
                f"<b>Speed:</b> {p.speed_mph:.1f} MPH ({p.speed_kmh:.1f} KM/H)" if p.speed_mph is not None else "<b>Speed:</b> N/A",
                f"<b>Video Time:</b> {p.video_time_sec:.2f}s",
            ]
            desc.text = "<br/>".join(desc_lines)

            point_elem = ET.SubElement(pm, "Point")
            pt_coords = ET.SubElement(point_elem, "coordinates")
            pt_coords.text = f"{p.longitude},{p.latitude},0"

        tree = ET.ElementTree(kml)
        ET.indent(tree, space="  ", level=0)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)


# ==============================================================================
# Telemetry Cleaning, Completeness Filtering & Sanity Check
# ==============================================================================

def haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates great-circle distance between two GPS coordinates in meters."""
    r_earth = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return r_earth * c


def clean_telemetry_points(
    points: List[TelemetryPoint],
    require_complete: bool = True,
    max_speed_kmh: float = 250.0,
    max_allowed_speed_mps: float = 70.0,
    debug_map: Optional[Dict[int, DebugFrameRecord]] = None,
) -> List[TelemetryPoint]:
    """
    Cleans and validates telemetry data:
    1. Completeness Filter: Discards any record missing valid coordinates, timestamp, or speed.
    2. Range & Bounds Check: Removes null island (0,0), out-of-bounds coordinates, and speed anomalies.
    3. Spatial Outlier / Spike Rejection: Detects and removes GPS coordinate glitches that jump
       away from preceding and succeeding neighboring waypoints.
    4. Deduplication: Removes consecutive identical duplicate records.
    """
    if not points:
        return []

    # Step 1: Completeness & Range Filtering
    valid_points: List[TelemetryPoint] = []
    for p in points:
        # Require all 3 fields: coordinates, timestamp, and speed
        if require_complete:
            if p.timestamp is None:
                if debug_map and p.frame_number in debug_map:
                    debug_map[p.frame_number].status = "DROPPED_INCOMPLETE"
                    debug_map[p.frame_number].drop_reason = "Missing valid parsed timestamp"
                continue
            if p.speed_mph is None and p.speed_kmh is None:
                if debug_map and p.frame_number in debug_map:
                    debug_map[p.frame_number].status = "DROPPED_INCOMPLETE"
                    debug_map[p.frame_number].drop_reason = "Missing valid parsed speed"
                continue

        # Coordinate bounds check
        if not (-90.0 <= p.latitude <= 90.0 and -180.0 <= p.longitude <= 180.0):
            if debug_map and p.frame_number in debug_map:
                debug_map[p.frame_number].status = "DROPPED_INVALID_COORDS"
                debug_map[p.frame_number].drop_reason = f"Coordinates out of bounds: ({p.latitude:.6f}, {p.longitude:.6f})"
            continue
        if abs(p.latitude) < 0.0001 and abs(p.longitude) < 0.0001:
            if debug_map and p.frame_number in debug_map:
                debug_map[p.frame_number].status = "DROPPED_INVALID_COORDS"
                debug_map[p.frame_number].drop_reason = "Null Island coordinates (0.0, 0.0)"
            continue

        # Speed bounds check
        if p.speed_kmh is not None and (p.speed_kmh < 0.0 or p.speed_kmh > max_speed_kmh):
            if debug_map and p.frame_number in debug_map:
                debug_map[p.frame_number].status = "DROPPED_SPEED_ANOMALY"
                debug_map[p.frame_number].drop_reason = f"Speed ({p.speed_kmh:.1f} km/h) exceeds maximum threshold ({max_speed_kmh:.1f} km/h)"
            continue

        valid_points.append(p)

    # Step 2: Deduplicate consecutive identical points
    deduped: List[TelemetryPoint] = []
    for p in valid_points:
        if deduped:
            last = deduped[-1]
            if (
                abs(last.latitude - p.latitude) < 1e-6
                and abs(last.longitude - p.longitude) < 1e-6
                and last.timestamp == p.timestamp
                and last.speed_mph == p.speed_mph
            ):
                if debug_map and p.frame_number in debug_map:
                    debug_map[p.frame_number].status = "DROPPED_DUPLICATE"
                    debug_map[p.frame_number].drop_reason = f"Identical coordinates and timestamp as previous frame #{last.frame_number}"
                continue
        deduped.append(p)

    if len(deduped) <= 2:
        for p in deduped:
            if debug_map and p.frame_number in debug_map:
                debug_map[p.frame_number].status = "KEPT"
                debug_map[p.frame_number].drop_reason = "Valid verified GPS waypoint"
        return deduped

    # Step 3: Neighbor-based GPS Sanity Check (Multi-point spike / glitch rejection)
    def filter_spikes_single_pass(pts: List[TelemetryPoint]) -> List[TelemetryPoint]:
        n = len(pts)
        if n <= 2:
            return pts

        outliers: Dict[int, str] = {}

        def is_spike(prev_pt: TelemetryPoint, curr_pt: TelemetryPoint, next_pt: TelemetryPoint) -> Tuple[bool, str]:
            d_prev_curr = haversine_distance_meters(prev_pt.latitude, prev_pt.longitude, curr_pt.latitude, curr_pt.longitude)
            d_curr_next = haversine_distance_meters(curr_pt.latitude, curr_pt.longitude, next_pt.latitude, next_pt.longitude)
            d_prev_next = haversine_distance_meters(prev_pt.latitude, prev_pt.longitude, next_pt.latitude, next_pt.longitude)

            dt_prev = max(0.05, abs((curr_pt.timestamp - prev_pt.timestamp).total_seconds()) if (curr_pt.timestamp and prev_pt.timestamp) else abs(curr_pt.video_time_sec - prev_pt.video_time_sec))
            dt_next = max(0.05, abs((next_pt.timestamp - curr_pt.timestamp).total_seconds()) if (next_pt.timestamp and curr_pt.timestamp) else abs(next_pt.video_time_sec - curr_pt.video_time_sec))

            v_prev = d_prev_curr / dt_prev
            v_next = d_curr_next / dt_next

            # If either jump implies an impossible speed and bypassing curr makes a consistent path
            if v_prev > max_allowed_speed_mps or v_next > max_allowed_speed_mps:
                dt_total = dt_prev + dt_next
                v_direct = d_prev_next / dt_total
                if v_direct <= max_allowed_speed_mps:
                    return (True, f"Jump of {d_prev_curr:.1f}m in {dt_prev:.2f}s implies {v_prev:.1f} m/s (deviates from #{prev_pt.frame_number} & #{next_pt.frame_number})")
                expected_dist = max(50.0, (curr_pt.speed_mps or 25.0) * 3.0 * dt_prev)
                if d_prev_curr > expected_dist and d_curr_next > expected_dist and d_prev_next < expected_dist:
                    return (True, f"Jump of {d_prev_curr:.1f}m deviates from path ({d_prev_next:.1f}m between #{prev_pt.frame_number} and #{next_pt.frame_number})")
            return (False, "")

        # Internal points
        for i in range(1, n - 1):
            spike, reason = is_spike(pts[i - 1], pts[i], pts[i + 1])
            if spike:
                outliers[i] = reason

        # Check endpoints
        if n >= 3 and 0 not in outliers and 1 not in outliers and 2 not in outliers:
            d01 = haversine_distance_meters(pts[0].latitude, pts[0].longitude, pts[1].latitude, pts[1].longitude)
            d12 = haversine_distance_meters(pts[1].latitude, pts[1].longitude, pts[2].latitude, pts[2].longitude)
            dt01 = max(0.05, abs((pts[1].timestamp - pts[0].timestamp).total_seconds()) if (pts[1].timestamp and pts[0].timestamp) else 1.0)
            dt12 = max(0.05, abs((pts[2].timestamp - pts[1].timestamp).total_seconds()) if (pts[2].timestamp and pts[1].timestamp) else 1.0)
            if (d01 / dt01 > max_allowed_speed_mps) and (d12 / dt12 <= max_allowed_speed_mps):
                outliers[0] = f"Endpoint jump {d01:.1f}m in {dt01:.2f}s implies {d01/dt01:.1f} m/s"

        if n >= 3 and (n - 1) not in outliers and (n - 2) not in outliers and (n - 3) not in outliers:
            d_last = haversine_distance_meters(pts[n - 2].latitude, pts[n - 2].longitude, pts[n - 1].latitude, pts[n - 1].longitude)
            d_prev = haversine_distance_meters(pts[n - 3].latitude, pts[n - 3].longitude, pts[n - 2].latitude, pts[n - 2].longitude)
            dt_last = max(0.05, abs((pts[n - 1].timestamp - pts[n - 2].timestamp).total_seconds()) if (pts[n - 1].timestamp and pts[n - 2].timestamp) else 1.0)
            dt_prev = max(0.05, abs((pts[n - 2].timestamp - pts[n - 3].timestamp).total_seconds()) if (pts[n - 2].timestamp and pts[n - 3].timestamp) else 1.0)
            if (d_last / dt_last > max_allowed_speed_mps) and (d_prev / dt_prev <= max_allowed_speed_mps):
                outliers[n - 1] = f"Endpoint jump {d_last:.1f}m in {dt_last:.2f}s implies {d_last/dt_last:.1f} m/s"

        for idx, reason in outliers.items():
            fnum = pts[idx].frame_number
            if debug_map and fnum in debug_map:
                debug_map[fnum].status = "DROPPED_OUTLIER_SPIKE"
                debug_map[fnum].drop_reason = reason

        return [pts[i] for i in range(n) if i not in outliers]

    sanitized = deduped
    for _ in range(2):
        prev_len = len(sanitized)
        sanitized = filter_spikes_single_pass(sanitized)
        if len(sanitized) == prev_len:
            break

    # Mark remaining points as KEPT
    for p in sanitized:
        if debug_map and p.frame_number in debug_map:
            debug_map[p.frame_number].status = "KEPT"
            debug_map[p.frame_number].drop_reason = "Valid verified GPS waypoint"

    return sanitized


# ==============================================================================
# CLI and Main Execution
# ==============================================================================

def check_ocr_availability() -> bool:
    """Checks if at least one OCR backend is installed."""
    if HAS_PYTESSERACT or HAS_TESSERACT_BIN or HAS_EASYOCR:
        return True
    print("\n" + "!" * 70, file=sys.stderr)
    print("[ERROR] No OCR engine detected on your system!", file=sys.stderr)
    print("To extract burned-in telemetry text from video frames, please install Tesseract OCR:", file=sys.stderr)
    print("  • Fedora / RHEL:   sudo dnf install tesseract", file=sys.stderr)
    print("  • Ubuntu / Debian: sudo apt install tesseract-ocr", file=sys.stderr)
    print("  • Arch Linux:      sudo pacman -S tesseract", file=sys.stderr)
    print("  • Python packages: pip install pytesseract opencv-python Pillow", file=sys.stderr)
    print("!" * 70 + "\n", file=sys.stderr)
    return False


def preview_roi(
    video_path: str,
    roi_bbox: Tuple[float, float, float, float],
    output_preview_path: str = "preview_roi.png",
) -> None:
    """Saves a preview image showing the cropped and preprocessed overlay bar."""
    check_ocr_availability()
    processor = VideoProcessor(video_path, roi_bbox=roi_bbox)
    gen = (
        processor.extract_frames_opencv(sample_interval_sec=1.0)
        if HAS_OPENCV
        else processor.extract_frames_ffmpeg(sample_interval_sec=1.0)
    )

    try:
        frame_num, sec, roi = next(gen)
    except StopIteration:
        print(f"Error: Could not extract any frame from {video_path}")
        return

    candidates = processor.preprocess_for_ocr(roi)
    text = processor.run_ocr(candidates)
    parsed = TelemetryParser.parse_frame_text(text, frame_num, sec)

    preview_img = candidates[0] if candidates else roi
    if HAS_OPENCV and isinstance(preview_img, np.ndarray):
        cv2.imwrite(output_preview_path, preview_img)
    elif hasattr(preview_img, "save"):
        preview_img.save(output_preview_path)

    print("\n" + "=" * 60)
    print(f"ROI Preview saved to: {output_preview_path}")
    print(f"Sample Video Time: {sec:.2f}s (Frame #{frame_num})")
    print(f"Raw OCR Output:\n---\n{text.strip()}\n---")
    if parsed:
        print(f"Parsed Latitude:  {parsed.latitude:.6f}")
        print(f"Parsed Longitude: {parsed.longitude:.6f}")
        print(f"Parsed Timestamp: {parsed.timestamp}")
        print(f"Parsed Speed:     {parsed.speed_mph} MPH ({parsed.speed_kmh} KM/H)")
    else:
        print("Warning: Could not parse GPS coordinates from sample frame.")
        print("Tip: Adjust --roi (ymin,ymax,xmin,xmax) or check overlay position.")
    print("=" * 60 + "\n")


def check_embedded_subtitles(video_path: str) -> Optional[List[TelemetryPoint]]:
    """Attempts to extract embedded subtitle / telemetry tracks via FFmpeg."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-i",
        video_path,
        "-map",
        "0:s:0?",
        "-f",
        "srt",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
        if not res.stdout or len(res.stdout.strip()) < 10:
            return None

        # Parse SRT blocks
        points: List[TelemetryPoint] = []
        blocks = res.stdout.strip().split("\n\n")
        for idx, block in enumerate(blocks):
            lines = block.splitlines()
            if len(lines) >= 3:
                text = " ".join(lines[2:])
                p = TelemetryParser.parse_frame_text(text, frame_num=idx, time_sec=float(idx))
                if p:
                    points.append(p)

        return points if points else None
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract burned-in GPS coordinates, speed, and timestamp from Garmin Travelapse videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract telemetry to GPX:
  python garmin_extractor.py -i travelapse.mp4 -o route.gpx

  # Extract all formats (GPX, CSV, GeoJSON, KML) sampling every 2 seconds:
  python garmin_extractor.py -i travelapse.mp4 --format all --interval 2.0

  # Check ROI preview on first frame to adjust bounding box:
  python garmin_extractor.py -i travelapse.mp4 --preview

  # Custom ROI bbox: bottom 10% of frame (ymin ymax xmin xmax):
  python garmin_extractor.py -i travelapse.mp4 -o route.csv --roi 0.90 1.0 0.0 1.0
        """,
    )

    parser.add_argument("-i", "--input", required=True, help="Path to input Garmin video file (.mp4, .mov, etc.)")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Path for output file (extension determines format: .gpx, .csv, .geojson, .kml). If omitted, matches video name.",
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=["gpx", "csv", "geojson", "kml", "all"],
        default="gpx",
        help="Export format (default: gpx). 'all' creates .gpx, .csv, .geojson, and .kml.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Frame sampling interval in seconds (default: 0.0 for every single frame). Set e.g. 1.0 to sample 1 frame per second.",
    )
    parser.add_argument(
        "--every-frame",
        action="store_true",
        default=True,
        help="Process every single video frame without skipping (default: True).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=min(32, max(2, (os.cpu_count() or 4))),
        help=f"Number of parallel worker threads for OCR processing (default: {min(32, max(2, (os.cpu_count() or 4)))}).",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=float,
        metavar=("YMIN", "YMAX", "XMIN", "XMAX"),
        default=[0.94, 0.998, 0.0, 0.85],
        help="Normalized ROI bounding box (default: 0.94 0.998 0.0 0.85 for Garmin bottom banner).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate preview_roi.png of the first sampled frame and print OCR output without processing whole video.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Export comprehensive debug CSV (<video_stem>_debug_raw.csv) with raw OCR text and per-frame keep/drop reasons.",
    )
    parser.add_argument(
        "--debug-output",
        default=None,
        help="Custom path for raw debug CSV output file (default: <video_dir>/<video_stem>_debug_raw.csv).",
    )
    parser.add_argument(
        "--check-embedded",
        action="store_true",
        default=True,
        help="Check for embedded subtitle/telemetry track in MP4 before OCR (default: True).",
    )

    args = parser.parse_args()

    video_path = Path(args.input)
    if not video_path.exists():
        print(f"Error: Input file '{video_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    roi_bbox = tuple(args.roi)
    debug_records: List[DebugFrameRecord] = []

    # Check if input is a single image file
    is_single_image = video_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    if is_single_image:
        print(f"[*] Processing single image frame: {video_path}")
        if HAS_OPENCV:
            img = cv2.imread(str(video_path))
            roi = VideoProcessor.crop_roi(img, roi_bbox)
        else:
            pil_img = Image.open(str(video_path))
            w, h = pil_img.size
            ymin, ymax, xmin, xmax = roi_bbox
            roi = pil_img.crop((int(xmin * w), int(ymin * h), int(xmax * w), int(ymax * h)))

        processor = VideoProcessor(str(video_path), roi_bbox=roi_bbox)
        candidates = processor.preprocess_for_ocr(roi)
        ocr_text = processor.run_ocr(candidates)
        print(f"[*] Raw OCR Output:\n---\n{ocr_text.strip()}\n---")
        pt, rec = TelemetryParser.parse_frame_debug(ocr_text, frame_num=1, time_sec=0.0)
        points = [pt] if pt else []
        debug_records = [rec]
    else:
        if args.preview:
            preview_roi(str(video_path), roi_bbox)
            return

        # Check for embedded subtitle track first
        points: List[TelemetryPoint] = []
        if args.check_embedded:
            print("[*] Checking for embedded telemetry/subtitle track in video...")
            embedded = check_embedded_subtitles(str(video_path))
            if embedded and len(embedded) > 0:
                print(f"[+] Found {len(embedded)} telemetry points in embedded track!")
                points = embedded

        # Run multi-threaded OCR if no embedded track found
        if not points:
            check_ocr_availability()
            mode_desc = "every frame" if (args.every_frame or args.interval <= 0) else f"{args.interval}s interval"
            print(f"[*] Starting multi-threaded OCR extraction ({mode_desc}, {args.workers} threads)...")
            processor = VideoProcessor(str(video_path), roi_bbox=roi_bbox)

            total_frames = None
            if HAS_OPENCV:
                try:
                    cap = cv2.VideoCapture(str(video_path))
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        if args.every_frame or args.interval <= 0:
                            total_frames = total_video_frames
                        else:
                            step = max(1, int(round(fps * args.interval)))
                            total_frames = total_video_frames // step
                        cap.release()
                except Exception:
                    pass

            frame_gen = (
                processor.extract_frames_opencv(sample_interval_sec=args.interval, every_frame=args.every_frame)
                if HAS_OPENCV
                else processor.extract_frames_ffmpeg(sample_interval_sec=args.interval, every_frame=args.every_frame)
            )

            points, debug_records = process_frames_parallel(frame_gen, num_workers=args.workers, total_frames=total_frames)

    debug_map = {r.frame_number: r for r in debug_records}
    print(f"[*] Raw extracted points: {len(points)}")
    points = clean_telemetry_points(points, debug_map=debug_map)
    print(f"[+] Validated points after cleaning: {len(points)}")

    base_stem = video_path.stem
    out_dir = video_path.parent

    # Export debug CSV if requested
    if args.debug:
        debug_out_file = Path(args.debug_output) if args.debug_output else out_dir / f"{base_stem}_debug_raw.csv"
        TelemetryExporter.to_debug_csv(debug_records, str(debug_out_file))
        print(f"[✓] Saved DEBUG raw OCR report: {debug_out_file}")

        # Print breakdown summary
        from collections import Counter
        status_counts = Counter(r.status for r in debug_records)
        print("\n" + "=" * 60)
        print("  DEBUG TELEMETRY CLEANING SUMMARY")
        print("=" * 60)
        print(f"  Total Frames Analyzed:      {len(debug_records)}")
        print(f"  ✓ Valid Kept Waypoints:     {status_counts.get('KEPT', 0)}")
        print(f"  • Dropped (Incomplete):     {status_counts.get('DROPPED_INCOMPLETE', 0)}")
        print(f"  • Dropped (No OCR text):    {status_counts.get('NO_TEXT_DETECTED', 0)}")
        print(f"  • Dropped (GPS Spike Jump): {status_counts.get('DROPPED_OUTLIER_SPIKE', 0)}")
        print(f"  • Dropped (Duplicate Pos):  {status_counts.get('DROPPED_DUPLICATE', 0)}")
        if status_counts.get("DROPPED_INVALID_COORDS", 0) > 0:
            print(f"  • Dropped (Invalid Coords): {status_counts.get('DROPPED_INVALID_COORDS', 0)}")
        if status_counts.get("DROPPED_SPEED_ANOMALY", 0) > 0:
            print(f"  • Dropped (Speed Anomaly):  {status_counts.get('DROPPED_SPEED_ANOMALY', 0)}")
        print("=" * 60 + "\n")

    if not points:
        print("[-] Warning: No valid GPS telemetry points were extracted.", file=sys.stderr)
        print("Tip: Run with --preview to inspect the OCR region and make sure the telemetry overlay is visible.", file=sys.stderr)
        if args.debug:
            print(f"Tip: Inspect {debug_out_file} to see raw OCR text and per-frame failure reasons.", file=sys.stderr)
        sys.exit(1)

    # Determine output file paths
    target_formats = ["gpx", "csv", "geojson", "kml"] if args.format == "all" else [args.format]

    for fmt in target_formats:
        if args.output and args.format != "all":
            out_file = Path(args.output)
        else:
            out_file = out_dir / f"{base_stem}.{fmt}"

        if fmt == "gpx":
            TelemetryExporter.to_gpx(points, str(out_file), track_name=base_stem)
        elif fmt == "csv":
            TelemetryExporter.to_csv(points, str(out_file))
        elif fmt == "geojson":
            TelemetryExporter.to_geojson(points, str(out_file))
        elif fmt == "kml":
            TelemetryExporter.to_kml(points, str(out_file), track_name=base_stem)

        print(f"[✓] Saved {fmt.upper()} export: {out_file}")


if __name__ == "__main__":
    main()

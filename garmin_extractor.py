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


# ==============================================================================
# Telemetry Text Parsing (GPS, Timestamp, Speed)
# ==============================================================================

class TelemetryParser:
    """Parses raw OCR text strings into structured telemetry data."""

    # Pre-compiled regular expressions for OCR variations

    # Garmin speed patterns: e.g. "45 MPH", "72 KM/H", "0 mph", "105.4 km/h"
    SPEED_PATTERN = re.compile(
        r"(?i)\b(?P<speed>\d{1,3}(?:\.\d+)?)\s*(?P<unit>mph|km/h|kmh|kph|km\/h|knots|kt)\b"
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
            "°": " ",
            "*": " ",
            "’": "'",
            "`": "'",
            "[": " ",
            "]": " ",
            "{": " ",
            "}": " ",
        }
        cleaned = text
        for k, v in replacements.items():
            cleaned = cleaned.replace(k, v)
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
            val = float(m.group("speed"))
            unit = m.group("unit").lower().replace("/", "")

            if unit in ("mph",):
                mph = val
                kmh = mph * 1.609344
            elif unit in ("kmh", "kph"):
                kmh = val
                mph = kmh / 1.609344
            elif unit in ("knots", "kt"):
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

    @staticmethod
    def preprocess_for_ocr(roi_image: Any) -> List[Any]:
        """
        Prepares video frame banner for OCR:
        1. Isolates bright/white overlay text from dark dashboard.
        2. Inverts to crisp black text on pure white background (Tesseract standard).
        3. Adds border padding so Tesseract character bounding boxes don't touch image edges.
        Returns a list of candidate preprocessed images for multi-pass OCR.
        """
        if not HAS_OPENCV or roi_image is None or roi_image.size == 0:
            return [roi_image]

        candidates = []

        # Convert to grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image.copy()

        # Upscale 2.5x to 3x for clear character definition
        h, w = gray.shape[:2]
        if h < 60:
            scale_factor = max(2.0, 70.0 / max(1, h))
            scaled_gray = cv2.resize(
                gray, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC
            )
        else:
            scaled_gray = gray

        # --- Candidate 1: White-Text Brightness Isolation + Inverted (Black on White) ---
        # Garmin text is bright (>160-190 intensity). Isolate bright text:
        _, bright_thresh = cv2.threshold(scaled_gray, 170, 255, cv2.THRESH_BINARY)
        # Invert: black text on white background (Tesseract performs vastly better)
        inverted_bright = cv2.bitwise_not(bright_thresh)
        # Add 15px white border padding around image
        padded_bright = cv2.copyMakeBorder(
            inverted_bright, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
        )
        candidates.append(padded_bright)

        # --- Candidate 2: Bilateral Filter + Otsu Inverted ---
        denoised = cv2.bilateralFilter(scaled_gray, 7, 50, 50)
        _, otsu = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inverted_otsu = cv2.bitwise_not(otsu)
        padded_otsu = cv2.copyMakeBorder(
            inverted_otsu, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
        )
        candidates.append(padded_otsu)

        # --- Candidate 3: Scaled Grayscale with Contrast Stretch ---
        norm_gray = cv2.normalize(scaled_gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        padded_gray = cv2.copyMakeBorder(
            norm_gray, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255
        )
        candidates.append(padded_gray)

        return candidates

    @classmethod
    def run_ocr(cls, processed_imgs: Any) -> str:
        """
        Runs multi-pass OCR on preprocessed image candidates.
        Thread-safe method called by parallel worker threads.
        """
        if not (HAS_PYTESSERACT or HAS_TESSERACT_BIN or HAS_EASYOCR):
            return ""

        if not isinstance(processed_imgs, list):
            processed_imgs = [processed_imgs]

        best_text = ""
        psm_modes = ["--psm 7", "--psm 6", "--psm 11"]

        for img in processed_imgs:
            for psm in psm_modes:
                text = ""
                if HAS_PYTESSERACT:
                    try:
                        pil_img = Image.fromarray(img) if (HAS_OPENCV and isinstance(img, np.ndarray)) else img
                        text = pytesseract.image_to_string(pil_img, config=f"{psm} --oem 3")
                    except Exception:
                        pass
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
                            ["tesseract", tmp_name, "stdout", psm, "--oem", "3"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL,
                            text=True,
                            timeout=5,
                        )
                        text = res.stdout
                    except Exception:
                        pass
                    finally:
                        if os.path.exists(tmp_name):
                            os.remove(tmp_name)

                # Check if this pass successfully parsed coordinates
                if text and TelemetryParser.parse_coordinates(text):
                    return text

                if len(text.strip()) > len(best_text.strip()):
                    best_text = text

        return best_text

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


def _process_frame_worker(task: Tuple[int, float, Any]) -> Optional[TelemetryPoint]:
    """Worker function executed in parallel threads to preprocess, OCR, and parse one frame."""
    frame_num, sec, roi = task
    if roi is None:
        return None
    try:
        candidates = VideoProcessor.preprocess_for_ocr(roi)
        ocr_text = VideoProcessor.run_ocr(candidates)
        if not ocr_text:
            return None
        return TelemetryParser.parse_frame_text(ocr_text, frame_num, sec)
    except Exception:
        return None


def process_frames_parallel(
    frame_gen: Generator[Tuple[int, float, Any], None, None],
    num_workers: int = 4,
    total_frames: Optional[int] = None,
) -> List[TelemetryPoint]:
    """
    Executes frame preprocessing and OCR across multiple worker threads in parallel.
    Uses a rolling buffer to minimize memory usage while keeping CPU workers fully saturated.
    """
    points: List[TelemetryPoint] = []
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
                    pt = f.result()
                    if pt:
                        points.append(pt)
                    if pbar is not None:
                        pbar.update(1)
                    del future_map[f]

        # Drain all remaining tasks
        for f in concurrent.futures.as_completed(future_map.keys()):
            pt = f.result()
            if pt:
                points.append(pt)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

    # Ensure chronological sort by frame number
    points.sort(key=lambda p: p.frame_number)
    return points


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
                continue
            if p.speed_mph is None and p.speed_kmh is None:
                continue

        # Coordinate bounds check
        if not (-90.0 <= p.latitude <= 90.0 and -180.0 <= p.longitude <= 180.0):
            continue
        if abs(p.latitude) < 0.0001 and abs(p.longitude) < 0.0001:
            continue  # Null island check

        # Speed bounds check
        if p.speed_kmh is not None and (p.speed_kmh < 0.0 or p.speed_kmh > max_speed_kmh):
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
                continue
        deduped.append(p)

    if len(deduped) <= 2:
        return deduped

    # Step 3: Neighbor-based GPS Sanity Check (Multi-point spike / glitch rejection)
    def filter_spikes_single_pass(pts: List[TelemetryPoint]) -> List[TelemetryPoint]:
        n = len(pts)
        if n <= 2:
            return pts

        outliers = set()

        def is_spike(prev_pt: TelemetryPoint, curr_pt: TelemetryPoint, next_pt: TelemetryPoint) -> bool:
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
                    return True
                # Also if the jump exceeds 50m when reported speed is near zero/slow
                expected_dist = max(50.0, (curr_pt.speed_mps or 25.0) * 3.0 * dt_prev)
                if d_prev_curr > expected_dist and d_curr_next > expected_dist and d_prev_next < expected_dist:
                    return True
            return False

        # Internal points
        for i in range(1, n - 1):
            if is_spike(pts[i - 1], pts[i], pts[i + 1]):
                outliers.add(i)

        # Check endpoints
        if n >= 3 and 0 not in outliers and 1 not in outliers and 2 not in outliers:
            d01 = haversine_distance_meters(pts[0].latitude, pts[0].longitude, pts[1].latitude, pts[1].longitude)
            d12 = haversine_distance_meters(pts[1].latitude, pts[1].longitude, pts[2].latitude, pts[2].longitude)
            dt01 = max(0.05, abs((pts[1].timestamp - pts[0].timestamp).total_seconds()) if (pts[1].timestamp and pts[0].timestamp) else 1.0)
            dt12 = max(0.05, abs((pts[2].timestamp - pts[1].timestamp).total_seconds()) if (pts[2].timestamp and pts[1].timestamp) else 1.0)
            if (d01 / dt01 > max_allowed_speed_mps) and (d12 / dt12 <= max_allowed_speed_mps):
                outliers.add(0)

        if n >= 3 and (n - 1) not in outliers and (n - 2) not in outliers and (n - 3) not in outliers:
            d_last = haversine_distance_meters(pts[n - 2].latitude, pts[n - 2].longitude, pts[n - 1].latitude, pts[n - 1].longitude)
            d_prev = haversine_distance_meters(pts[n - 3].latitude, pts[n - 3].longitude, pts[n - 2].latitude, pts[n - 2].longitude)
            dt_last = max(0.05, abs((pts[n - 1].timestamp - pts[n - 2].timestamp).total_seconds()) if (pts[n - 1].timestamp and pts[n - 2].timestamp) else 1.0)
            dt_prev = max(0.05, abs((pts[n - 2].timestamp - pts[n - 3].timestamp).total_seconds()) if (pts[n - 2].timestamp and pts[n - 3].timestamp) else 1.0)
            if (d_last / dt_last > max_allowed_speed_mps) and (d_prev / dt_prev <= max_allowed_speed_mps):
                outliers.add(n - 1)

        return [pts[i] for i in range(n) if i not in outliers]

    sanitized = deduped
    for _ in range(2):
        prev_len = len(sanitized)
        sanitized = filter_spikes_single_pass(sanitized)
        if len(sanitized) == prev_len:
            break

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
        p = TelemetryParser.parse_frame_text(ocr_text, frame_num=1, time_sec=0.0)
        points = [p] if p else []
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

            points = process_frames_parallel(frame_gen, num_workers=args.workers, total_frames=total_frames)

    print(f"[*] Raw extracted points: {len(points)}")
    points = clean_telemetry_points(points)
    print(f"[+] Validated points after cleaning: {len(points)}")

    if not points:
        print("[-] Warning: No valid GPS telemetry points were extracted.", file=sys.stderr)
        print("Tip: Run with --preview to inspect the OCR region and make sure the telemetry overlay is visible.", file=sys.stderr)
        sys.exit(1)

    # Determine output file paths
    base_stem = video_path.stem
    out_dir = video_path.parent

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

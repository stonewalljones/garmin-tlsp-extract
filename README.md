# Garmin Travelapse & Dashcam Telemetry Extractor

A Python tool to extract burned-in (or embedded) **Timestamps**, **GPS Coordinates**, and **Speed** from Garmin Travelapse and dashcam videos and export them into standard geospatial formats:
- **GPX (`.gpx`)**: For Strava, Garmin Connect, Google Earth, and GIS mapping.
- **CSV (`.csv`)**: For Excel, Pandas, data analysis, and spreadsheets.
- **GeoJSON (`.geojson`)**: For web maps (Mapbox, Leaflet, geojson.io) and QGIS.
- **KML (`.kml`)**: For Google Earth 3D fly-throughs with speed/time waypoint pins.

---

## 🛠️ Features

- **Multi-Format Coordinate Parsing**:
  - Degrees & Decimal Minutes (*Garmin Dashcam default*): `N 37° 46.123' W 122° 25.456'`
  - Decimal Degrees: `37.774900 N, 122.419400 W` or `37.774900, -122.419400`
  - Signed Coordinates: `LAT: 37.7749, LON: -122.4194`
- **Speed & Unit Conversion**:
  - Automatically parses `MPH`, `KM/H`, `knots` and computes speed in m/s (GPX standard).
- **Date & Time Extraction**:
  - Automatically parses `YYYY/MM/DD HH:MM:SS`, `MM/DD/YYYY hh:mm:ss AM/PM`, etc., and generates ISO-8601 UTC timestamps.
- **OCR Enhancement Pipeline**:
  - Cropping region of interest (ROI), grayscale conversion, 2x upscaling, bilateral edge-preserving smoothing, and Otsu binarization tailored for Garmin white overlay fonts.
- **Preview Mode**:
  - Generates a preview image (`preview_roi.png`) and tests OCR on sample frames before processing long videos.
- **Embedded Telemetry Fast-Path**:
  - Automatically checks if the MP4 container has an embedded subtitle/telemetry track and extracts it directly via FFmpeg.
- **High-Performance Multi-Threading**:
  - Distributes image preprocessing and OCR recognition across multiple CPU worker threads in parallel to maximize throughput on multi-core systems.
- **Strict Completeness & Quality Filter**:
  - Automatically discards incomplete frames (must contain valid GPS coordinates, Date/Time, and Speed).
- **Neighbor-Based GPS Sanity Check (Outlier & Glitch Rejection)**:
  - Compares each waypoint against its surrounding chronological neighbors.
  - Automatically detects and removes single-frame OCR coordinate misreads (e.g. OCR reading an `8` instead of a `5` that would cause a 300km spike) and impossible velocity jumps.
- **Deduplication & Smoothing**:
  - Removes redundant static duplicates while preserving authentic movement and speed transitions.

---

## 📦 Installation

### Prerequisites
- Python 3.8+
- [FFmpeg](https://ffmpeg.org/) (installed and on your PATH)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (`sudo dnf install tesseract` or `sudo apt install tesseract-ocr`)

### Python Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Basic Export (Every Frame -> Multi-threaded -> GPX)
```bash
python garmin_extractor.py -i travelapse_video.mp4 -o my_route.gpx
```
*By default, the script analyzes **every single frame** of the video using all available CPU threads.*

### 2. Specify Number of Worker Threads
```bash
# Run with 8 parallel worker threads:
python garmin_extractor.py -i travelapse_video.mp4 --workers 8 --format all
```

### 2. Export All Formats (.gpx, .csv, .geojson, .kml)
```bash
python garmin_extractor.py -i travelapse_video.mp4 --format all
```

### 3. Optional: Downsample Frame Rate
If you have a very long video and want to sample at specific intervals (e.g. 1 frame every 2 seconds to speed up processing):
```bash
# Sample 1 frame every 2.0 seconds:
python garmin_extractor.py -i travelapse_video.mp4 --format all --interval 2.0
```

### 4. Preview ROI & Tune Bounding Box
Before running a long video, verify that the telemetry banner is cleanly captured in the ROI:
```bash
python garmin_extractor.py -i travelapse_video.mp4 --preview
```
This saves `preview_roi.png` and prints the test OCR extraction.

If your camera places the banner in a different location (e.g. top 12% or bottom 10%):
```bash
# ROI arguments: YMIN YMAX XMIN XMAX (normalized from 0.0 to 1.0)

# Bottom 10% overlay:
python garmin_extractor.py -i travelapse_video.mp4 -o route.gpx --roi 0.90 1.00 0.0 1.0

# Top 12% overlay:
python garmin_extractor.py -i travelapse_video.mp4 -o route.gpx --roi 0.0 0.12 0.0 1.0
```

---

## 🧪 Testing

Run the built-in test suite:
```bash
python -m unittest test_extractor.py
```

---

## 📄 Output Formats Details

| Format | Extension | Description | Compatible With |
| :--- | :--- | :--- | :--- |
| **GPX** | `.gpx` | Full GPS track with `<trkpt>`, timestamps, speed (m/s), and Garmin extensions | Strava, Garmin Connect, Gaia GPS, QGIS |
| **CSV** | `.csv` | Tabular data (`timestamp_iso`, `latitude`, `longitude`, `speed_mph`, `speed_kmh`, `frame_number`, `video_time_sec`) | Excel, Pandas, Tableau |
| **GeoJSON** | `.geojson` | Route LineString and Point feature collection with metadata | geojson.io, Leaflet, Mapbox, QGIS |
| **KML** | `.kml` | 3D route trajectory and clickable waypoint pins | Google Earth |

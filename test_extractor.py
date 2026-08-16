#!/usr/bin/env python3
"""
Unit and Integration Tests for Garmin Telemetry Extractor
"""

import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from garmin_extractor import (
    TelemetryExporter,
    TelemetryParser,
    TelemetryPoint,
    clean_telemetry_points,
)


class TestTelemetryParser(unittest.TestCase):
    """Tests for coordinate, speed, and timestamp regex parsers."""

    def test_parse_coordinates_ddm(self):
        """Test Degrees and Decimal Minutes (Garmin Dashcam default)."""
        # N 37° 46.123' W 122° 25.456'
        text = "N 37° 46.123' W 122° 25.456' 45 MPH 2024-05-10 14:30:15"
        coords = TelemetryParser.parse_coordinates(text)
        self.assertIsNotNone(coords)
        lat, lon = coords
        # 37 + 46.123/60 = 37.7687167
        self.assertAlmostEqual(lat, 37.768717, places=5)
        # -(122 + 25.456/60) = -122.424267
        self.assertAlmostEqual(lon, -122.424267, places=5)

    def test_parse_coordinates_dd_with_hemi(self):
        """Test Decimal Degrees with N/S/E/W indicator."""
        text = "34.05223 N, 118.24368 W 55 MPH"
        coords = TelemetryParser.parse_coordinates(text)
        self.assertIsNotNone(coords)
        lat, lon = coords
        self.assertAlmostEqual(lat, 34.05223, places=5)
        self.assertAlmostEqual(lon, -118.24368, places=5)

    def test_parse_coordinates_southern_eastern(self):
        """Test Southern and Eastern hemisphere conversions."""
        text = "S 33° 51.540' E 151° 12.480'"
        coords = TelemetryParser.parse_coordinates(text)
        self.assertIsNotNone(coords)
        lat, lon = coords
        self.assertAlmostEqual(lat, -33.859, places=3)
        self.assertAlmostEqual(lon, 151.208, places=3)

    def test_parse_speed_mph_and_kmh(self):
        """Test parsing of speed units and m/s conversions."""
        # MPH
        mph, kmh, mps = TelemetryParser.parse_speed("GARMIN DASH CAM 65 MPH 2024/01/01")
        self.assertAlmostEqual(mph, 65.0, places=1)
        self.assertAlmostEqual(kmh, 104.6, places=1)
        self.assertAlmostEqual(mps, 29.05, places=1)

        # KM/H
        mph2, kmh2, mps2 = TelemetryParser.parse_speed("100 KM/H N 37 46.123")
        self.assertAlmostEqual(kmh2, 100.0, places=1)
        self.assertAlmostEqual(mph2, 62.13, places=1)
        self.assertAlmostEqual(mps2, 27.78, places=1)

    def test_parse_timestamp(self):
        """Test timestamps in YYYY/MM/DD and MM/DD/YYYY formats."""
        dt1 = TelemetryParser.parse_timestamp("2024/08/16 12:45:30 45 MPH")
        self.assertEqual(dt1, datetime(2024, 8, 16, 12, 45, 30, tzinfo=timezone.utc))

        dt2 = TelemetryParser.parse_timestamp("08/16/2024 02:15:00 PM")
        self.assertEqual(dt2, datetime(2024, 8, 16, 14, 15, 0, tzinfo=timezone.utc))

    def test_full_frame_text_parser(self):
        """Test end-to-end parsing of a realistic Garmin overlay text string."""
        sample_banner = "GARMIN 2024/09/20 18:25:40 N 40° 42.750' W 74° 00.300' 52 MPH"
        p = TelemetryParser.parse_frame_text(sample_banner, frame_num=100, time_sec=3.33)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.latitude, 40.7125, places=4)
        self.assertAlmostEqual(p.longitude, -74.0050, places=4)
        self.assertEqual(p.timestamp, datetime(2024, 9, 20, 18, 25, 40, tzinfo=timezone.utc))
        self.assertAlmostEqual(p.speed_mph, 52.0, places=1)
        self.assertEqual(p.frame_number, 100)

    def test_user_sample_frame_format(self):
        """Test exact format from user's Travelapse video frame: GARMIN 06/20/2025 10:24:56 AM 35.63910 -106.02303 21 MPH."""
        sample_banner = "GARMIN 06/20/2025 10:24:56 AM 35.63910 -106.02303 21 MPH"
        p = TelemetryParser.parse_frame_text(sample_banner, frame_num=1, time_sec=0.0)
        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.latitude, 35.63910, places=5)
        self.assertAlmostEqual(p.longitude, -106.02303, places=5)
        self.assertEqual(p.timestamp, datetime(2025, 6, 20, 10, 24, 56, tzinfo=timezone.utc))
        self.assertAlmostEqual(p.speed_mph, 21.0, places=1)
        self.assertAlmostEqual(p.speed_kmh, 33.796, places=2)


class TestExporters(unittest.TestCase):
    """Tests for GPX, CSV, GeoJSON, and KML exporters."""

    def setUp(self):
        self.sample_points = [
            TelemetryPoint(
                latitude=37.7749,
                longitude=-122.4194,
                timestamp=datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
                speed_mph=30.0,
                speed_kmh=48.28,
                speed_mps=13.41,
                frame_number=0,
                video_time_sec=0.0,
            ),
            TelemetryPoint(
                latitude=37.7755,
                longitude=-122.4180,
                timestamp=datetime(2024, 6, 1, 10, 0, 5, tzinfo=timezone.utc),
                speed_mph=35.0,
                speed_kmh=56.32,
                speed_mps=15.64,
                frame_number=150,
                video_time_sec=5.0,
            ),
        ]
        self.tmp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_csv_export(self):
        csv_path = os.path.join(self.tmp_dir.name, "test.csv")
        TelemetryExporter.to_csv(self.sample_points, csv_path)
        self.assertTrue(os.path.exists(csv_path))

        with open(csv_path, "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn("37.774900", content)
            self.assertIn("-122.419400", content)
            self.assertIn("30.00", content)

    def test_gpx_export_and_xml_validity(self):
        gpx_path = os.path.join(self.tmp_dir.name, "test.gpx")
        TelemetryExporter.to_gpx(self.sample_points, gpx_path, track_name="Test Track")
        self.assertTrue(os.path.exists(gpx_path))

        tree = ET.parse(gpx_path)
        root = tree.getroot()
        self.assertTrue(root.tag.endswith("gpx"))

        # Find trkpt elements (handling XML namespace)
        trkpts = [elem for elem in root.iter() if elem.tag.endswith("trkpt")]
        self.assertEqual(len(trkpts), 2)
        self.assertEqual(trkpts[0].attrib["lat"], "37.774900")
        self.assertEqual(trkpts[0].attrib["lon"], "-122.419400")

    def test_geojson_export_and_json_validity(self):
        geojson_path = os.path.join(self.tmp_dir.name, "test.geojson")
        TelemetryExporter.to_geojson(self.sample_points, geojson_path)
        self.assertTrue(os.path.exists(geojson_path))

        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.assertEqual(data["type"], "FeatureCollection")
            # 1 LineString feature + 2 Point features
            self.assertEqual(len(data["features"]), 3)
            self.assertEqual(data["features"][0]["geometry"]["type"], "LineString")
            self.assertEqual(data["features"][1]["geometry"]["type"], "Point")

    def test_kml_export_and_xml_validity(self):
        kml_path = os.path.join(self.tmp_dir.name, "test.kml")
        TelemetryExporter.to_kml(self.sample_points, kml_path, track_name="KML Track")
        self.assertTrue(os.path.exists(kml_path))

        tree = ET.parse(kml_path)
        root = tree.getroot()
        self.assertTrue(root.tag.endswith("kml"))

    def test_cleaning_and_deduplication(self):
        bad_points = [
            # Missing timestamp -> should be removed
            TelemetryPoint(latitude=37.77, longitude=-122.41, speed_mph=30.0, speed_kmh=48.28),
            # Missing speed -> should be removed
            TelemetryPoint(latitude=37.77, longitude=-122.41, timestamp=datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)),
            # Valid point
            TelemetryPoint(
                latitude=37.7700,
                longitude=-122.4100,
                timestamp=datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
                speed_mph=30.0,
                speed_kmh=48.28,
            ),
            # Exact duplicate -> should be deduped
            TelemetryPoint(
                latitude=37.7700,
                longitude=-122.4100,
                timestamp=datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
                speed_mph=30.0,
                speed_kmh=48.28,
            ),
            # Impossible speed spike (> 250 km/h) -> should be removed
            TelemetryPoint(
                latitude=37.7705,
                longitude=-122.4105,
                timestamp=datetime(2024, 6, 1, 10, 0, 1, tzinfo=timezone.utc),
                speed_mph=600.0,
                speed_kmh=965.0,
            ),
            # Out of bounds lat -> should be removed
            TelemetryPoint(
                latitude=120.0,
                longitude=-122.41,
                timestamp=datetime(2024, 6, 1, 10, 0, 2, tzinfo=timezone.utc),
                speed_mph=30.0,
                speed_kmh=48.28,
            ),
        ]
        cleaned = clean_telemetry_points(bad_points, require_complete=True)
        self.assertEqual(len(cleaned), 1)
        self.assertAlmostEqual(cleaned[0].latitude, 37.7700)

    def test_neighbor_gps_sanity_check(self):
        """Verify that OCR glitch coordinates jumping away from neighbors are filtered out."""
        # 5 consecutive valid points driving along a road in 1-second intervals (~15 meters apart)
        base_time = datetime(2025, 6, 20, 10, 0, 0, tzinfo=timezone.utc)
        trajectory = [
            TelemetryPoint(latitude=35.63910, longitude=-106.02300, timestamp=datetime(2025, 6, 20, 10, 0, 0, tzinfo=timezone.utc), speed_mph=25.0, speed_kmh=40.23, speed_mps=11.17, video_time_sec=0.0),
            TelemetryPoint(latitude=35.63920, longitude=-106.02305, timestamp=datetime(2025, 6, 20, 10, 0, 1, tzinfo=timezone.utc), speed_mph=25.0, speed_kmh=40.23, speed_mps=11.17, video_time_sec=1.0),
            # Glitch point: OCR misread '5' as '8' (38.63920 is over 300 km away!)
            TelemetryPoint(latitude=38.63920, longitude=-106.02310, timestamp=datetime(2025, 6, 20, 10, 0, 2, tzinfo=timezone.utc), speed_mph=25.0, speed_kmh=40.23, speed_mps=11.17, video_time_sec=2.0),
            TelemetryPoint(latitude=35.63940, longitude=-106.02315, timestamp=datetime(2025, 6, 20, 10, 0, 3, tzinfo=timezone.utc), speed_mph=25.0, speed_kmh=40.23, speed_mps=11.17, video_time_sec=3.0),
            TelemetryPoint(latitude=35.63950, longitude=-106.02320, timestamp=datetime(2025, 6, 20, 10, 0, 4, tzinfo=timezone.utc), speed_mph=25.0, speed_kmh=40.23, speed_mps=11.17, video_time_sec=4.0),
        ]

        cleaned = clean_telemetry_points(trajectory)
        self.assertEqual(len(cleaned), 4)
        # Verify the glitch point at 38.63920 was cleanly removed
        lats = [p.latitude for p in cleaned]
        self.assertNotIn(38.63920, lats)
        self.assertEqual(lats, [35.63910, 35.63920, 35.63940, 35.63950])


if __name__ == "__main__":
    unittest.main()

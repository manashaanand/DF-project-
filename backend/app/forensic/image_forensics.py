"""
Real image forensic analysis.
Extracts actionable forensic findings (EOF data, LSB entropy, metadata anomalies)
independent of the ML pipeline.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ExifTags

from app.detectors.base import ForensicFinding


def _calculate_entropy(data: np.ndarray) -> float:
    """Calculate Shannon entropy of a 1D array of bytes/bits."""
    if data.size == 0:
        return 0.0
    _, counts = np.unique(data, return_counts=True)
    probs = counts / data.size
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)


def analyze_eof_data(file_path: Path) -> ForensicFinding | None:
    """
    Check for data appended after the End-Of-File marker.
    Supports JPEG (FF D9) and PNG (IEND).
    """
    try:
        with open(file_path, "rb") as f:
            content = f.read()
    except Exception:
        return None

    # JPEG EOF marker is FF D9
    if content.startswith(b"\xff\xd8"):
        # Find the last FF D9
        eof_idx = content.rfind(b"\xff\xd9")
        if eof_idx != -1 and eof_idx + 2 < len(content):
            appended_data = content[eof_idx + 2:]
            
            # Sometimes a few padding bytes (e.g. 0x00 or 0x0A) exist.
            # Only flag if there's substantial data or non-null/newline bytes.
            non_padding = appended_data.strip(b"\x00\r\n")
            
            if len(non_padding) > 0:
                return ForensicFinding(
                    category="appended_data",
                    description=f"Found {len(appended_data)} bytes of appended data after JPEG EOF marker.",
                    severity="high",
                    evidence={
                        "eof_marker": "FF D9",
                        "appended_length": len(appended_data),
                        "preview": str(appended_data[:32])
                    }
                )

    # PNG EOF marker is IEND block: \x00\x00\x00\x00\x49\x45\x4e\x44\xae\x42\x60\x82
    elif content.startswith(b"\x89PNG\r\n\x1a\n"):
        iend_signature = b"\x49\x45\x4E\x44\xae\x42\x60\x82"
        eof_idx = content.rfind(iend_signature)
        if eof_idx != -1 and eof_idx + 8 < len(content):
            appended_data = content[eof_idx + 8:]
            
            non_padding = appended_data.strip(b"\x00\r\n")
            if len(non_padding) > 0:
                return ForensicFinding(
                    category="appended_data",
                    description=f"Found {len(appended_data)} bytes of appended data after PNG IEND chunk.",
                    severity="high",
                    evidence={
                        "eof_marker": "IEND",
                        "appended_length": len(appended_data),
                        "preview": str(appended_data[:32])
                    }
                )

    return None


def analyze_lsb_entropy(file_path: Path) -> list[ForensicFinding]:
    """
    Calculate the entropy of the Least Significant Bit plane.
    High entropy (~1.0 for a bit plane) indicates randomness, often encrypted payload.
    Supports RGB and Grayscale analysis.
    """
    findings = []
    try:
        # Load color image to analyze channels independently if possible
        image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            return findings

        if len(image.shape) == 2:  # Grayscale
            lsb = image & 1
            entropy = _calculate_entropy(lsb.flatten())
            # For a binary array, max entropy is 1.0
            if entropy > 0.99:
                findings.append(ForensicFinding(
                    category="lsb_analysis",
                    description=f"Highly random LSB plane detected in grayscale image (Entropy: {entropy:.4f}).",
                    severity="medium",
                    evidence={"channel": "gray", "entropy": entropy}
                ))
        
        elif len(image.shape) == 3:  # BGR or BGRA
            channels = cv2.split(image)
            channel_names = ["blue", "green", "red", "alpha"]
            
            high_entropy_channels = []
            for i, ch in enumerate(channels):
                lsb = ch & 1
                entropy = _calculate_entropy(lsb.flatten())
                if entropy > 0.99:
                    high_entropy_channels.append({
                        "channel": channel_names[i] if i < 4 else f"ch_{i}",
                        "entropy": entropy
                    })
            
            if high_entropy_channels:
                findings.append(ForensicFinding(
                    category="lsb_analysis",
                    description=f"Highly random LSB plane detected in {len(high_entropy_channels)} color channel(s).",
                    severity="medium",
                    evidence={"channels": high_entropy_channels}
                ))

    except Exception:
        pass
        
    return findings


def analyze_metadata(file_path: Path) -> list[ForensicFinding]:
    """
    Analyze image metadata (EXIF/info) for suspiciously large text fields
    or anomalies.
    """
    findings = []
    try:
        with Image.open(file_path) as img:
            # Check info dictionary (PNG text chunks, etc.)
            for key, value in img.info.items():
                if isinstance(value, (str, bytes)):
                    # A metadata field larger than 10KB is very suspicious
                    if len(value) > 10240:
                        findings.append(ForensicFinding(
                            category="metadata_anomaly",
                            description=f"Unusually large metadata field '{key}' ({len(value)} bytes).",
                            severity="high",
                            evidence={"field": key, "size": len(value)}
                        ))
            
            # Check EXIF specifically if available
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, (str, bytes)):
                        if len(value) > 10240:
                            findings.append(ForensicFinding(
                                category="metadata_anomaly",
                                description=f"Unusually large EXIF tag '{tag_name}' ({len(value)} bytes).",
                                severity="high",
                                evidence={"tag": str(tag_name), "size": len(value)}
                            ))

    except Exception:
        pass

    return findings


def perform_image_forensics(file_path: Path) -> list[ForensicFinding]:
    """
    Run all forensic analysis modules on an image file.
    """
    findings = []
    
    # 1. EOF Analysis
    eof_finding = analyze_eof_data(file_path)
    if eof_finding:
        findings.append(eof_finding)
        
    # 2. LSB Entropy Analysis
    findings.extend(analyze_lsb_entropy(file_path))
    
    # 3. Metadata Analysis
    findings.extend(analyze_metadata(file_path))
    
    return findings

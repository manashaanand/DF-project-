"""
Orchestrates payload extraction based on forensic findings.
"""
from __future__ import annotations
from pathlib import Path

from app.config import settings
from app.detectors.base import DetectionResult, ExtractionResult, ForensicFinding
from app.extractors.appended_data_extractor import AppendedDataExtractor

class ExtractionManager:
    def __init__(self, recovered_dir: Path | None = None):
        self.recovered_dir = recovered_dir or settings.recovered_dir
        self.appended_extractor = AppendedDataExtractor(self.recovered_dir)

    def attempt_extraction(self, file_path: Path, result: DetectionResult) -> ExtractionResult:
        """
        Attempt to extract payload based on the findings in DetectionResult.
        """
        # If no findings, don't attempt
        if not result.forensic_findings:
            return ExtractionResult(extraction_status="NOT_ATTEMPTED")

        # Prioritize Appended Data (easiest and most reliable)
        appended_finding = next((f for f in result.forensic_findings if f.category == "appended_data"), None)
        if appended_finding:
            ext_result = self.appended_extractor.extract(file_path, appended_finding)
            if ext_result.extraction_status in ["RECOVERED", "KEY_REQUIRED"]:
                return ext_result

        # Check for LSB
        lsb_finding = next((f for f in result.forensic_findings if f.category == "lsb_analysis"), None)
        if lsb_finding:
            # We explicitly decline arbitrary LSB extraction without parameters (stego key, capacity, embedding path).
            return ExtractionResult(
                extraction_status="UNSUPPORTED",
                message="LSB payload detected (high entropy), but arbitrary LSB extraction without embedding parameters or stego key is unsupported."
            )
            
        # Metadata Injection
        metadata_finding = next((f for f in result.forensic_findings if f.category == "metadata_anomaly"), None)
        if metadata_finding:
            return ExtractionResult(
                extraction_status="UNSUPPORTED",
                message="Metadata extraction is currently unsupported in the automatic extractor."
            )

        return ExtractionResult(extraction_status="NOT_ATTEMPTED")

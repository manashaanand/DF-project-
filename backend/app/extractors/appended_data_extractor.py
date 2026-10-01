"""
Extracts payloads explicitly appended after the EOF marker.
"""

from __future__ import annotations
from pathlib import Path

from app.detectors.base import ForensicFinding, ExtractionResult
from app.extractors.payload_detector import identify_payload
from app.utils.file_utils import compute_sha256


class AppendedDataExtractor:
    def __init__(self, recovered_dir: Path):
        self.recovered_dir = recovered_dir

    def extract(self, file_path: Path, finding: ForensicFinding) -> ExtractionResult:
        """
        Extracts bytes after the specified EOF marker length.
        """
        if finding.category != "appended_data":
            return ExtractionResult(extraction_status="NOT_ATTEMPTED")

        appended_length = finding.evidence.get("appended_length", 0)
        if appended_length <= 0:
            return ExtractionResult(extraction_status="INVALID_PAYLOAD", message="Appended length is 0")

        try:
            with open(file_path, "rb") as f:
                content = f.read()
                
            payload = content[-appended_length:]
            
            # Validate payload
            status, mime_type = identify_payload(payload)
            
            if status == "INVALID_PAYLOAD":
                return ExtractionResult(
                    extraction_status="INVALID_PAYLOAD",
                    message="Extracted payload appears to be random noise or unsupported padding."
                )
                
            if status == "KEY_REQUIRED":
                return ExtractionResult(
                    extraction_status="KEY_REQUIRED",
                    size_bytes=len(payload),
                    message="Payload extracted but appears to be encrypted. Key/password required for decryption."
                )

            # Valid payload (RECOVERED). Save it.
            import uuid
            download_id = str(uuid.uuid4())
            ext = mime_type.split("/")[-1] if mime_type else "bin"
            if ext == "plain": ext = "txt"
            
            save_path = self.recovered_dir / f"{download_id}.{ext}"
            self.recovered_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, "wb") as f:
                f.write(payload)
                
            payload_sha256 = compute_sha256(save_path)
            
            return ExtractionResult(
                extraction_status="RECOVERED",
                type=mime_type or "application/octet-stream",
                size_bytes=len(payload),
                sha256=payload_sha256,
                download_id=download_id,
                message="Successfully extracted appended payload."
            )

        except Exception as e:
            return ExtractionResult(
                extraction_status="NOT_RECOVERABLE",
                message=f"Extraction failed: {str(e)}"
            )

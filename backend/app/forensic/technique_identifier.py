"""
Rules-based technique identifier.
Maps forensic findings and ML scores to specific steganography techniques.
"""

from __future__ import annotations

from app.detectors.base import ForensicFinding, TechniqueResult


def identify_techniques(
    findings: list[ForensicFinding],
    classical_score: float | None = None
) -> list[TechniqueResult]:
    """
    Identify potential steganography techniques based on gathered evidence.
    """
    techniques = []

    # 1. Appended Data / Embedded Archive
    appended_findings = [f for f in findings if f.category == "appended_data"]
    if appended_findings:
        evidence_list = []
        for f in appended_findings:
            preview = f.evidence.get("preview", "")
            # Check if preview contains magic bytes of known archives (PK for ZIP, Rar! for RAR)
            if "b'PK\\x03\\x04'" in preview or "b'Rar!\\x1a'" in preview:
                techniques.append(TechniqueResult(
                    technique="EMBEDDED_ARCHIVE",
                    confidence="high",
                    evidence=["Found archive magic bytes in appended data after image EOF marker."]
                ))
            else:
                evidence_list.append(f.description)
                
        if evidence_list and not any(t.technique == "EMBEDDED_ARCHIVE" for t in techniques):
            techniques.append(TechniqueResult(
                technique="APPENDED_DATA",
                confidence="high",
                evidence=evidence_list
            ))

    # 2. Metadata / EXIF Injection
    metadata_findings = [f for f in findings if f.category == "metadata_anomaly"]
    if metadata_findings:
        evidence_list = [f.description for f in metadata_findings]
        techniques.append(TechniqueResult(
            technique="METADATA_INJECTION",
            confidence="high",
            evidence=evidence_list
        ))

    # 3. LSB Steganography
    lsb_findings = [f for f in findings if f.category == "lsb_analysis"]
    # If we have high LSB entropy, AND the classical ML score is high, it's very likely LSB.
    # If just the ML score is high, it's suspected LSB or residual steganography.
    
    if lsb_findings:
        evidence = [f.description for f in lsb_findings]
        confidence = "high" if (classical_score is not None and classical_score >= 0.55) else "medium"
        if classical_score is not None:
            evidence.append(f"Classical ML model output a stego probability of {classical_score:.2f}.")
            
        techniques.append(TechniqueResult(
            technique="LSB_REPLACEMENT",
            confidence=confidence,
            evidence=evidence
        ))
    elif classical_score is not None and classical_score >= 0.55:
        # ML says stego, but no blatant LSB entropy spike
        techniques.append(TechniqueResult(
            technique="UNKNOWN_STEGANOGRAPHY",
            confidence="medium",
            evidence=[
                f"Statistical ML model detected anomalies (Probability: {classical_score:.2f}).",
                "May indicate adaptive LSB matching, DCT modification, or highly sparse payloads."
            ]
        ))

    return techniques

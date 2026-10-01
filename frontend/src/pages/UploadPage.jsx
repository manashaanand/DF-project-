import { useEffect, useState } from 'react'
import { analyzeImage, checkHealth } from '../services/api'

function UploadPage() {
  const [health, setHealth] = useState(null)
  const [healthError, setHealthError] = useState(null)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)

  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    checkHealth()
      .then((data) => {
        setHealth(data)
        setHealthError(null)
      })
      .catch((err) => {
        setHealth(null)
        setHealthError(err.message || 'Backend unavailable')
      })
  }, [])

  function handleFileChange(event) {
    const selectedFile = event.target.files?.[0]

    if (!selectedFile) {
      return
    }

    if (!selectedFile.type.startsWith('image/')) {
      setError('Please select a valid image file.')
      setFile(null)
      setPreview(null)
      setResult(null)
      return
    }

    setFile(selectedFile)
    setResult(null)
    setError(null)

    if (preview) {
      URL.revokeObjectURL(preview)
    }

    const previewUrl = URL.createObjectURL(selectedFile)
    setPreview(previewUrl)
  }

  async function handleAnalyze() {
    if (!file) {
      setError('Please select an image first.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await analyzeImage(file)
      setResult(data)
    } catch (err) {
      const message =
        err.response?.data?.detail ||
        err.message ||
        'Image analysis failed.'

      setError(message)
    } finally {
      setLoading(false)
    }
  }

  function handleClear() {
    if (preview) {
      URL.revokeObjectURL(preview)
    }

    setFile(null)
    setPreview(null)
    setResult(null)
    setError(null)
  }

  return (
    <div className="page-card">
      <h2>Steganography Detection</h2>

      <p>
        Upload an image to detect whether it may contain hidden
        steganographic information.
      </p>

      <div className="upload-area">
        <input
          type="file"
          accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff,.webp,image/*"
          onChange={handleFileChange}
        />

        {file && (
          <div className="selected-file">
            <strong>Selected file:</strong> {file.name}
          </div>
        )}

        {preview && (
          <div className="preview-container">
            <img
              src={preview}
              alt="Selected image preview"
              className="image-preview"
            />
          </div>
        )}

        <div className="button-row">
          <button
            className="detect-button"
            onClick={handleAnalyze}
            disabled={!file || loading}
          >
            {loading ? 'Analyzing...' : 'Detect Steganography'}
          </button>

          {file && (
            <button
              className="clear-button"
              onClick={handleClear}
              disabled={loading}
            >
              Clear
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="error-box">
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && (
        <div className="result-card">
          <h3>Detection Result</h3>

          <div className="result-row">
            <span>File</span>
            <strong>{result.file?.filename || result.filename}</strong>
          </div>
          {result.file?.sha256 && (
            <div className="result-row">
              <span>SHA-256</span>
              <strong style={{ fontSize: '0.85em', wordBreak: 'break-all' }}>
                {result.file.sha256}
              </strong>
            </div>
          )}

          <div className="result-row">
            <span>Prediction</span>
            <strong
              className={
                result.steganography_detected
                  ? 'prediction-stego'
                  : 'prediction-cover'
              }
            >
              {result.status?.replace('_', ' ')}
            </strong>
          </div>

          <div className="result-row">
            <span>Confidence</span>
            <strong>
              {(result.confidence * 100).toFixed(2)}%
            </strong>
          </div>

          {result.detectors?.map((det, idx) => (
            <div className="result-row" key={idx}>
              <span>{det.name} Probability</span>
              <strong>{det.score !== null ? det.score.toFixed(4) : 'N/A'}</strong>
            </div>
          ))}

          {result.techniques && result.techniques.length > 0 && (
            <div className="techniques-box" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#fff3cd', borderRadius: '4px', border: '1px solid #ffe69c' }}>
              <strong style={{ color: '#856404' }}>Identified Techniques</strong>
              <ul style={{ color: '#856404', margin: '0.5rem 0 0 0', paddingLeft: '1.2rem' }}>
                {result.techniques.map((tech, index) => (
                  <li key={index}>
                    <strong>{tech.technique}</strong> ({tech.confidence} confidence)
                    {tech.evidence && tech.evidence.length > 0 && (
                      <ul style={{ marginTop: '0.2rem', fontSize: '0.9em' }}>
                        {tech.evidence.map((ev, i) => <li key={i}>{ev}</li>)}
                      </ul>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.forensic_findings && result.forensic_findings.length > 0 && (
            <div className="forensics-box" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#e2e3e5', borderRadius: '4px', border: '1px solid #d6d8db' }}>
              <strong style={{ color: '#383d41' }}>Forensic Findings</strong>
              <ul style={{ color: '#383d41', margin: '0.5rem 0 0 0', paddingLeft: '1.2rem' }}>
                {result.forensic_findings.map((finding, index) => (
                  <li key={index}>
                    {finding.description} <span style={{fontSize:'0.8em', textTransform: 'uppercase', color: finding.severity === 'high' ? 'red' : 'inherit'}}>({finding.severity})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.payload && result.payload.extraction_status && result.payload.extraction_status !== 'NOT_ATTEMPTED' && (
            <div className="payload-box" style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#d4edda', borderRadius: '4px', border: '1px solid #c3e6cb' }}>
              <strong style={{ color: '#155724' }}>Payload Extraction: {result.payload.extraction_status.replace('_', ' ')}</strong>
              <ul style={{ color: '#155724', margin: '0.5rem 0 0 0', paddingLeft: '1.2rem', fontSize: '0.9em' }}>
                {result.payload.type && <li>Type: {result.payload.type}</li>}
                {result.payload.size_bytes && <li>Size: {result.payload.size_bytes} bytes</li>}
                {result.payload.message && <li>Note: {result.payload.message}</li>}
              </ul>
            </div>
          )}

          {result.warnings?.length > 0 && (
            <div className="warnings-box" style={{ marginTop: '1rem' }}>
              <strong>Warnings</strong>
              <ul>
                {result.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="backend-status">
        <strong>Backend status:</strong>{' '}

        {health && (
          <span className="status-badge ok">
            {health.status} — Model loaded:{' '}
            {health.models_loaded ? 'Yes' : 'No'}
          </span>
        )}

        {healthError && (
          <span className="status-badge error">
            Offline — {healthError}
          </span>
        )}

        {!health && !healthError && (
          <span>Checking...</span>
        )}
      </div>
    </div>
  )
}

export default UploadPage
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
            <strong>{result.filename}</strong>
          </div>

          <div className="result-row">
            <span>Prediction</span>
            <strong
              className={
                result.label === 'stego'
                  ? 'prediction-stego'
                  : 'prediction-cover'
              }
            >
              {result.label?.toUpperCase()}
            </strong>
          </div>

          <div className="result-row">
            <span>Confidence</span>
            <strong>
              {(result.confidence * 100).toFixed(2)}%
            </strong>
          </div>

          <div className="result-row">
            <span>Stego Probability</span>
            <strong>
              {result.classical_score !== null &&
              result.classical_score !== undefined
                ? result.classical_score.toFixed(4)
                : 'N/A'}
            </strong>
          </div>

          <div className="result-row">
            <span>Detection Model</span>
            <strong>
              {result.model_version || 'Unknown'}
            </strong>
          </div>

          <div className="result-row">
            <span>Features</span>
            <strong>294</strong>
          </div>

          <div className="result-row">
            <span>StegExpose</span>
            <strong>
              {result.stegexpose_available
                ? 'Available'
                : 'Not configured'}
            </strong>
          </div>

          {result.warnings?.length > 0 && (
            <div className="warnings-box">
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
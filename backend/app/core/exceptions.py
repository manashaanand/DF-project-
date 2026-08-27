from fastapi import HTTPException, status


class AppError(Exception):
    """Base application error."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class UnsupportedMediaError(AppError):
    def __init__(self, message: str = "Unsupported media type"):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)


class FileTooLargeError(AppError):
    def __init__(self, message: str = "File exceeds maximum upload size"):
        super().__init__(message, status.HTTP_413_REQUEST_ENTITY_TOO_LARGE)


class CorruptMediaError(AppError):
    def __init__(self, message: str = "File is corrupt or unreadable"):
        super().__init__(message, status.HTTP_400_BAD_REQUEST)


class AnalysisNotFoundError(AppError):
    def __init__(self, analysis_id: str):
        super().__init__(f"Analysis not found: {analysis_id}", status.HTTP_404_NOT_FOUND)


class ModelUnavailableError(AppError):
    def __init__(self, message: str = "ML model is not available"):
        super().__init__(message, status.HTTP_503_SERVICE_UNAVAILABLE)


def app_error_to_http(exc: AppError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)

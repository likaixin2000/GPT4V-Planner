# --------------------------------------
#              Exceptions
# --------------------------------------

class NetworkError(Exception):
    """Exception raised for errors in network operations."""
    pass

class PlanException(Exception):
    """Base exception for custom plan-related errors."""
    pass

class APICallError(PlanException):
    """Exception raised for errors during API calls."""
    pass

class CodeExtractionError(PlanException):
    """Base exception for errors during code extraction."""
    pass

class EmptyCodeError(CodeExtractionError):
    """Exception raised when no code is found."""
    pass

class BadCodeError(CodeExtractionError):
    """Exception raised for errors in the extracted code."""
    pass

class EmptyObjectOfInterestError(CodeExtractionError):
    """Exception raised when the object of interest is empty."""
    pass

class DetectionError(PlanException):
    """Base exception for errors in detection processes."""
    pass

class MissingObjectError(DetectionError):
    """Exception raised when an object is not detected."""
    pass

class NameConflictError(PlanException):
    """Exception raised when a object name has multiple instances. Only raised in text-based planner. """
    pass
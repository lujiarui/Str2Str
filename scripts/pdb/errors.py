class DataProcessingError(Exception):
    """Data exception."""
    pass


class FileExistsError(DataProcessingError):
    """Raised when file already exists."""
    pass


class MmcifParsingError(DataProcessingError):
    """Raised when mmcif parsing fails."""
    pass


class ResolutionError(DataProcessingError):
    """Raised when resolution isn't acceptable."""
    pass


class LengthError(DataProcessingError):
    """Raised when length isn't acceptable."""
    pass
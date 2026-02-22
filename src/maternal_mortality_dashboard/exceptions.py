"""Project-specific exception hierarchy."""


class DashboardProjectError(Exception):
    """Base exception for all project errors."""


class ConfigurationError(DashboardProjectError):
    """Configuration or environment variable error."""


class DataIngestionError(DashboardProjectError):
    """External source extraction or ingestion error."""


class DataCleaningError(DashboardProjectError):
    """Validation or transformation error in cleaning layer."""


class ModelingError(DashboardProjectError):
    """Error during metric computation or feature generation."""


class PipelineExecutionError(DashboardProjectError):
    """Error while orchestrating the end-to-end pipeline."""


class PipelineIOError(DashboardProjectError):
    """Error while persisting or loading local artifacts."""


class DashboardDataError(DashboardProjectError):
    """Error loading or preparing dashboard data artifacts."""

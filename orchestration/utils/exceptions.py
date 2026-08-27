"""
orchestration.utils.exceptions - Custom exception classes

Base Exception Hierarchy:
    AirymaxError
    +-- AgentError
    |   +-- AgentInitializationError
    |   +-- AgentExecutionError
    |   +-- AgentNotFoundError
    +-- TaskError
    |   +-- TaskCreationError
    |   +-- TaskExecutionError
    |   +-- TaskNotFoundError
    +-- ToolError
    |   +-- ToolInitializationError
    |   +-- ToolExecutionError
    |   +-- ToolNotFoundError
    +-- StorageError
    |   +-- StorageConnectionError
    |   +-- StorageReadError
    |   +-- StorageWriteError
    +-- ValidationError
        +-- InputValidationError
        +-- ConfigurationError
"""


class AirymaxError(Exception):
    def __init__(self, message: str = "", code: str = "", details: dict = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self):
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class AgentError(AirymaxError):
    pass


class AgentInitializationError(AgentError):
    pass


class AgentExecutionError(AgentError):
    pass


class AgentNotFoundError(AgentError):
    pass


class TaskError(AirymaxError):
    pass


class TaskCreationError(TaskError):
    pass


class TaskExecutionError(TaskError):
    pass


class TaskNotFoundError(TaskError):
    pass


class ToolError(AirymaxError):
    pass


class ToolInitializationError(ToolError):
    pass


class ToolExecutionError(ToolError):
    pass


class ToolNotFoundError(ToolError):
    pass


class StorageError(AirymaxError):
    pass


class StorageConnectionError(StorageError):
    pass


class StorageReadError(StorageError):
    pass


class StorageWriteError(StorageError):
    pass


class ValidationError(AirymaxError):
    pass


class InputValidationError(ValidationError):
    pass


class ConfigurationError(ValidationError):
    pass


__all__ = [
    "AirymaxError",
    "AgentError", "AgentInitializationError", "AgentExecutionError", "AgentNotFoundError",
    "TaskError", "TaskCreationError", "TaskExecutionError", "TaskNotFoundError",
    "ToolError", "ToolInitializationError", "ToolExecutionError", "ToolNotFoundError",
    "StorageError", "StorageConnectionError", "StorageReadError", "StorageWriteError",
    "ValidationError", "InputValidationError", "ConfigurationError",
]

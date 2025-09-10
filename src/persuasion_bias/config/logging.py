import json
import logging
from logging.config import dictConfig
from pathlib import Path
from typing import Optional, Union

import yaml

from persuasion_bias.config.path import CONFIG_PATH


###
class SimpleLogger:
    """Custom Logger that can be configured from any JSON/YAML file."""

    _configured = False
    _CONFIG = CONFIG_PATH

    def __init__(self, name: str) -> None:
        self.name = name

        # Configure logging once when first logger is created
        if not self._configured:
            self._configure_logging()

    @classmethod
    def _configure_logging(cls, config_file: Optional[Union[str, Path]] = None) -> None:
        """Configure logging once at startup."""

        if cls._configured:
            return

        config_path = Path(config_file or cls._CONFIG)

        try:
            if config_path.exists():
                with Path.open(config_path, "rt") as stream:
                    config = yaml.safe_load(stream=stream)
                dictConfig(config=config)
            else:
                # Fallback to basicConfig
                logging.basicConfig(
                    level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(lineno)d [%(levelname)s]: %(message)s",
                )

        except (json.JSONDecodeError, yaml.YAMLError, ValueError, TypeError):
            # Fallback on configuration errors
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(lineno)d [%(levelname)s]: %(message)s",
            )

        cls._configured = True

    @property
    def get(self) -> logging.Logger:
        """Get logger instance - Configuration already done in the constructor."""

        return logging.getLogger(name=self.name)

    @classmethod
    def configure_from_file(cls, config_file: Union[str, Path]) -> None:
        """Allow manual configuration with another file."""

        cls._configured = False  # Resets
        cls._config_file = config_file
        cls._configure_logging(config_file=config_file)

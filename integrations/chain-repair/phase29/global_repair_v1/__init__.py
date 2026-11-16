"""Unified frozen-perception evidence, memory and lifecycle repair."""
from .paths import install
DEPENDENCIES=install()
from .pipeline import EvidencePipeline

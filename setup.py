"""
Minimal setup.py shim retained for compatibility with tools that do not
yet support PEP 517 (e.g. ``pip install -e .`` on older pip versions).

All authoritative metadata lives in pyproject.toml.
"""
from setuptools import setup

setup()

"""Phenikaa Campus Courier.

Import from ``courier.common`` and the relevant module package explicitly.
Keeping this root package lightweight prevents the CV, NLP, and solver
workstreams from loading or depending on each other's optional libraries.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]

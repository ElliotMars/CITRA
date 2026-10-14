"""CITRA correctness prototype; independent of the legacy ReTaMTSF pipeline."""

from .temporal import Document, TemporalDocumentStore
from .index import CITRAIndex, TemporalCITRAIndex, QueryResult, QueryStats
from .kernel import RBFKernel, exact_aggregate

__all__ = ["Document", "TemporalDocumentStore", "CITRAIndex", "TemporalCITRAIndex", "QueryResult",
           "QueryStats", "RBFKernel", "exact_aggregate"]

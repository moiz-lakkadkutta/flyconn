"""DuckDB-backed, read-only access to a converted dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from flyconn.data.cache import store_dir
from flyconn.data.registry import Citation, DatasetSpec, get_dataset
from flyconn.data.schema import EDGES_SCHEMA, NEURONS_SCHEMA

_TABLES = ("neurons", "edges", "edges_roi")


class Store:
    """Query the harmonized tables of one dataset version offline.

    Tables are exposed as DuckDB views named ``neurons``, ``edges`` and (when
    present) ``edges_roi`` over the Parquet files in ``root``.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        prov_path = self.root / "provenance.json"
        if not prov_path.exists():
            msg = f"{self.root} has no provenance.json; run the converter first"
            raise FileNotFoundError(msg)
        self.provenance: dict[str, Any] = json.loads(prov_path.read_text())
        self._con = duckdb.connect(database=":memory:")
        self.tables: tuple[str, ...] = tuple(
            t for t in _TABLES if (self.root / f"{t}.parquet").exists()
        )
        for t in self.tables:
            path = str(self.root / f"{t}.parquet").replace("'", "''")
            self._con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{path}')")

    @classmethod
    def open(cls, ref: str) -> Store:
        """Open the cached store for ``name@version`` (see :func:`flyconn.data.pull`)."""
        return cls(store_dir(get_dataset(ref)))

    @property
    def ref(self) -> str:
        return f"{self.provenance['dataset']}@{self.provenance['version']}"

    @property
    def spec(self) -> DatasetSpec:
        return get_dataset(self.ref)

    def query(self, sql: str, params: Sequence[Any] | None = None) -> pd.DataFrame:
        """Run SQL against the ``neurons`` / ``edges`` / ``edges_roi`` views."""
        return self._con.execute(sql, params).df()

    def neurons(
        self, columns: Sequence[str] | None = None, where: str | None = None
    ) -> pd.DataFrame:
        cols = ", ".join(_quote(c) for c in columns) if columns else "*"
        sql = f"SELECT {cols} FROM neurons"
        if where:
            sql += f" WHERE {where}"
        return self.query(sql + " ORDER BY neuron_id")

    def edges(
        self,
        *,
        min_weight: int = 1,
        pre: Sequence[int] | None = None,
        post: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        if "edges" not in self.tables:
            return EDGES_SCHEMA.empty_table().to_pandas()
        clauses = ["weight >= ?"]
        params: list[Any] = [min_weight]
        if pre is not None:
            self._con.register("_pre_ids", pd.DataFrame({"id": list(pre)}))
            clauses.append("pre IN (SELECT id FROM _pre_ids)")
        if post is not None:
            self._con.register("_post_ids", pd.DataFrame({"id": list(post)}))
            clauses.append("post IN (SELECT id FROM _post_ids)")
        sql = "SELECT * FROM edges WHERE " + " AND ".join(clauses) + " ORDER BY pre, post"
        return self.query(sql, params)

    def citations(self) -> tuple[Citation, ...]:
        """Citations the registry attaches to this dataset."""
        return self.spec.citations

    def citation_text(self) -> str:
        """Plain-text citation list (one per line) for reports."""
        return "\n".join(
            f"- {c.text} doi:{c.doi}" if c.doi else f"- {c.text}" for c in self.citations()
        )

    def close(self) -> None:
        self._con.close()

    def __repr__(self) -> str:
        counts = self.provenance.get("counts", {})
        return f"Store({self.ref}, tables={self.tables}, counts={counts})"


def _quote(name: str) -> str:
    if name not in NEURONS_SCHEMA.names and not name.isidentifier():
        msg = f"invalid column name {name!r}"
        raise ValueError(msg)
    return f'"{name}"'

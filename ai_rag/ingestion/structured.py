from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Sequence

from sqlalchemy import MetaData, Table, create_engine, inspect, select, text as sql_text
from sqlalchemy.engine import Engine

from .models import Document, IngestionResult
from .storage import persist_result
from .utils import clean_text, slugify


class StructuredDataIngestor:
    """Ingest structured data sources such as CSV files and SQL tables."""

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        storage_dir: Path | str = Path("data/ingestion/structured"),
        persist: bool = True,
        engine: Optional[Engine] = None,
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.storage_dir = Path(storage_dir)
        self.persist = persist
        self._engine = engine

    # CSV INGESTION -----------------------------------------------------------------
    def ingest_csv(
        self,
        path: Path | str,
        *,
        encoding: str = "utf-8",
        delimiter: str = ",",
        limit: Optional[int] = None,
    ) -> IngestionResult:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        with csv_path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            fieldnames = reader.fieldnames or []
            narratives = []
            for index, row in enumerate(reader, start=1):
                if limit is not None and index > limit:
                    break
                narratives.append(self._row_to_narrative(row, index, fieldnames))

        document = Document(
            id=f"csv-{slugify(csv_path.stem)}",
            text="\n".join(narratives),
            source_type="structured",
            path=str(csv_path.resolve()),
            metadata={
                "format": "csv",
                "source": str(csv_path.resolve()),
                "row_count": len(narratives),
                "columns": fieldnames,
            },
            schema={"columns": [{"name": name} for name in fieldnames]},
        )

        result = IngestionResult(
            documents=[document],
            chunks=document.chunk(self.chunk_size, self.chunk_overlap),
            source=str(csv_path.resolve()),
        )

        if self.persist:
            persist_result(self.storage_dir / "csv", slugify(csv_path.stem), result)

        return result

    # SQL INGESTION -----------------------------------------------------------------
    def ingest_sql(
        self,
        connection_string: str,
        table: str,
        *,
        columns: Optional[Sequence[str]] = None,
        where: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> IngestionResult:
        engine = self._engine or create_engine(connection_string)
        metadata = MetaData()
        table_obj = Table(table, metadata, autoload_with=engine)

        inspector = inspect(engine)
        column_info = inspector.get_columns(table)
        available_columns = [column["name"] for column in column_info]
        selected_columns = list(columns) if columns else available_columns

        for column in selected_columns:
            if column not in table_obj.columns:
                raise ValueError(f"Column '{column}' not found in table '{table}'")

        query = select(*(table_obj.c[column] for column in selected_columns))
        if where:
            query = query.where(sql_text(where))
        if limit is not None:
            query = query.limit(limit)

        with engine.connect() as connection:
            rows = connection.execute(query).mappings().all()

        narratives = [
            self._row_to_narrative(dict(row), index, selected_columns)
            for index, row in enumerate(rows, start=1)
        ]

        schema_columns = [
            {
                "name": column["name"],
                "type": str(column.get("type")),
                "nullable": column.get("nullable", True),
            }
            for column in column_info
        ]

        document = Document(
            id=f"sql-{slugify(table)}",
            text="\n".join(narratives),
            source_type="structured",
            metadata={
                "format": "sql",
                "table": table,
                "row_count": len(narratives),
                "columns": selected_columns,
                "database": connection_string,
            },
            schema={"columns": schema_columns},
        )

        result = IngestionResult(
            documents=[document],
            chunks=document.chunk(self.chunk_size, self.chunk_overlap),
            source=table,
        )

        if self.persist:
            persist_result(self.storage_dir / "sql", slugify(table), result)

        if self._engine is None:
            engine.dispose()

        return result

    # HELPERS -----------------------------------------------------------------------
    @staticmethod
    def _row_to_narrative(row: dict, index: int, columns: Sequence[str]) -> str:
        parts = []
        for column in columns:
            raw_value = row.get(column, "")
            value = "" if raw_value is None else str(raw_value)
            normalized = clean_text(value)
            parts.append(f"{column}={normalized}")
        filtered_parts = [part for part in parts if part.split("=", 1)[1]]
        if not filtered_parts:
            filtered_parts = parts
        return f"Row {index}: {'; '.join(filtered_parts)}"

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

from .models import Document, IngestionResult
from .storage import persist_result
from .utils import canonicalize_url, clean_text, slugify


class WebsiteIngestor:
    """Fetch and normalize web content into canonical documents."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
        rate_limit_per_second: float = 1.0,
        storage_dir: Path | str = Path("data/ingestion/web"),
        persist: bool = True,
        session: Optional[requests.Session] = None,
        request_timeout: float = 10.0,
    ) -> None:
        self.chunk_size = max(1, chunk_size)
        self.chunk_overlap = max(0, chunk_overlap)
        self.rate_limit_per_second = max(0.0, rate_limit_per_second)
        self.storage_dir = Path(storage_dir)
        self.persist = persist
        self.session = session or requests.Session()
        self.request_timeout = request_timeout
        self._last_request_ts: float = 0.0

    def ingest(self, url: str, *, html: Optional[str] = None) -> IngestionResult:
        canonical_url = canonicalize_url(url)
        html_content = html or self._fetch(canonical_url)
        if not html_content:
            raise ValueError(f"No content retrieved from {url}")

        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe"]):
            tag.decompose()

        text = clean_text(soup.get_text("\n"))
        if not text:
            raise ValueError(f"Unable to extract text content from {url}")

        title = clean_text(soup.title.string) if soup.title and soup.title.string else None
        language = soup.html.get("lang") if soup.html else None

        document = Document(
            id=f"{slugify(canonical_url)}",
            text=text,
            source_type="web",
            uri=canonical_url,
            metadata={
                "source": canonical_url,
                "title": title,
                "language": language,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        chunks = document.chunk(self.chunk_size, self.chunk_overlap)
        result = IngestionResult(documents=[document], chunks=chunks, source=canonical_url)

        if self.persist:
            persist_result(self.storage_dir, slugify(canonical_url), result)

        return result

    def _fetch(self, url: str) -> str:
        if self.rate_limit_per_second > 0 and self._last_request_ts:
            interval = 1.0 / self.rate_limit_per_second
            elapsed = time.perf_counter() - self._last_request_ts
            if elapsed < interval:
                time.sleep(interval - elapsed)

        response = self.session.get(url, timeout=self.request_timeout, headers={"User-Agent": "cto.new-ingestor/1.0"})
        response.raise_for_status()
        self._last_request_ts = time.perf_counter()
        response.encoding = response.encoding or "utf-8"
        return response.text

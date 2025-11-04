from __future__ import annotations

import itertools
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
from langchain_core.documents import Document

from .types import RetrievalResult


def _split_sentences(text: str) -> List[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [part.strip() for part in parts if part.strip()]


def _tokenize(text: str) -> Set[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return {token for token in tokens if len(token) > 1}


_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "by",
    "is",
    "are",
    "be",
    "this",
    "that",
}


@dataclass(slots=True)
class GraphSentenceRecord:
    key: str
    sentence: str
    context: str
    entities: Set[str]
    metadata: Dict[str, Any]
    tokens: Set[str] = field(default_factory=set)


class GraphRetriever:
    """Retrieves contextual knowledge using an entity co-occurrence graph."""

    retriever_name = "graph"

    def __init__(self, *, max_depth: int = 2, top_k: int = 5) -> None:
        self.max_depth = max(1, max_depth)
        self.top_k = max(1, top_k)
        self.graph = nx.Graph()
        self._entity_records: Dict[str, List[GraphSentenceRecord]] = defaultdict(list)
        self._records: List[GraphSentenceRecord] = []
        self._indexed_documents: Set[str] = set()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.graph.clear()
        self._entity_records.clear()
        self._records.clear()
        self._indexed_documents.clear()

    def index_documents(self, documents: Sequence[Document]) -> None:
        for document in documents:
            doc_key = self._document_key(document)
            if doc_key in self._indexed_documents:
                continue

            sentences = _split_sentences(document.page_content)
            if not sentences:
                continue

            metadata = dict(document.metadata or {})
            metadata.setdefault("source", metadata.get("source_id", "unknown"))

            for idx, sentence in enumerate(sentences):
                entities = self._extract_entities(sentence)
                if not entities:
                    continue

                before_start = max(0, idx - 1)
                after_end = min(len(sentences), idx + 2)
                context_sentences = sentences[before_start:after_end]
                context = " ".join(context_sentences).strip()

                record = GraphSentenceRecord(
                    key=f"{doc_key}-{idx}",
                    sentence=sentence,
                    context=context or sentence,
                    entities=entities,
                    metadata=metadata,
                    tokens=_tokenize(sentence),
                )

                self._records.append(record)
                for entity in entities:
                    self.graph.add_node(entity)
                    self._entity_records[entity].append(record)

                for left, right in itertools.combinations(sorted(entities), 2):
                    if self.graph.has_edge(left, right):
                        self.graph[left][right]["weight"] = self.graph[left][right].get("weight", 1) + 1
                    else:
                        self.graph.add_edge(left, right, weight=1)

            self._indexed_documents.add(doc_key)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    def retrieve(self, query: str, *, k: Optional[int] = None, max_depth: Optional[int] = None) -> List[RetrievalResult]:
        if not query.strip() or not self.graph.number_of_nodes():
            return []

        depth = max(1, max_depth or self.max_depth)
        limit = max(1, k or self.top_k)

        query_entities = self._extract_entities(query)
        query_tokens = _tokenize(query)

        distance_maps: Dict[str, Dict[str, int]] = {}
        for entity in query_entities:
            if entity in self.graph:
                distance_maps[entity] = nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=depth
                )

        scored: List[Tuple[float, GraphSentenceRecord, List[str]]] = []
        for record in self._records:
            score = 0.0
            best_path: List[str] = []
            best_distance = None

            if query_entities:
                for entity in record.entities:
                    if entity in query_entities:
                        score += 1.5
                        best_path = [entity]
                        best_distance = 0
                        continue

                    for root, distances in distance_maps.items():
                        distance = distances.get(entity)
                        if distance is None:
                            continue
                        contribution = max(depth - distance + 1, 0) / (depth + 1)
                        score += contribution
                        if best_distance is None or distance < best_distance:
                            best_distance = distance
                            try:
                                best_path = nx.shortest_path(self.graph, root, entity)
                            except nx.NetworkXNoPath:  # pragma: no cover - defensive
                                best_path = [root, entity]

            if not query_entities and query_tokens:
                overlap = len(query_tokens & record.tokens)
                if overlap:
                    union = len(query_tokens | record.tokens)
                    score += overlap / union if union else 0.0

            if score <= 0:
                continue

            scored.append((score, record, best_path))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_records = scored[:limit]

        results: List[RetrievalResult] = []
        for score, record, path in top_records:
            metadata = dict(record.metadata)
            metadata.update(
                {
                    "sentence": record.sentence,
                    "entities": sorted(record.entities),
                    "graph_path": path,
                }
            )
            results.append(
                RetrievalResult(
                    content=record.context,
                    score=score,
                    metadata=metadata,
                    retriever=self.retriever_name,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _extract_entities(self, text: str) -> Set[str]:
        candidates = re.findall(r"\b([A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*)\b", text)
        entities = {
            " ".join(token.strip() for token in candidate.split())
            for candidate in candidates
            if candidate and candidate.lower() not in _STOPWORDS
        }
        normalized = {entity for entity in entities if len(entity) > 1}
        return normalized

    def _document_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        source = metadata.get("source") or metadata.get("source_id") or "doc"
        page = metadata.get("page") or metadata.get("id") or 0
        return f"{source}:{page}"


__all__ = ["GraphRetriever"]

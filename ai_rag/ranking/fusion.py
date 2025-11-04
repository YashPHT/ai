from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from langchain_core.documents import Document


class FusionReranker(Protocol):
    def score(self, query: str, documents: Sequence[Document]) -> Sequence[float]:
        ...


@dataclass
class FusionCandidate:
    key: str
    doc: Document
    raw_score: float = 0.0
    contributions: Dict[str, float] = field(default_factory=dict)
    base_score: float = 0.0
    reranker_score: Optional[float] = None
    final_score: float = 0.0
    tokens: int = 0


@dataclass
class FusionResult:
    documents: List[Document]
    selected: List[Dict[str, Any]]
    truncated: List[Dict[str, Any]]
    total_candidates: int
    deduplicated_candidates: int
    token_usage: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class KeywordOverlapReranker:
    def __init__(self, *, min_token_length: int = 2) -> None:
        self._pattern = re.compile(r"\b\w+\b")
        self._min_length = max(1, min_token_length)

    def score(self, query: str, documents: Sequence[Document]) -> Sequence[float]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [0.0 for _ in documents]

        query_counter = Counter(query_tokens)
        query_total = sum(query_counter.values())

        scores: List[float] = []
        for document in documents:
            doc_tokens = self._tokenize(document.page_content)
            if not doc_tokens or query_total == 0:
                scores.append(0.0)
                continue

            doc_counter = Counter(doc_tokens)
            overlap = sum(min(query_counter[token], doc_counter[token]) for token in query_counter)
            scores.append(overlap / query_total if query_total else 0.0)
        return scores

    def _tokenize(self, text: str) -> List[str]:
        tokens = [token.lower() for token in self._pattern.findall(text or "")]
        return [token for token in tokens if len(token) >= self._min_length]


class FusionPipeline:
    def __init__(
        self,
        *,
        token_budget: Optional[int],
        max_results: Optional[int],
        rrf_k: int = 60,
        reranker: Optional[FusionReranker] = None,
        reranker_weight: float = 0.35,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.token_budget = token_budget if token_budget and token_budget > 0 else None
        self.max_results = max_results if max_results and max_results > 0 else None
        self.rrf_k = max(1, rrf_k)
        self.reranker = reranker
        self.reranker_weight = min(max(reranker_weight, 0.0), 1.0)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def fuse(
        self,
        query: str,
        retriever_outputs: Dict[str, List[Document]],
        retriever_weights: Optional[Dict[str, float]] = None,
    ) -> FusionResult:
        per_retriever_counts = {name: len(documents) for name, documents in retriever_outputs.items()}
        total_candidates = sum(per_retriever_counts.values())

        weights_input = retriever_weights or {}
        weights: Dict[str, float] = {}
        for name in retriever_outputs:
            value = weights_input.get(name, 1.0)
            try:
                weight = float(value)
            except (TypeError, ValueError):
                weight = 1.0
            weights[name] = max(weight, 0.0)

        if total_candidates == 0:
            metadata = {
                "weights": weights,
                "per_retriever_counts": per_retriever_counts,
                "reranker_applied": False,
                "token_budget": self.token_budget,
                "rrf_k": self.rrf_k,
                "reranker_weight": 0.0,
            }
            return FusionResult(
                documents=[],
                selected=[],
                truncated=[],
                total_candidates=0,
                deduplicated_candidates=0,
                token_usage=0,
                metadata=metadata,
            )

        metadata_ranges = self._metadata_ranges(retriever_outputs)

        candidates: Dict[str, FusionCandidate] = {}
        for name, documents in retriever_outputs.items():
            weight = weights.get(name, 1.0)
            if weight == 0 or not documents:
                continue

            stats = metadata_ranges.get(name)
            for rank, document in enumerate(documents):
                key = self._document_key(document)
                rrf_component = 1.0 / (self.rrf_k + rank + 1)
                normalized_metadata = 0.0
                meta_score = self._extract_metadata_score(document)
                if stats and meta_score is not None:
                    normalized_metadata = self._normalize_value(meta_score, stats)

                weighted_score = weight * (rrf_component + normalized_metadata)

                candidate = candidates.get(key)
                if not candidate:
                    candidate = FusionCandidate(key=key, doc=document)
                    candidates[key] = candidate
                candidate.raw_score += weighted_score
                candidate.contributions[name] = weighted_score

        candidate_list = list(candidates.values())
        raw_scores = [candidate.raw_score for candidate in candidate_list]
        normalized_base_scores = self._normalize_scores(raw_scores)
        for candidate, base in zip(candidate_list, normalized_base_scores):
            candidate.base_score = base
            candidate.final_score = base

        reranker_applied = False
        if self.reranker and candidate_list:
            reranker_scores = list(self.reranker.score(query, [candidate.doc for candidate in candidate_list]))
            if len(reranker_scores) == len(candidate_list):
                reranker_norm = self._normalize_scores(reranker_scores)
                for candidate, rerank in zip(candidate_list, reranker_norm):
                    candidate.reranker_score = rerank
                    candidate.final_score = (
                        (1.0 - self.reranker_weight) * candidate.final_score + self.reranker_weight * rerank
                    )
                reranker_applied = True
            else:
                self.logger.warning(
                    "Reranker returned %s scores for %s candidates; ignoring reranker output",
                    len(reranker_scores),
                    len(candidate_list),
                )

        for candidate in candidate_list:
            candidate.tokens = self._estimate_tokens(candidate.doc)

        candidate_list.sort(key=lambda item: item.final_score, reverse=True)

        token_usage = 0
        selected_entries: List[Dict[str, Any]] = []
        truncated_entries: List[Dict[str, Any]] = []
        documents: List[Document] = []

        for index, candidate in enumerate(candidate_list):
            rank = index + 1
            entry = self._build_entry(candidate, rank)

            can_select = True
            if self.token_budget is not None and token_usage + candidate.tokens > self.token_budget:
                if selected_entries:
                    can_select = False
                else:
                    can_select = True

            if can_select:
                selected_entries.append(entry)
                documents.append(candidate.doc)
                token_usage += candidate.tokens

                if self.max_results is not None and len(selected_entries) >= self.max_results:
                    for follow_rank, follow_candidate in enumerate(candidate_list[index + 1 :], start=rank + 1):
                        truncated_entries.append(self._build_entry(follow_candidate, follow_rank))
                    break
            else:
                truncated_entries.append(entry)

        metadata = {
            "weights": weights,
            "per_retriever_counts": per_retriever_counts,
            "reranker_applied": reranker_applied,
            "token_budget": self.token_budget,
            "rrf_k": self.rrf_k,
            "reranker_weight": self.reranker_weight if reranker_applied else 0.0,
        }

        return FusionResult(
            documents=documents,
            selected=selected_entries,
            truncated=truncated_entries,
            total_candidates=total_candidates,
            deduplicated_candidates=len(candidate_list),
            token_usage=token_usage,
            metadata=metadata,
        )

    def _metadata_ranges(self, retriever_outputs: Dict[str, List[Document]]) -> Dict[str, tuple[float, float]]:
        ranges: Dict[str, tuple[float, float]] = {}
        for name, documents in retriever_outputs.items():
            scores: List[float] = []
            for document in documents:
                score = self._extract_metadata_score(document)
                if score is not None:
                    scores.append(score)
            if scores:
                ranges[name] = (min(scores), max(scores))
        return ranges

    def _extract_metadata_score(self, document: Document) -> Optional[float]:
        metadata = document.metadata or {}
        for key in ("score", "similarity", "relevance", "distance"):
            if key not in metadata:
                continue
            value = metadata[key]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if key == "distance":
                return -numeric
            return numeric
        return None

    def _normalize_value(self, value: float, stats: tuple[float, float]) -> float:
        low, high = stats
        if math.isclose(high, low):
            return 1.0
        return (value - low) / (high - low)

    def _normalize_scores(self, scores: Sequence[float]) -> List[float]:
        if not scores:
            return []
        minimum = min(scores)
        maximum = max(scores)
        if math.isclose(maximum, minimum):
            return [0.5 for _ in scores]
        return [(score - minimum) / (maximum - minimum) for score in scores]

    def _estimate_tokens(self, document: Document) -> int:
        text = document.page_content or ""
        token_count = len(text.split())
        return max(token_count, 1)

    def _document_key(self, document: Document) -> str:
        metadata = document.metadata or {}
        source = str(metadata.get("source", "unknown"))
        reference = str(metadata.get("page", metadata.get("id", "?")))
        return f"{source}:{reference}:{hash(document.page_content)}"

    def _build_entry(self, candidate: FusionCandidate, rank: int) -> Dict[str, Any]:
        metadata = dict(candidate.doc.metadata) if candidate.doc.metadata else {}
        return {
            "document_id": candidate.key,
            "final_score": candidate.final_score,
            "raw_score": candidate.raw_score,
            "base_score": candidate.base_score,
            "reranker_score": candidate.reranker_score,
            "tokens": candidate.tokens,
            "rank": rank,
            "contributions": dict(candidate.contributions),
            "metadata": metadata,
        }


__all__ = [
    "FusionPipeline",
    "FusionResult",
    "FusionReranker",
    "KeywordOverlapReranker",
]

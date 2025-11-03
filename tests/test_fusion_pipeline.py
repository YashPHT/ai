from langchain.schema import Document

from fusion import FusionPipeline, KeywordOverlapReranker


def test_fusion_pipeline_weighted_rrf_orders_documents() -> None:
    doc_primary = Document(page_content="Alpha beta guidance", metadata={"source": "primary", "page": 1})
    doc_secondary = Document(page_content="Beta gamma context", metadata={"source": "secondary", "page": 2})

    pipeline = FusionPipeline(token_budget=100, max_results=5, rrf_k=1)
    retriever_outputs = {
        "primary": [doc_primary, doc_secondary],
        "secondary": [doc_secondary],
    }
    weights = {"primary": 2.0, "secondary": 0.2}

    result = pipeline.fuse("alpha beta question", retriever_outputs, weights)

    assert result.documents[0] == doc_primary
    assert result.total_candidates == 3
    assert result.deduplicated_candidates == 2
    assert len(result.selected) == 2
    assert not result.truncated


def test_fusion_pipeline_applies_reranker() -> None:
    off_topic = Document(page_content="Completely unrelated passage", metadata={"source": "off", "page": 1})
    on_topic = Document(page_content="Alpha beta insights and guidance", metadata={"source": "on", "page": 2})

    pipeline = FusionPipeline(
        token_budget=100,
        max_results=5,
        rrf_k=1,
        reranker=KeywordOverlapReranker(),
        reranker_weight=0.8,
    )

    result = pipeline.fuse(
        "alpha beta question",
        {"semantic": [off_topic, on_topic]},
        {"semantic": 1.0},
    )

    assert result.documents[0] == on_topic
    assert result.metadata["reranker_applied"] is True
    assert result.selected[0]["reranker_score"] > result.selected[1]["reranker_score"]


def test_fusion_pipeline_respects_token_budget() -> None:
    doc_short = Document(page_content="alpha beta gamma", metadata={"source": "one", "page": 1})
    doc_medium = Document(page_content="delta epsilon zeta", metadata={"source": "two", "page": 2})
    doc_large = Document(page_content="eta theta iota", metadata={"source": "three", "page": 3})

    pipeline = FusionPipeline(token_budget=5, max_results=5, rrf_k=1)

    result = pipeline.fuse(
        "alpha",
        {"semantic": [doc_short, doc_medium, doc_large]},
        {"semantic": 1.0},
    )

    assert len(result.documents) == 1
    assert len(result.truncated) == 2
    assert result.token_usage <= 5
    assert result.metadata["token_budget"] == 5
    assert result.truncated[0]["rank"] == 2

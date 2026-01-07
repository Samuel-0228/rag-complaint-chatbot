import pytest
from src.rag_pipeline import RAGPipeline


@pytest.fixture
def sample_rag():
    # Mock small index or skip heavy load for CI
    pass  # Or load sample from Task 2


def test_retrieve(sample_rag):
    docs = sample_rag.retrieve("test query", k=1)
    assert len(docs) == 1


def test_rag_query(sample_rag):
    result = sample_rag.rag_query("test query")
    assert 'answer' in result
    assert 'sources' in result


def test_rag_no_retrieval(sample_rag):
    result = sample_rag.rag_query("unrelated query")
    assert result['answer'] != ""
    assert len(result['sources']) == 0


if __name__ == "__main__":
    pytest.main()

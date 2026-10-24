import sys
from pathlib import Path
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nlp.preprocess import preprocess_text
from rag.retriever import SimpleRAGRetriever


_retriever_instance = None


def get_retriever(doc_dir: str = None) -> SimpleRAGRetriever:
    """
    Returns a cached instance of SimpleRAGRetriever.
    Points to rag/documents/ inside the project root by default.
    """
    global _retriever_instance
    if _retriever_instance is None:
        if doc_dir is None:
            doc_dir = str(PROJECT_ROOT / "rag" / "documents")
        _retriever_instance = SimpleRAGRetriever(doc_dir=doc_dir)
    return _retriever_instance


def run_pipeline(idea: str, top_k: int = 3, retriever: SimpleRAGRetriever = None) -> Dict[str, Any]:
    """
    Executes the integrated NLP and RAG pipeline:
    1. Preprocesses the idea using classical NLP (cleaning, tokenization, stop-words, keywords).
    2. Uses the processed text to query the FAISS RAG index.
    3. Retrieves the top_k most relevant knowledge chunks.

    Args:
        idea (str): Raw input idea.
        top_k (int): Number of relevant knowledge chunks to retrieve (default: 3).
        retriever (SimpleRAGRetriever, optional): Pre-initialized or cached retriever instance.

    Returns:
        Dict[str, Any]: Clean Python dictionary containing:
            - 'idea': Original idea string
            - 'nlp': Structured NLP results (cleaned_text, tokens, filtered_tokens, keywords)
            - 'rag': List of retrieved knowledge chunks (text, source, score)
    """
  
    nlp_results = preprocess_text(idea)

    query = nlp_results.get("cleaned_text", "")
    if retriever is None:
        retriever = get_retriever()
    rag_results = retriever.retrieve(query=query, top_k=top_k)

    return {
        "idea": idea,
        "nlp": nlp_results,
        "rag": rag_results,
    }


if __name__ == "__main__":
    sample_idea = (
        "An online platform that helps college students discover hackathons "
        "and form teams based on their skills."
    )

  
    pipeline_result = run_pipeline(sample_idea, top_k=3)

    
    print("=" * 60)
    print("--- NLP OUTPUT ---")
    print("Cleaned Text:")
    print(pipeline_result["nlp"]["cleaned_text"])
    print()
    print("Filtered Tokens:")
    print(pipeline_result["nlp"]["filtered_tokens"])
    print()
    print("Keywords:")
    print(pipeline_result["nlp"]["keywords"])
    print()
    print("--- RAG OUTPUT ---")
    for idx, item in enumerate(pipeline_result["rag"], start=1):
        print(f"Retrieved Knowledge {idx}:")
        print(f"Source: {item['source']}")
        print(f"Content:\n{item['text']}")
        print()
    print("=" * 60)

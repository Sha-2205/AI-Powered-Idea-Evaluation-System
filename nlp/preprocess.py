import re
from typing import Dict, List, Any
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


def ensure_nltk_resources() -> None:
    """
    Ensure the required NLTK datasets are downloaded.
    Required resources:
    - 'punkt' & 'punkt_tab' for tokenization
    - 'stopwords' for stop-word removal
    """
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ]
    for path, name in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


ensure_nltk_resources()


def preprocess_text(text: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Preprocess raw idea text using a classical NLP pipeline:
    1. Text cleaning: removes punctuation and special characters, normalizes whitespace.
    2. Lowercasing: standardizes text case.
    3. Tokenization: splits text into word tokens using NLTK.
    4. Stop-word removal: filters out common English stopwords.
    5. Basic keyword extraction: extracts top frequent terms using NLTK FreqDist.

    Args:
        text (str): The raw input idea string.
        top_n (int): Number of top keywords to extract (default: 5).

    Returns:
        dict: Structured dictionary containing:
            - 'cleaned_text' (str): Cleaned and lowercased text string.
            - 'tokens' (list): Full list of word tokens.
            - 'filtered_tokens' (list): Tokens after removing stopwords.
            - 'keywords' (list): Top extracted keywords based on frequency.
    """
    
    if not text or not isinstance(text, str) or not text.strip():
        return {
            "cleaned_text": "",
            "tokens": [],
            "filtered_tokens": [],
            "keywords": [],
        }


    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    cleaned_text = cleaned.lower()

   
    tokens = word_tokenize(cleaned_text)

    
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    
    freq_dist = FreqDist(filtered_tokens)
    keywords = [word for word, _ in freq_dist.most_common(top_n)]

    return {
        "cleaned_text": cleaned_text,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "keywords": keywords,
    }


if __name__ == "__main__":
    # Test with the sample idea
    sample_idea = (
        "An online platform that helps college students discover hackathons "
        "and form teams based on their skills."
    )

    print("=" * 60)
    print("Testing Classical NLP Preprocessing Module")
    print("=" * 60)
    print(f"Raw text:\n{sample_idea}\n")

    result = preprocess_text(sample_idea)

    print(f"Cleaned text:\n{result['cleaned_text']}\n")
    print(f"Tokens ({len(result['tokens'])}):\n{result['tokens']}\n")
    print(f"Filtered tokens ({len(result['filtered_tokens'])}):\n{result['filtered_tokens']}\n")
    print(f"Keywords:\n{result['keywords']}\n")

    print("-" * 60)
    print("Testing edge case (empty input):")
    print(preprocess_text(""))
    print("=" * 60)

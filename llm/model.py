import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

MODEL_NAME = "gemini-3.5-flash-lite"


_gemini_client = None


def get_gemini_client() -> genai.Client:
    """
    Retrieves or initializes the Gemini API client using the API key from .env.
    """
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Please verify that GEMINI_API_KEY is configured in your .env file."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def generate_evaluation(prompt: str) -> str:
    """
    Sends the structured evaluation prompt to the Gemini API and returns the response text.

    Args:
        prompt (str): The complete evaluation prompt string.

    Returns:
        str: Raw response text returned by Gemini (structured JSON).
    """
    client = get_gemini_client()

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )

    return response.text


if __name__ == "__main__":
    from rag.pipeline import run_pipeline
    from prompts.evaluation_prompt import build_evaluation_prompt

    sample_idea = (
        "An online platform that helps college students discover hackathons "
        "and form teams based on their skills."
    )
    sample_target_users = "College students"

    print("=" * 70)
    print("Step 1: Running NLP + RAG Pipeline ...")
    pipeline_result = run_pipeline(sample_idea, top_k=3)

    print("Step 2: Building Structured Evaluation Prompt ...")
    prompt = build_evaluation_prompt(
        idea=sample_idea,
        target_users=sample_target_users,
        nlp_results=pipeline_result["nlp"],
        retrieved_context=pipeline_result["rag"],
    )

    print(f"Step 3: Sending Prompt to Gemini ({MODEL_NAME}) ...")
    evaluation_response = generate_evaluation(prompt)

    print("=" * 70)
    print("GEMINI EVALUATION RESPONSE:")
    print("=" * 70)
    print(evaluation_response)
    print("=" * 70)

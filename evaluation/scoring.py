import json
from typing import Dict, Any, Union

CRITERIA_KEYS = [
    "customer_need",
    "market_demand",
    "competition",
    "technical_feasibility",
    "business_potential",
]


def calculate_scores(evaluation_data: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Extracts, validates, and aggregates the five evaluation scores from the Gemini response.

    Note: The calculated overall score is purely an aggregated decision-support metric
    and does not guarantee or predict business success.

    Args:
        evaluation_data (dict | str): The Gemini response as a dictionary or raw JSON string.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'individual_scores': Dict of validated scores for each criterion
            - 'overall_score': Average score rounded to 2 decimal places

    Raises:
        ValueError: If JSON is invalid, a criterion is missing, or a score is not between 0 and 100.
        TypeError: If input data is neither a dict nor a str.
    """
    
    if isinstance(evaluation_data, str):
        try:
            data = json.loads(evaluation_data)
        except json.JSONDecodeError as err:
            raise ValueError(f"Failed to parse evaluation JSON: {err}") from err
    elif isinstance(evaluation_data, dict):
        data = evaluation_data
    else:
        raise TypeError(
            f"Expected dict or JSON string, received: {type(evaluation_data).__name__}"
        )

   
    scores_container = data.get("scores")
    if not isinstance(scores_container, dict):
        raise ValueError("Missing or invalid 'scores' block in evaluation data.")

    validated_scores: Dict[str, Union[int, float]] = {}

    
    for key in CRITERIA_KEYS:
        if key not in scores_container:
            raise ValueError(f"Missing required score for '{key}'.")

        val = scores_container[key]

        # Check numeric type (exclude boolean, which is a subclass of int in Python)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            raise ValueError(
                f"Score for '{key}' must be a number, received {type(val).__name__} ({val})."
            )

      
        if not (0 <= val <= 100):
            raise ValueError(
                f"Score for '{key}' is out of bounds ({val}). Must be between 0 and 100."
            )

        validated_scores[key] = val

    score_sum = sum(validated_scores.values())
    overall_score = round(score_sum / len(CRITERIA_KEYS), 2)

    return {
        "individual_scores": validated_scores,
        "overall_score": overall_score,
    }


if __name__ == "__main__":
    # Sample Gemini response matching Phase 7 output
    sample_gemini_output = {
        "scores": {
            "customer_need": 85,
            "market_demand": 75,
            "competition": 60,
            "technical_feasibility": 90,
            "business_potential": 50,
        },
        "reasoning": {
            "customer_need": "Strong pain point among hackathon participants.",
            "market_demand": "Consistent recurring college audience.",
            "competition": "Informal alternatives like Discord exist.",
            "technical_feasibility": "Straightforward web tech stack.",
            "business_potential": "Monetizing college students is difficult.",
        },
        "strengths": ["Clear pain point", "High feasibility"],
        "weaknesses": ["Low student willingness to pay"],
        "risks": ["Retention drop after hackathons"],
        "hidden_assumptions": ["Students prefer a dedicated platform"],
        "recommendations": ["Start with MVP at one university"],
    }

    print("=" * 60)
    print("TESTING EVALUATION & SCORING MODULE")
    print("=" * 60)

    results = calculate_scores(sample_gemini_output)
    scores = results["individual_scores"]

    print(f"Customer Need Score:        {scores['customer_need']}")
    print(f"Market Demand Score:        {scores['market_demand']}")
    print(f"Competition Score:          {scores['competition']}")
    print(f"Technical Feasibility Score:{scores['technical_feasibility']}")
    print(f"Business Potential Score:   {scores['business_potential']}")
    print("-" * 60)
    print(f"Overall Score:              {results['overall_score']}")
    print("=" * 60)

    # Test error handling with invalid score
    print("\nTesting Error Handling (out-of-bounds score):")
    try:
        calculate_scores({"scores": {"customer_need": 150}})
    except ValueError as e:
        print(f"Caught expected error successfully: {e}")
    print("=" * 60)

import sys
from pathlib import Path
from typing import Dict, Any, List, Union


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_evaluation_prompt(
    idea: str,
    target_users: str,
    nlp_results: Dict[str, Any],
    retrieved_context: List[Dict[str, Any]]
) -> str:
    """
    Constructs a comprehensive, structured evaluation prompt designed for an LLM.

    Args:
        idea (str): The submitted user idea.
        target_users (str): Specified target users or demographic.
        nlp_results (Dict[str, Any]): Preprocessing dictionary containing cleaned_text,
                                      filtered_tokens, and keywords.
        retrieved_context (List[Dict[str, Any]]): Retrieved knowledge chunks from FAISS,
                                                  each containing 'source' and 'text'.

    Returns:
        str: Fully formatted prompt string ready to be supplied to an LLM.
    """
   
    cleaned_text = nlp_results.get("cleaned_text", "")
    filtered_tokens = nlp_results.get("filtered_tokens", [])
    keywords = nlp_results.get("keywords", [])


    rag_context_blocks = []
    if retrieved_context:
        for idx, item in enumerate(retrieved_context, start=1):
            source = item.get("source", "Knowledge Base")
            content = item.get("text", "").strip()
            rag_context_blocks.append(f"[Source {idx}: {source}]\n{content}")
        rag_context_str = "\n\n".join(rag_context_blocks)
    else:
        rag_context_str = "No specific retrieved context provided."

    target_users_str = target_users.strip() if target_users and target_users.strip() else "Not specified"

    prompt = f"""### 1. ROLE
You are an expert AI Idea Evaluation Assistant providing structured, objective decision-support analysis for early-stage startup, business, and project ideas.

### 2. TASK
Evaluate the submitted idea using the following five criteria:
1. Customer Need
2. Market Demand
3. Competition
4. Technical Feasibility
5. Business Potential

### 3. USER IDEA
{idea.strip()}

### 4. TARGET USERS
{target_users_str}

### 5. NLP INFORMATION
- Cleaned Text: {cleaned_text}
- Filtered Tokens: {filtered_tokens}
- Extracted Keywords: {keywords}

### 6. RETRIEVED RAG CONTEXT
{rag_context_str}

### 7. EVALUATION INSTRUCTIONS
Carefully analyze the idea using the NLP insights and the retrieved domain knowledge:
- Evaluate each of the five criteria: Customer Need, Market Demand, Competition, Technical Feasibility, and Business Potential.
- Provide an integer score from 0 to 100 for each criterion (0 = extremely weak / non-viable, 100 = exceptionally strong / validated).
- Explain the reasoning for each score in clear, concise detail.
- Identify key Strengths of the idea.
- Identify key Weaknesses of the idea.
- Identify major Risks (technical, market, operational, or adoption risks).
- Identify critical Hidden Assumptions that the idea relies upon.
- Provide practical, actionable Recommendations for next steps.

IMPORTANT DECISION-SUPPORT PRINCIPLES:
- This system is an advisory decision-support tool, not an absolute decision maker.
- Do NOT claim that the idea will definitely succeed or fail.
- Do NOT present the evaluation as a guaranteed prediction.
- Base your analysis directly on the provided idea, target users, NLP information, and retrieved knowledge.
- Clearly identify uncertainty and highlight areas where information is insufficient.

### 8. STRUCTURED JSON OUTPUT
Return your evaluation strictly as a single valid JSON object with NO extra commentary, NO explanation outside the JSON, and NO markdown code block fences.

Use exactly this JSON structure:
{{
  "scores": {{
    "customer_need": 0,
    "market_demand": 0,
    "competition": 0,
    "technical_feasibility": 0,
    "business_potential": 0
  }},
  "reasoning": {{
    "customer_need": "",
    "market_demand": "",
    "competition": "",
    "technical_feasibility": "",
    "business_potential": ""
  }},
  "strengths": [],
  "weaknesses": [],
  "risks": [],
  "hidden_assumptions": [],
  "recommendations": []
}}

Requirements:
- The five score values must be integers from 0 to 100.
- All lists must contain clear, descriptive string items.
- Output valid JSON only."""

    return prompt


if __name__ == "__main__":
    from rag.pipeline import run_pipeline

    sample_idea = (
        "An online platform that helps college students discover hackathons "
        "and form teams based on their skills."
    )
    sample_target_users = "College students"

    print("=" * 70)
    print("Generating Pipeline Results (NLP + RAG) for Test ...")
    pipeline_result = run_pipeline(sample_idea, top_k=3)

    nlp_info = pipeline_result["nlp"]
    rag_context = pipeline_result["rag"]

    print("Building Evaluation Prompt ...")
    generated_prompt = build_evaluation_prompt(
        idea=sample_idea,
        target_users=sample_target_users,
        nlp_results=nlp_info,
        retrieved_context=rag_context
    )

    print("=" * 70)
    print("GENERATED EVALUATION PROMPT:")
    print("=" * 70)
    print(generated_prompt)
    print("=" * 70)

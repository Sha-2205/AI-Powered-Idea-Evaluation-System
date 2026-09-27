import json
from typing import Dict, Any, Union
import streamlit as st

from rag.pipeline import run_pipeline
from prompts.evaluation_prompt import build_evaluation_prompt
from llm.model import generate_evaluation
from evaluation.scoring import calculate_scores


def parse_evaluation_response(response: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parses JSON response from the LLM, handling markdown code fences if present.
    """
    if isinstance(response, dict):
        return response

    content = response.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove opening fence (```json or ```)
        if len(lines) > 1 and lines[0].startswith("```"):
            lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    return json.loads(content)


# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AI Powered Idea Evaluation System",
    page_icon="💡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom CSS for Premium Dark UI (Matching Reference Screenshot)
# ---------------------------------------------------------
CUSTOM_CSS = """
<style>
/* Base typography and background */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: #070E1A !important;
    color: #F1F5F9;
}

/* Completely remove and hide sidebar, its controls, and expand buttons */
[data-testid="stSidebar"],
section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapsedControl"],
button[aria-label="Expand sidebar"],
button[aria-label="Collapse sidebar"] {
    display: none !important;
    visibility: hidden !important;
    width: 0px !important;
    height: 0px !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Ensure main content container uses full available width cleanly */
.block-container,
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewBlockContainer"] {
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1440px !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Hide Streamlit default header decoration */
header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Header bar */
.top-header-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
    padding-top: 4px;
}
.top-header-title {
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.top-header-subtitle {
    font-size: 14px;
    color: #94A3B8;
    margin-top: 4px;
}
.top-header-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 9999px;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #A5B4FC;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.2px;
}

/* Card Containers */
.saas-card {
    background: #0D192C;
    border: 1px solid #182844;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 24px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
}

.card-title-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 17px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 18px;
}
.card-icon-badge {
    width: 28px;
    height: 28px;
    border-radius: 7px;
    background: #1E293B;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #38BDF8;
    font-size: 14px;
}

/* Form inputs styling */
.stTextArea textarea, .stTextInput input {
    background-color: #081222 !important;
    border: 1px solid #1C2B44 !important;
    border-radius: 8px !important;
    color: #F1F5F9 !important;
    font-size: 13.5px !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6366F1 !important;
    box-shadow: 0 0 0 1px #6366F1 !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #CBD5E1 !important;
    margin-bottom: 4px !important;
}

/* Primary Button */
div.stButton > button {
    background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 26px !important;
    border: none !important;
    border-radius: 9999px !important;
    box-shadow: 0 4px 14px rgba(79, 70, 229, 0.4) !important;
    transition: all 0.2s ease-in-out !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6) !important;
}

/* Status Badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 9999px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.35);
    color: #34D399;
    font-size: 12px;
    font-weight: 600;
}

/* Overall Score Box */
.overall-card {
    background: #0D192C;
    border: 1px solid #1B2C49;
    border-radius: 12px;
    padding: 16px 20px;
    display: flex;
    align-items: center;
    gap: 20px;
    height: 100%;
}
.overall-info-title {
    font-size: 15px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
}
.overall-info-desc {
    font-size: 12px;
    color: #94A3B8;
    line-height: 1.35;
}

/* Metric Cards */
.criterion-card {
    background: #0D192C;
    border: 1px solid #1B2C49;
    border-radius: 12px;
    padding: 14px 16px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.criterion-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.criterion-icon {
    width: 24px;
    height: 24px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
}
.criterion-name {
    font-size: 12px;
    font-weight: 600;
    color: #CBD5E1;
}
.criterion-score {
    font-size: 19px;
    font-weight: 800;
    color: #FFFFFF;
    margin-bottom: 8px;
}
.criterion-score-max {
    font-size: 12px;
    font-weight: 500;
    color: #64748B;
}
.progress-track {
    height: 5px;
    background: #13223A;
    border-radius: 9999px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 9999px;
}

/* Insights Cards */
.insight-card {
    background: #0D192C;
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 16px;
}
.insight-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 12px;
}
.insight-list {
    margin: 0;
    padding-left: 18px;
    color: #CBD5E1;
    font-size: 12.5px;
    line-height: 1.55;
}
.insight-list li {
    margin-bottom: 6px;
}

/* Disclaimer panel */
.disclaimer-panel {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid #1E293B;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11.5px;
    color: #64748B;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 24px;
}

/* Streamlit expander overrides */
[data-testid="stExpander"] {
    background-color: #0D192C !important;
    border: 1px solid #182844 !important;
    border-radius: 10px !important;
    margin-bottom: 10px !important;
}
[data-testid="stExpander"] details {
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    font-size: 13.5px !important;
    font-weight: 600 !important;
    color: #E2E8F0 !important;
    padding: 12px 14px !important;
}
[data-testid="stExpander"] summary:hover {
    color: #38BDF8 !important;
}
[data-testid="stExpander"] div[role="region"] {
    padding: 4px 16px 16px 16px !important;
    font-size: 13px !important;
    color: #CBD5E1 !important;
    line-height: 1.5 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)




# ---------------------------------------------------------
# Top Header Bar
# ---------------------------------------------------------
st.markdown(
    """
    <div class="top-header-container">
        <div>
            <div class="top-header-title">
                <span style="color: #6366F1;">✦</span> AI Powered Idea Evaluation System
            </div>
            <div class="top-header-subtitle">
                Evaluate your startup, business, or project idea using NLP, RAG, Prompt Engineering, and Generative AI.
            </div>
        </div>
        <div class="top-header-badge">
            <span>✦</span> AI-Powered Evaluation
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# Idea Input Section (Card Layout)
# ---------------------------------------------------------
st.markdown(
    """
    <div class="card-title-bar" style="margin-bottom: 6px;">
        <div class="card-icon-badge">📝</div>
        <span>Evaluate Your Idea</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Render input form in 2 columns matching the reference layout
col_left, col_right = st.columns([1.15, 1])

with col_left:
    idea = st.text_area(
        "Idea *",
        placeholder="Describe your startup, business, or project idea...",
        height=180,
        help="Required field"
    )

    evaluate_btn = st.button("✦ Evaluate Idea →", type="primary")

with col_right:
    target_users = st.text_input(
        "Target Users *",
        placeholder="e.g., College students, Working professionals...",
        help="Required field"
    )

    problem_description = st.text_area(
        "Problem Description (Optional)",
        placeholder="What problem does your idea solve? ...",
        height=95,
        help="Optional field"
    )


# ---------------------------------------------------------
# Session State for Preserving Evaluation Results
# ---------------------------------------------------------
if "evaluation_data" not in st.session_state:
    st.session_state["evaluation_data"] = None


# ---------------------------------------------------------
# Evaluation Action (Preserves Existing Pipeline Exactly)
# ---------------------------------------------------------
if evaluate_btn:
    # Validate required inputs
    if not idea or not idea.strip():
        st.warning("Please enter your idea before submitting.")
    elif not target_users or not target_users.strip():
        st.warning("Please enter the target users before submitting.")
    else:
        # Construct full idea representation incorporating problem description if provided
        full_idea = idea.strip()
        if problem_description and problem_description.strip():
            full_idea = f"{full_idea}\n\nProblem Description: {problem_description.strip()}"

        try:
            with st.spinner("Processing idea through NLP, RAG retrieval, and Gemini..."):
                # Step 1 & 2: Existing NLP Preprocessing & RAG Retrieval
                pipeline_result = run_pipeline(full_idea, top_k=3)
                nlp_results = pipeline_result["nlp"]
                retrieved_context = pipeline_result["rag"]

                # Step 3: Existing Prompt Construction
                prompt = build_evaluation_prompt(
                    idea=full_idea,
                    target_users=target_users.strip(),
                    nlp_results=nlp_results,
                    retrieved_context=retrieved_context
                )

                # Step 4: Existing Gemini LLM Call
                raw_response = generate_evaluation(prompt)

                # Step 5: Existing JSON Parsing
                evaluation = parse_evaluation_response(raw_response)

                # Step 6: Existing Score Calculation & Validation
                score_data = calculate_scores(evaluation)

                # Store in session state for seamless interactive display
                st.session_state["evaluation_data"] = {
                    "evaluation": evaluation,
                    "score_data": score_data,
                    "nlp_results": nlp_results,
                    "retrieved_context": retrieved_context
                }

        except json.JSONDecodeError:
            st.error(
                "An error occurred while evaluating the idea: Gemini returned a response that could not be converted into valid JSON."
            )
        except Exception as e:
            error_text = str(e)
            if "AIza" in error_text:
                error_text = "API authentication error occurred."
            st.error(f"An error occurred while evaluating the idea: {error_text}")


# ---------------------------------------------------------
# Display Evaluation Results Dashboard
# ---------------------------------------------------------
if st.session_state["evaluation_data"] is not None:
    eval_state = st.session_state["evaluation_data"]
    evaluation = eval_state["evaluation"]
    score_data = eval_state["score_data"]
    individual_scores = score_data["individual_scores"]
    overall_score = score_data["overall_score"]
    nlp_results = eval_state["nlp_results"]
    retrieved_context = eval_state["retrieved_context"]

    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)

    # Section Title & Status Badge
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.markdown(
            """
            <div class="card-title-bar" style="margin-bottom: 14px;">
                <div class="card-icon-badge">📊</div>
                <span>Evaluation Results</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_hdr2:
        st.markdown(
            """
            <div style="text-align: right; margin-bottom: 14px;">
                <span class="status-badge">✓ Evaluation completed successfully</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # 1. Scores Row: Overall Score Ring + 5 Criteria Cards
    # -----------------------------------------------------
    col_ovr, c1, c2, c3, c4, c5 = st.columns([1.6, 1, 1, 1, 1, 1])

    # Dynamic SVG circular progress calculation
    # Radius = 40, Circumference = 2 * pi * 40 ≈ 251.2
    circumference = 251.2
    stroke_offset = max(0.0, circumference * (1.0 - (float(overall_score) / 100.0)))

    with col_ovr:
        st.markdown(
            f"""
            <div class="overall-card">
                <div style="flex-shrink: 0;">
                    <svg width="88" height="88" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#13233C" stroke-width="8"/>
                        <circle cx="50" cy="50" r="40" fill="none" stroke="#10B981" stroke-width="8"
                                stroke-dasharray="{circumference}" stroke-dashoffset="{stroke_offset:.2f}"
                                stroke-linecap="round" transform="rotate(-90 50 50)"/>
                        <text x="50" y="47" text-anchor="middle" font-size="22" font-weight="700" fill="#FFFFFF">{int(overall_score)}</text>
                        <text x="50" y="64" text-anchor="middle" font-size="10" font-weight="600" fill="#64748B">/ 100</text>
                    </svg>
                </div>
                <div>
                    <div class="overall-info-title">Overall Evaluation Score</div>
                    <div class="overall-info-desc">This score represents the overall potential of your idea based on five key criteria.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Criterion 1: Customer Need (Emerald)
    cn_score = individual_scores.get("customer_need", 0)
    with c1:
        st.markdown(
            f"""
            <div class="criterion-card">
                <div>
                    <div class="criterion-header">
                        <div class="criterion-icon" style="background: rgba(16, 185, 129, 0.15); color: #10B981;">👤</div>
                        <div class="criterion-name">Customer Need</div>
                    </div>
                    <div class="criterion-score">{cn_score} <span class="criterion-score-max">/ 100</span></div>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {cn_score}%; background: #10B981;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Criterion 2: Market Demand (Blue)
    md_score = individual_scores.get("market_demand", 0)
    with c2:
        st.markdown(
            f"""
            <div class="criterion-card">
                <div>
                    <div class="criterion-header">
                        <div class="criterion-icon" style="background: rgba(59, 130, 246, 0.15); color: #3B82F6;">📈</div>
                        <div class="criterion-name">Market Demand</div>
                    </div>
                    <div class="criterion-score">{md_score} <span class="criterion-score-max">/ 100</span></div>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {md_score}%; background: #3B82F6;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Criterion 3: Competition (Red)
    cp_score = individual_scores.get("competition", 0)
    with c3:
        st.markdown(
            f"""
            <div class="criterion-card">
                <div>
                    <div class="criterion-header">
                        <div class="criterion-icon" style="background: rgba(239, 68, 68, 0.15); color: #EF4444;">🏆</div>
                        <div class="criterion-name">Competition</div>
                    </div>
                    <div class="criterion-score">{cp_score} <span class="criterion-score-max">/ 100</span></div>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {cp_score}%; background: #EF4444;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Criterion 4: Technical Feasibility (Purple)
    tf_score = individual_scores.get("technical_feasibility", 0)
    with c4:
        st.markdown(
            f"""
            <div class="criterion-card">
                <div>
                    <div class="criterion-header">
                        <div class="criterion-icon" style="background: rgba(139, 92, 246, 0.15); color: #8B5CF6;">⚙️</div>
                        <div class="criterion-name">Technical Feasibility</div>
                    </div>
                    <div class="criterion-score">{tf_score} <span class="criterion-score-max">/ 100</span></div>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {tf_score}%; background: #8B5CF6;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Criterion 5: Business Potential (Amber)
    bp_score = individual_scores.get("business_potential", 0)
    with c5:
        st.markdown(
            f"""
            <div class="criterion-card">
                <div>
                    <div class="criterion-header">
                        <div class="criterion-icon" style="background: rgba(245, 158, 11, 0.15); color: #F59E0B;">↗️</div>
                        <div class="criterion-name">Business Potential</div>
                    </div>
                    <div class="criterion-score">{bp_score} <span class="criterion-score-max">/ 100</span></div>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: {bp_score}%; background: #F59E0B;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 2. Criterion Analysis (2-Column Expandable Cards)
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="card-title-bar" style="margin-bottom: 12px;">
            <div class="card-icon-badge">🔍</div>
            <span>Criterion Analysis</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    reasoning = evaluation.get("reasoning", {})
    col_an1, col_an2 = st.columns(2)

    with col_an1:
        with st.expander(f"👤 Customer Need &nbsp;&nbsp;·&nbsp;&nbsp; Score: {cn_score}/100", expanded=True):
            st.write(reasoning.get("customer_need", "No reasoning provided."))

        with st.expander(f"📈 Market Demand &nbsp;&nbsp;·&nbsp;&nbsp; Score: {md_score}/100", expanded=True):
            st.write(reasoning.get("market_demand", "No reasoning provided."))

        with st.expander(f"🏆 Competition &nbsp;&nbsp;·&nbsp;&nbsp; Score: {cp_score}/100", expanded=True):
            st.write(reasoning.get("competition", "No reasoning provided."))

    with col_an2:
        with st.expander(f"⚙️ Technical Feasibility &nbsp;&nbsp;·&nbsp;&nbsp; Score: {tf_score}/100", expanded=True):
            st.write(reasoning.get("technical_feasibility", "No reasoning provided."))

        with st.expander(f"↗️ Business Potential &nbsp;&nbsp;·&nbsp;&nbsp; Score: {bp_score}/100", expanded=True):
            st.write(reasoning.get("business_potential", "No reasoning provided."))

    st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # 3. Additional Insights (2-Column Semantic Cards)
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="card-title-bar" style="margin-bottom: 12px;">
            <div class="card-icon-badge">💡</div>
            <span>Additional Insights</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_ins1, col_ins2 = st.columns(2)

    with col_ins1:
        # Strengths (Emerald)
        strengths_html = "".join([f"<li>{item}</li>" for item in evaluation.get("strengths", [])]) or "<li>No strengths listed.</li>"
        st.markdown(
            f"""
            <div class="insight-card" style="border: 1px solid rgba(16, 185, 129, 0.35);">
                <div class="insight-card-header" style="color: #10B981;">
                    <span>✓</span> <span>Strengths</span>
                </div>
                <ul class="insight-list">
                    {strengths_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risks (Red)
        risks_html = "".join([f"<li>{item}</li>" for item in evaluation.get("risks", [])]) or "<li>No risks listed.</li>"
        st.markdown(
            f"""
            <div class="insight-card" style="border: 1px solid rgba(239, 68, 68, 0.35);">
                <div class="insight-card-header" style="color: #EF4444;">
                    <span>⚠️</span> <span>Risks</span>
                </div>
                <ul class="insight-list">
                    {risks_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_ins2:
        # Weaknesses (Amber)
        weaknesses_html = "".join([f"<li>{item}</li>" for item in evaluation.get("weaknesses", [])]) or "<li>No weaknesses listed.</li>"
        st.markdown(
            f"""
            <div class="insight-card" style="border: 1px solid rgba(245, 158, 11, 0.35);">
                <div class="insight-card-header" style="color: #F59E0B;">
                    <span>⚠️</span> <span>Weaknesses</span>
                </div>
                <ul class="insight-list">
                    {weaknesses_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Hidden Assumptions (Purple)
        assumptions_html = "".join([f"<li>{item}</li>" for item in evaluation.get("hidden_assumptions", [])]) or "<li>No hidden assumptions listed.</li>"
        st.markdown(
            f"""
            <div class="insight-card" style="border: 1px solid rgba(139, 92, 246, 0.35);">
                <div class="insight-card-header" style="color: #8B5CF6;">
                    <span>◈</span> <span>Hidden Assumptions</span>
                </div>
                <ul class="insight-list">
                    {assumptions_html}
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # 4. Recommendations (Full Width Card, Cyan Accent)
    # -----------------------------------------------------
    recommendations_html = "".join([f"<li>{item}</li>" for item in evaluation.get("recommendations", [])]) or "<li>No recommendations listed.</li>"
    st.markdown(
        f"""
        <div class="insight-card" style="border: 1px solid rgba(6, 182, 212, 0.35); margin-top: 4px;">
            <div class="insight-card-header" style="color: #06B6D4;">
                <span>🎯</span> <span>Recommendations</span>
            </div>
            <ul class="insight-list">
                {recommendations_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # 5. Disclaimer Note Bar
    # -----------------------------------------------------
    st.markdown(
        """
        <div class="disclaimer-panel">
            <span style="font-size: 14px; color: #38BDF8;">ℹ️</span>
            <span><b>Note:</b> This evaluation provides AI-generated decision support based on the submitted idea and retrieved knowledge. It does not guarantee business success or replace detailed market research.</span>
        </div>
        """,
        unsafe_allow_html=True
    )

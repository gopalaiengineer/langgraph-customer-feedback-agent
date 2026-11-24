"""Streamlit UI for the LangGraph customer feedback analysis workflow."""

import html
import json
import re
import uuid

import streamlit as st

from src.app.pipelines.pipeline import (
    customer_feedback_Analysis,
    resume_customer_feedback_Analysis,
)
from src.app.configs.config import settings
from src.app.guardrail import (
    ALLOWED_KEYWORDS,
    REVIEW_ALLOWED_KEYWORDS,
    custom_guardrail,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Feedback AI | Command Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111a30 55%, #101827 100%);
        color: #edf2ff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        background: #0c1325;
        border-right: 1px solid #25314b;
    }

    [data-testid="stSidebar"] * {
        color: #e5ecff;
    }

    [data-testid="stSidebar"] hr {
        border-color: #2c3b59;
    }

    .hero {
        padding: 30px;
        border: 1px solid #344568;
        border-radius: 22px;
        background: linear-gradient(120deg, #172849, #25204b, #123b4a);
        margin-bottom: 24px;
    }

    .hero-tag {
        color: #72e4d1;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 36px;
        font-weight: 800;
        line-height: 1.25;
        margin: 10px 0;
    }

    .hero p {
        color: #c5d2ed !important;
        font-size: 15px;
        line-height: 1.8;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #f0f5ff;
        margin: 12px 0 16px;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #f0f5ff !important;
    }

    p, li, label {
        color: #e5ecff;
    }

    [data-testid="stCaptionContainer"] p {
        color: #aebbd4 !important;
    }

    .panel {
        background: rgba(23, 34, 57, 0.92);
        border: 1px solid #2c3b59;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        overflow-wrap: anywhere;
        min-width: 0;
    }

    .panel-label {
        color: #9cafd0 !important;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    .panel-value {
        color: #f4f7ff !important;
        font-size: 18px;
        font-weight: 700;
        line-height: 1.6;
        margin-top: 8px;
        overflow-wrap: anywhere;
        word-break: normal;
        white-space: normal;
    }

    .muted {
        color: #aebbd4 !important;
        font-size: 13px;
        line-height: 1.7;
        overflow-wrap: anywhere;
        white-space: normal;
    }

    .stTextInput input,
    .stTextArea textarea {
        background-color: #111b30 !important;
        color: #f1f5ff !important;
        -webkit-text-fill-color: #f1f5ff !important;
        caret-color: #ffffff !important;
        border: 1px solid #435575 !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
    }

    .stTextInput input:-webkit-autofill,
    .stTextInput input:-webkit-autofill:hover,
    .stTextInput input:-webkit-autofill:focus,
    .stTextInput input:-webkit-autofill:active {
        -webkit-box-shadow: 0 0 0 1000px #111b30 inset !important;
        -webkit-text-fill-color: #f1f5ff !important;
        caret-color: #ffffff !important;
    }

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #8494b2 !important;
        -webkit-text-fill-color: #8494b2 !important;
        opacity: 1 !important;
    }

    .stTextInput input:focus,
    .stTextArea textarea:focus {
        border-color: #818cff !important;
        box-shadow: 0 0 0 2px rgba(91, 108, 255, 0.20) !important;
        outline: none !important;
    }

    .stTextInput label,
    .stTextArea label {
        color: #e5ecff !important;
        font-weight: 600 !important;
    }

    .stTextArea textarea,
    .stTextArea textarea:disabled {
        line-height: 1.8 !important;
        font-size: 14px !important;
        opacity: 1 !important;
    }

    .stTextArea textarea:disabled {
        cursor: text !important;
    }

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(90deg, #5b6cff, #7c5cff) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: 1px solid #7c8cff !important;
        border-radius: 11px !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        min-height: 46px;
        padding: 10px 18px;
        opacity: 1 !important;
        box-shadow: 0 4px 14px rgba(91, 108, 255, 0.20);
        transition: background 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    [data-testid="stFormSubmitButton"] button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFormSubmitButton"] button:hover {
        background: linear-gradient(90deg, #7180ff, #9675ff) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border-color: #c0c7ff !important;
        box-shadow: 0 6px 20px rgba(91, 108, 255, 0.35);
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible,
    [data-testid="stFormSubmitButton"] button:focus-visible {
        outline: 2px solid #aeb8ff !important;
        outline-offset: 3px;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled {
        background: #28344b !important;
        color: #aebbd4 !important;
        -webkit-text-fill-color: #aebbd4 !important;
        border-color: #3a4963 !important;
        box-shadow: none !important;
        opacity: 0.8 !important;
    }

    .stCheckbox {
        padding: 5px 0;
    }

    .stCheckbox label,
    .stCheckbox label p,
    .stCheckbox label span {
        color: #e5ecff !important;
        -webkit-text-fill-color: #e5ecff !important;
        font-size: 13px !important;
    }

    .stCheckbox input {
        accent-color: #7c5cff;
    }

    [data-testid="stMetric"] {
        background: #172239;
        border: 1px solid #2c3b59;
        border-radius: 15px;
        padding: 18px;
        min-width: 0;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p {
        color: #aebbd4 !important;
        font-size: 13px !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 800 !important;
        overflow-wrap: anywhere;
        white-space: normal !important;
    }

    [data-testid="stMetricDelta"] {
        color: #72e4d1 !important;
    }

    [data-testid="stExpander"] {
        background: #121c30;
        border: 1px solid #2c3b59;
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        color: #edf2ff !important;
        background: #172239;
        white-space: normal;
    }

    [data-testid="stExpander"] summary:hover {
        background: #202e49;
    }

    [data-testid="stExpander"] summary p {
        color: #edf2ff !important;
        font-weight: 600;
        white-space: normal;
    }

    [data-testid="stJson"],
    [data-testid="stCode"],
    .stJson {
        background: #111b30 !important;
        color: #e5ecff !important;
        border-radius: 10px;
    }

    pre, code {
        color: #dbe7ff !important;
    }

    .status {
        padding: 12px 15px;
        border-radius: 10px;
        background: #172b3c;
        border: 1px solid #31556a;
        color: #b6f2e6 !important;
        font-weight: 600;
    }

    [data-testid="stAlertContainer"] {
        background: #172239 !important;
        border: 1px solid #344568 !important;
        border-radius: 12px !important;
        color: #e5ecff !important;
    }

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {
        background: #162b43 !important;
        border-color: #315779 !important;
        color: #d5eaff !important;
    }

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {
        background: #17382f !important;
        border-color: #34745d !important;
        color: #c8f3df !important;
    }

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {
        background: #3a3020 !important;
        border-color: #80632e !important;
        color: #ffe4a8 !important;
    }

    [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
        background: #3b2229 !important;
        border-color: #87404d !important;
        color: #ffd1d6 !important;
    }

    [data-testid="stAlertContainer"] *,
    [data-testid^="stAlertContent"] {
        background: transparent !important;
        color: inherit !important;
        -webkit-text-fill-color: currentColor !important;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background-color: #111b30 !important;
        border-color: #435575 !important;
    }

    .stSelectbox [data-baseweb="select"] [role="combobox"],
    .stSelectbox [data-baseweb="select"] [role="combobox"] *,
    .stSelectbox [data-baseweb="select"] span {
        color: #f1f5ff !important;
        -webkit-text-fill-color: #f1f5ff !important;
    }

    [data-baseweb="popover"] [role="listbox"] {
        background: #172239 !important;
        border: 1px solid #435575 !important;
    }

    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] * {
        background: #172239 !important;
        color: #e5ecff !important;
        -webkit-text-fill-color: #e5ecff !important;
    }

    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"]:hover *,
    [data-baseweb="popover"] [role="option"][aria-selected="true"],
    [data-baseweb="popover"] [role="option"][aria-selected="true"] * {
        background: #344568 !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    .field-error {
        color: #ff6b6b !important;
        background: transparent !important;
        border: 0 !important;
        padding: 0 !important;
        margin: -1.25rem 0 -0.25rem !important;
        font-size: 13px;
        line-height: 1.5;
    }

    .stDownloadButton {
        margin-top: 10px;
    }

    .stDownloadButton > button {
        width: 100%;
    }

    hr {
        border-color: #2c3b59 !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem;
        }

        .hero {
            padding: 22px;
        }

        .hero h1 {
            font-size: 28px;
        }

        .panel {
            padding: 16px;
        }

        .panel-value {
            font-size: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def display_value(value):
    """Convert dictionaries and other values into readable text."""
    if value is None:
        return "Not available"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    return str(value)


def display_analysis(analysis):
    """Normalize analysis output from a Pydantic model, dict, or JSON string."""
    if hasattr(analysis, "model_dump"):
        return analysis.model_dump()

    if isinstance(analysis, dict):
        return analysis

    if isinstance(analysis, str):
        try:
            parsed = json.loads(analysis)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"details": analysis}

    if analysis is None:
        return {}

    return {"details": display_value(analysis)}


def metric_card(label, value, help_text=None):
    """Render a Streamlit metric card."""
    st.metric(label=label, value=value, help=help_text)


def safe_html(value):
    """Escape dynamic values before inserting them into HTML."""
    return html.escape(display_value(value))


def show_field_error(placeholder, message):
    placeholder.markdown(
        f'<div class="field-error">{safe_html(message)}</div>',
        unsafe_allow_html=True,
    )


def valid_customer_name(name):
    """
    Validate a name containing Unicode letters and common separators.
    Allows names such as Alex Johnson, O'Connor, Anne-Marie, and José.
    """
    name = name.strip()

    if not 2 <= len(name) <= 100:
        return False

    if not any(character.isalpha() for character in name):
        return False

    if any(character.isdigit() for character in name):
        return False

    # Names may contain letters separated by spaces, apostrophes,
    # periods, or hyphens. Underscores and other symbols are rejected.
    return bool(
        re.fullmatch(
            r"[^\W\d_]+(?:[ .'-][^\W\d_]+)*\.?",
            name,
            flags=re.UNICODE,
        )
    )


def valid_email(email):
    """Perform practical format validation for an email address."""
    email = email.strip()

    if not email or len(email) > 254:
        return False

    if any(character.isspace() for character in email):
        return False

    return bool(
        re.fullmatch(
            r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+",
            email,
        )
    )


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🤖 Feedback AI")
    st.caption("CUSTOMER EXPERIENCE PLATFORM")
    st.divider()

    st.markdown("### Workflow")
    st.markdown(
        """
        **01** · Sentiment classification

        **02** · Conditional routing

        **03** · Complaint analysis

        **04** · Email generation

        **05** · Human email review

        **06** · Email delivery after approval
        """
    )

    st.divider()
    st.markdown("### Built with")
    st.markdown(
        """
        - Python
        - Streamlit
        - LangGraph
        - LangChain
        - Groq LLM
        - Pydantic
        """
    )
    st.divider()
    st.caption("AI-powered customer feedback automation")


# --------------------------------------------------
# HERO SECTION
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">AI Automation · Live Demo</div>
        <h1>Customer Feedback<br>Command Center</h1>
        <p>
            Turn customer feedback into actionable insights.
            Classify sentiment, analyze complaints, and generate
            customer-support emails with your LangGraph workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.markdown(
        '<div class="section-title">Customer Feedback</div>',
        unsafe_allow_html=True,
    )

    with st.form("feedback_form"):
        customer_name = st.text_input(
            "Customer name *",
            placeholder="Enter customer's full name",
            max_chars=100,
            autocomplete="off",
        )
        customer_name_error = st.empty()

        customer_email = st.text_input(
            "Customer email *",
            placeholder="name@example.com",
            max_chars=254,
            autocomplete="off",
        )
        customer_email_error = st.empty()

        feedback = st.text_area(
            "Feedback message *",
            height=180,
            placeholder="Enter the customer's feedback...",
            max_chars=10000,
        )
        feedback_error = st.empty()

        st.caption(
            "All fields are required. The customer name, email, and feedback "
            "will be passed to the analysis workflow."
        )

        acknowledge_email = st.checkbox(
            "I understand the email will only be sent after I approve its draft. *"
        )
        acknowledge_email_error = st.empty()

        submitted = st.form_submit_button(
            "Analyze Feedback & Run Workflow",
            use_container_width=True,
        )

with right:
    st.markdown(
        '<div class="section-title">What the AI does</div>',
        unsafe_allow_html=True,
    )

    workflow_steps = [
        ("01", "Sentiment Detection", "Identify positive or negative feedback."),
        ("02", "Smart Routing", "Choose the correct LangGraph branch."),
        ("03", "Priority Routing", "Classify urgency and escalate P1/P2 issues."),
        ("04", "Email Generation", "Draft a context-aware support email."),
        ("05", "Human review", "Approve the draft or request changes."),
        ("06", "Delivery", "Send only after explicit human approval."),
    ]

    for number, title, description in workflow_steps:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">STEP {number}</div>
                <div class="panel-value">{safe_html(title)}</div>
                <div class="muted">{safe_html(description)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# VALIDATION AND PIPELINE EXECUTION
# --------------------------------------------------

if submitted:
    cleaned_name = customer_name.strip()
    cleaned_email = customer_email.strip()
    cleaned_feedback = feedback.strip()
    name_error = None
    email_error = None
    feedback_message_error = None
    acknowledgement_error = None

    if not cleaned_name:
        name_error = "Customer name is required."
    elif not valid_customer_name(cleaned_name):
        name_error = (
            "Enter a valid customer name using letters and common name separators "
            "(spaces, apostrophes, periods, or hyphens). Numbers and other symbols "
            "are not allowed."
        )

    if not cleaned_email:
        email_error = "Customer email is required."
    elif not valid_email(cleaned_email):
        email_error = (
            "Enter a valid email address, such as name@example.com."
        )

    if not cleaned_feedback:
        feedback_message_error = "Feedback message is required."
    elif len(cleaned_feedback) < 5:
        feedback_message_error = (
            "Feedback message must contain at least 5 characters."
        )
    else:
        cd=1
    #     feedback_validation = custom_guardrail(cleaned_feedback)
    #     if not feedback_validation.allowed:
    #         feedback_message_error = (
    #             f"{feedback_validation.reason} "
    #             f"({feedback_validation.category.replace('_', ' ').title()})"
    #         )

    # if not acknowledge_email:
    #     acknowledgement_error = (
    #         "Please confirm that you understand the workflow may send a real email."
    #     )

    if name_error:
        show_field_error(customer_name_error, name_error)
    if email_error:
        show_field_error(customer_email_error, email_error)
    if feedback_message_error:
        show_field_error(feedback_error, feedback_message_error)
    if acknowledgement_error:
        show_field_error(acknowledge_email_error, acknowledgement_error)

    if not any(
        (name_error, email_error, feedback_message_error, acknowledgement_error)
    ):
        try:
            thread_id = uuid.uuid4().hex
            with st.spinner(
                "AI agents are processing feedback. Please wait..."
            ):
                result = customer_feedback_Analysis(
                    cleaned_feedback,
                    cleaned_name,
                    cleaned_email,
                    thread_id,
                )

            if not isinstance(result, dict):
                st.error(
                    "The pipeline did not return a dictionary. "
                    "Check customer_feedback_Analysis() output."
                )
            else:
                st.session_state["feedback_result"] = result
                st.session_state["email_review_thread_id"] = thread_id
                st.session_state["customer_name"] = cleaned_name
                st.session_state["customer_email"] = cleaned_email
                st.session_state["customer_feedback"] = cleaned_feedback
                st.success("Draft prepared. Review it below before sending.")

        except Exception as exc:
            st.error("Workflow execution failed. Check the pipeline configuration.")
            st.exception(exc)


# --------------------------------------------------
# RESULTS DASHBOARD
# --------------------------------------------------

result = st.session_state.get("feedback_result")

if result:
    st.divider()

    st.markdown(
        '<div class="section-title">AI Analysis Dashboard</div>',
        unsafe_allow_html=True,
    )

    customer_name_result = st.session_state.get("customer_name", "Customer")
    customer_email_result = st.session_state.get(
        "customer_email", "Not available"
    )

    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">CUSTOMER</div>
            <div class="panel-value">{safe_html(customer_name_result)}</div>
            <div class="muted">{safe_html(customer_email_result)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sentiment = str(result.get("sentiment", "Unknown")).upper()
    sentiment_reason = result.get("sentiment_reason", "Not available")
    analysis = display_analysis(result.get("analysis", {}))

    priority = str(analysis.get("priority", "N/A")).upper()
    urgency = str(analysis.get("urgency", "N/A")).upper()
    severity = str(analysis.get("severity", "N/A")).upper()
    issue_type = (
        str(
            analysis.get(
                "issue_category",
                analysis.get("issue_type", "N/A"),
            )
        )
        .replace("_", " ")
        .title()
    )
    tone = str(analysis.get("tone", "N/A")).title()

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        metric_card("Sentiment", sentiment)

    with col2:
        metric_card("Priority", priority)

    with col3:
        metric_card("Urgency", urgency)

    with col4:
        metric_card("Issue Category", issue_type)

    with col5:
        metric_card("Severity", severity)

    with col6:
        metric_card("Customer Tone", tone)

    # ----------------------------------------------
    # SENTIMENT DETAILS
    # ----------------------------------------------

    st.markdown("### Sentiment Intelligence")

    sentiment_col, reason_col = st.columns([0.7, 1.3])

    with sentiment_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">DETECTED SENTIMENT</div>
                <div class="panel-value">{safe_html(sentiment)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with reason_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">AI REASONING</div>
                <div class="muted">{safe_html(sentiment_reason)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ----------------------------------------------
    # ISSUE ANALYSIS
    # ----------------------------------------------

    st.markdown("### Complaint Analysis")

    summary = analysis.get("summary", "Not available")
    suggested_action = analysis.get(
        "recommended_action",
        analysis.get("suggested_action", "Not available"),
    )
    escalation_level = analysis.get("escalation_level", "Not available")
    escalation_reason = analysis.get("escalation_reason", "Not available")

    detail_col, action_col = st.columns(2)

    with detail_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">ISSUE SUMMARY</div>
                <div class="panel-value">{safe_html(summary)}</div>
                <br>
                <div class="panel-label">EMOTIONAL TONE</div>
                <div class="muted">{safe_html(tone)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with action_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">RECOMMENDED NEXT ACTION</div>
                <div class="panel-value">{safe_html(suggested_action)}</div>
                <br>
                <div class="panel-label">ISSUE CATEGORY</div>
                <div class="muted">{safe_html(issue_type)}</div>
                <br>
                <div class="panel-label">ESCALATION LEVEL</div>
                <div class="muted">{safe_html(escalation_level)}</div>
                <br>
                <div class="panel-label">ESCALATION REASON</div>
                <div class="muted">{safe_html(escalation_reason)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    support_ticket = result.get("support_ticket")
    if support_ticket:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">SUPPORT TICKET</div>
                <div class="panel-value">
                    {safe_html(support_ticket.get("ticket_id"))}
                </div>
                <div class="muted">
                    {safe_html(support_ticket.get("priority"))} ·
                    {safe_html(support_ticket.get("status"))} ·
                    Internal notification:
                    {safe_html(result.get("internal_notification_status"))}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("View complete structured analysis"):
        st.json(analysis)

    # ----------------------------------------------
    # GENERATED EMAIL
    # ----------------------------------------------

    st.markdown("### Generated Customer Email")

    email_subject = result.get("email_subject", "")
    email_body = result.get("email_body", "")

    email_col, status_col = st.columns([1.5, 0.7])

    with email_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">EMAIL SUBJECT</div>
                <div class="panel-value">{safe_html(email_subject)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.text_area(
            "Email body preview",
            value=display_value(email_body),
            height=300,
            disabled=True,
            key="email_preview",
            help="The email body generated by your AI workflow.",
        )

        if email_body:
            st.download_button(
                "Download email body",
                data=display_value(email_body),
                file_name="customer_support_email.txt",
                mime="text/plain",
            )

        email_status = str(result.get("email_status", "Unknown"))
        draft_history = result.get("email_draft_history", [])
        if draft_history:
            with st.expander("Human scores for reviewed drafts"):
                st.table(
                    [
                        {
                            "Draft": index,
                            "Subject": draft["subject"],
                            "Score": f"{draft['score']}/10",
                        }
                        for index, draft in enumerate(draft_history, start=1)
                    ]
                )

        review_attempts = result.get("email_review_attempts", 0)
        if email_status == "awaiting_final_selection":
            st.warning(
                f"The limit of {settings.MAX_EMAIL_REVISIONS} evaluation "
                "requests has been reached. Choose a draft to send; its score "
                "is for comparison only and does not affect whether it can be "
                "sent. Stop workflow ends without sending."
            )
            selected_draft_index = st.radio(
                "Choose the email draft to send",
                options=list(range(len(draft_history))),
                format_func=lambda index: (
                    f"Draft {index + 1} — "
                    f"{draft_history[index]['score']}/10 — "
                    f"{draft_history[index]['subject']}"
                ),
                key=f"final_email_draft_{review_attempts}",
            )
            selected_draft = draft_history[selected_draft_index]
            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-label">SELECTED EMAIL SUBJECT</div>
                    <div class="panel-value">
                        {safe_html(selected_draft["subject"])}
                    </div>
                    <div class="panel-label">SCORE</div>
                    <div class="muted">{selected_draft["score"]}/10</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.text_area(
                "Selected email details",
                value=display_value(selected_draft["body"]),
                height=300,
                disabled=True,
                key=f"final_email_body_{review_attempts}_{selected_draft_index}",
            )
            send_col, stop_col = st.columns(2)
            with send_col:
                send_selected = st.button(
                    "Send selected email",
                    key=f"send_selected_email_{review_attempts}",
                    use_container_width=True,
                )
            with stop_col:
                stop_workflow = st.button(
                    "Stop workflow",
                    key=f"stop_email_workflow_{review_attempts}",
                    use_container_width=True,
                )

            if send_selected or stop_workflow:
                final_decision = "send" if send_selected else "stop"
                try:
                    with st.spinner(
                        "Sending selected email..."
                        if send_selected
                        else "Stopping workflow..."
                    ):
                        updated_result = resume_customer_feedback_Analysis(
                            st.session_state["email_review_thread_id"],
                            final_decision=final_decision,
                            selected_draft_index=(
                                selected_draft_index
                                if send_selected
                                else None
                            ),
                        )
                    st.session_state["feedback_result"] = updated_result
                    st.rerun()
                except Exception as exc:
                    st.error(
                        "The final email decision could not be completed. "
                        "Check the workflow configuration."
                    )
                    st.exception(exc)
        elif email_status == "awaiting_approval":
            st.warning(
                "Select Accepted to send the draft, Needs Evaluation with "
                "clarification to request a revision, or Stop to end the "
                "workflow. Score the draft when accepting it or requesting "
                "evaluation."
            )
            with st.form(f"email_review_form_{review_attempts}"):
                approval_status = st.selectbox(
                    "Approver status",
                    options=[None, "Accepted", "Needs Evaluation", "Stop"],
                    format_func=lambda value: (
                        "Select a status" if value is None else value
                    ),
                    key=f"email_approval_status_{review_attempts}",
                )
                approval_status_error = st.empty()
                score = st.selectbox(
                    "Score this draft (1 = lowest, 10 = highest)",
                    options=[None, *range(1, 11)],
                    format_func=lambda value: (
                        "Select a score" if value is None else f"{value}/10"
                    ),
                    key=f"email_score_{review_attempts}",
                )
                score_error = st.empty()
                clarification = st.text_area(
                    "Clarification (required for Needs Evaluation or Stop)",
                    placeholder=(
                        "For example: Make the tone warmer and clarify that "
                        "we will investigate the billing issue."
                    ),
                    key=f"email_clarification_{review_attempts}",
                )
                clarification_error = st.empty()
                submit_review = st.form_submit_button(
                    "Submit review",
                    use_container_width=True,
                )

            if submit_review:
                approval_error_message = None
                score_error_message = None
                clarification_error_message = None
                if approval_status is None:
                    approval_error_message = (
                        "Please select Accepted, Needs Evaluation, or Stop."
                    )
                if approval_status != "Stop" and score is None:
                    score_error_message = (
                        "Please score this draft from 1 to 10."
                    )
                if (
                    approval_status in {"Needs Evaluation", "Stop"}
                    and not clarification.strip()
                ):
                    clarification_error_message = (
                        "Please provide a clarification for this status."
                    )
                if (
                    approval_status == "Needs Evaluation"
                    and not clarification_error_message
                ):
                    c =1;
                    # clarification_validation = custom_guardrail(
                    #     clarification,
                    #     allowed_keywords=[
                    #         *ALLOWED_KEYWORDS,
                    #         *REVIEW_ALLOWED_KEYWORDS,
                    #     ],
                    # )
                    # if not clarification_validation.allowed:
                    #     clarification_error_message = (
                    #         f"{clarification_validation.reason} "
                    #         f"({clarification_validation.category.replace('_', ' ').title()})"
                    #     )

                if approval_error_message:
                    show_field_error(
                        approval_status_error,
                        approval_error_message,
                    )
                if score_error_message:
                    show_field_error(score_error, score_error_message)
                if clarification_error_message:
                    show_field_error(
                        clarification_error,
                        clarification_error_message,
                    )

                if not any(
                    (
                        approval_error_message,
                        score_error_message,
                        clarification_error_message,
                    )
                ):
                    try:
                        with st.spinner(
                            "Sending accepted email..."
                            if approval_status == "Accepted"
                            else "Preparing a revised email..."
                            if approval_status == "Needs Evaluation"
                            else "Stopping workflow..."
                        ):
                            updated_result = (
                                resume_customer_feedback_Analysis(
                                    st.session_state[
                                        "email_review_thread_id"
                                    ],
                                    approval_status=approval_status,
                                    score=score,
                                    clarification=clarification,
                                )
                            )
                        st.session_state["feedback_result"] = updated_result
                        st.rerun()
                    except Exception as exc:
                        st.error(
                            "Email review could not be completed. "
                            "Check the workflow configuration."
                        )
                        st.exception(exc)

    with status_col:
        email_status = str(result.get("email_status", "Unknown"))
        if (
            email_status == "sent"
            and result.get("email_selected_score") is not None
        ):
            status_description = (
                "Sent the draft selected by the approver "
                f"({result.get('email_selected_score')}/10)."
            )
        elif email_status == "sent":
            status_description = "Email sent after your approval."
        elif email_status == "stopped":
            status_description = (
                "The approver stopped the workflow; no email was sent."
            )
        elif email_status == "awaiting_final_selection":
            status_description = (
                "Waiting for the approver to select a reviewed email or stop."
            )
        else:
            status_description = (
                "Waiting for your approval; no email has been sent."
            )

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">DELIVERY STATUS</div>
                <div class="panel-value">{safe_html(email_status.upper())}</div>
                <p class="muted">
                    {safe_html(status_description)}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("**Workflow result**")

        st.write(
            "Positive branch"
            if sentiment == "POSITIVE"
            else "Negative-feedback branch"
        )

        st.caption(
            "The displayed status reflects the pipeline result; "
            "it does not independently verify inbox delivery."
        )

    # ----------------------------------------------
    # COMPLETE OUTPUT
    # ----------------------------------------------

    st.markdown("### Complete Pipeline Output")

    with st.expander("Inspect raw result returned by the pipeline"):
        st.json(result)

    st.download_button(
        "Download complete analysis as JSON",
        data=json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        file_name="customer_feedback_analysis.json",
        mime="application/json",
    )

else:
    st.info(
        "Your analysis will appear here after you run the workflow."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Customer Feedback AI Command Center · "
    "Powered by Streamlit, LangGraph, LangChain and Groq"
)
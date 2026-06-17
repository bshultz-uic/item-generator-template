"""
NBME Item Generator
===================

A web-based assessment tool that helps health science item writers construct
high-quality multiple-choice questions following NBME guidelines.

The app walks the writer through three steps:

    Step 1 - The Target Planner   : choose the competency / subcomponent and
                                    understand what each one assesses.
    Step 2 - The Vignette Builder : fill out the standard NBME patient vignette
                                    template and pick a specific lead-in.
    Step 3 - API Generation       : send the structured data to Gemini and get
                                    back a polished clinical vignette + lead-in.

----------------------------------------------------------------------------
SETUP
----------------------------------------------------------------------------
1. Install dependencies:

       pip install streamlit google-genai

2. Provide your Gemini API key securely. Create a file at
   ``.streamlit/secrets.toml`` (relative to this app) with the contents:

       # .streamlit/secrets.toml
       GEMINI_API_KEY = "your-api-key-here"

   Never hard-code the key in this file or commit secrets.toml to source
   control. (Add ``.streamlit/secrets.toml`` to your .gitignore.)

3. Run the app:

       streamlit run app.py
"""

import streamlit as st

# The Google Gen AI SDK (NOT the legacy google-generativeai package).
from google import genai


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Default Free Tier model. "gemini-flash-latest" is a moving alias that always
# resolves to the current Gemini Flash model, so it keeps working as Google
# rotates version numbers. You can override it without editing code by adding a
# line to .streamlit/secrets.toml, e.g.:
#
#     GEMINI_MODEL = "gemini-2.5-flash"
#
# If you hit a "model not found" error, the app will list the exact model IDs
# your API key is allowed to use so you can pick a valid one.
DEFAULT_MODEL = "gemini-flash-latest"


# ---------------------------------------------------------------------------
# The Taxonomy
# ---------------------------------------------------------------------------
# A strict hierarchical structure describing each Main Task Competency, its
# Subcomponents (if any), and the specific lead-ins that belong to each.
#
#   - "Foundational (Basic) Science" has NO subcomponents; its lead-ins live
#     directly under a "lead_ins" key.
#   - "Diagnosis" and "Management" each have subcomponents, and every
#     subcomponent owns its own list of lead-ins.
TAXONOMY = {
    "Foundational (Basic) Science": {
        "description": (
            "Assesses the basic-science principles and mechanisms that "
            "underlie normal function and disease (e.g., anatomy, physiology, "
            "pathology, pharmacology, microbiology, biochemistry). Choose this "
            "direction when you want the writer to test *why* something "
            "happens at a mechanistic or structural level rather than how to "
            "diagnose or manage a patient."
        ),
        # No subcomponents -> lead-ins attach directly.
        "lead_ins": [
            "Which of the following is the most likely cause/mechanism of this effect?",
            "Which of the following is the most likely infectious agent?",
            "Which of the following is the most likely explanation for these findings?",
            "Which of the following is the most likely location of this patient’s lesion?",
            "Which of the following is the most likely pathogen?",
            "Which of the following findings is most likely to be increased/decreased in this patient?",
            "A biopsy specimen is most likely to show which of the following?",
            "This patient most likely has a defect in which of the following?",
            "This patient most likely has a deficiency in which of the following enzymes?",
            "Which of the following cytokines is the most likely cause of this condition?",
            "Which of the following structures is at greatest risk for damage during this procedure?",
            "The most appropriate medication for this patient will have which of the following mechanisms of action?",
        ],
    },
    "Diagnosis": {
        "description": (
            "Assesses the clinical reasoning used to identify a patient’s "
            "condition—gathering history and physical findings, selecting "
            "and interpreting studies, formulating the diagnosis, and "
            "predicting prognosis. Choose this direction when you want the "
            "writer to test *what is going on* with the patient."
        ),
        "subcomponents": {
            "Obtaining and Predicting History and Physical Examination": {
                "description": (
                    "Tests the ability to gather and prioritize the most "
                    "relevant historical and physical-examination information."
                ),
                "lead_ins": [
                    "Which of the following factors in this patient’s history most increased her risk for developing this condition?",
                    "Which of the following additional information regarding this patient’s history is most appropriate to obtain at this time?",
                    "Which of the following is the most appropriate focus of the physical examination at this time?",
                ],
            },
            "Selecting and Interpreting Diagnostic Studies": {
                "description": (
                    "Tests the ability to choose appropriate diagnostic "
                    "studies and to interpret their results."
                ),
                "lead_ins": [
                    "Which of the following is the most appropriate diagnostic study to obtain at this time?",
                    "Which of the following laboratory studies is most likely to confirm the diagnosis?",
                    "Which of the following is the most likely explanation for these laboratory findings?",
                    "Arterial blood gas analysis is most likely to show which of the following sets of findings?",
                ],
            },
            "Formulating the Diagnosis": {
                "description": (
                    "Tests the ability to synthesize available information "
                    "into the single most likely diagnosis."
                ),
                "lead_ins": [
                    "Which of the following is the most likely diagnosis?",
                ],
            },
            "Determining Prognosis/Outcome": {
                "description": (
                    "Tests the ability to anticipate the likely course, "
                    "complications, or outcome of a patient’s condition."
                ),
                "lead_ins": [
                    "Based on these findings, this patient is most likely to develop which of the following?",
                    "Which of the following is the most likely complication of this patient’s current condition?",
                ],
            },
        },
    },
    "Management": {
        "description": (
            "Assesses the clinical decisions involved in caring for the "
            "patient—health maintenance and disease prevention, and "
            "pharmacotherapy / clinical interventions and treatments. Choose "
            "this direction when you want the writer to test *what to do* for "
            "the patient."
        ),
        "subcomponents": {
            "Health Maintenance and Disease Prevention": {
                "description": (
                    "Tests prevention, screening, and risk-reduction "
                    "decisions."
                ),
                "lead_ins": [
                    "Which of the following vaccines is most appropriate to administer at this time?",
                    "Which of the following is the most appropriate screening test?",
                    "Which of the following tests would have predicted these findings?",
                    "Which of the following is the most appropriate intervention?",
                    "For which of the following conditions is this patient at greatest risk?",
                    "Which of the following is most likely to have prevented this condition?",
                    "Which of the following is the most appropriate next step in management to prevent [morbidity/mortality/disability]?",
                    "Which of the following is the most appropriate recommendation to prevent disability from this patient’s injury/condition?",
                ],
            },
            "Pharmacotherapy/Clinical Interventions and Treatments": {
                "description": (
                    "Tests treatment selection, prioritization of care, and "
                    "next steps in patient care."
                ),
                "lead_ins": [
                    "Which of the following is the most appropriate initial or next step in patient care?",
                    "Which of the following is the most appropriate management?",
                    "Which of the following is the most appropriate pharmacotherapy?",
                    "Which of the following is the first priority in caring for this patient?",
                ],
            },
        },
    },
}


# The vignette template fields, presented to the writer in this exact order.
VIGNETTE_FIELDS = [
    "Age & Gender",
    "Site of care",
    "Presenting symptoms",
    "Duration of symptoms",
    "Patient history",
    "Physical findings",
    "Results of diagnostic studies",
    "Initial treatment, subsequent findings",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def get_lead_ins(competency: str, subcomponent: str | None) -> list[str]:
    """Return the list of lead-ins for the chosen competency / subcomponent."""
    node = TAXONOMY[competency]
    if "lead_ins" in node:
        # Foundational science: lead-ins live directly on the competency.
        return node["lead_ins"]
    # Diagnosis / Management: lead-ins live on the chosen subcomponent.
    if subcomponent and subcomponent in node.get("subcomponents", {}):
        return node["subcomponents"][subcomponent]["lead_ins"]
    return []


def build_prompt(field_values: dict[str, str], lead_in: str) -> str:
    """Construct the prompt sent to Gemini from the populated vignette data."""
    # Only include fields the user actually filled in.
    populated = {k: v.strip() for k, v in field_values.items() if v and v.strip()}

    details = "\n".join(f"- {label}: {value}" for label, value in populated.items())
    if not details:
        details = "(No patient details were provided.)"

    prompt = f"""You are an expert NBME item writer. Using ONLY the structured \
patient data below, write a single, cohesive clinical vignette opening \
paragraph in standard NBME style.

Patient data:
{details}

Requirements:
- Write one cohesive paragraph (no bullet points, no headings, no labels).
- Use standard clinical linking phrases, e.g. "A 45-year-old woman presents \
to the emergency department because of...".
- Integrate the provided details naturally and in clinically logical order. \
Do not invent significant new clinical facts that were not provided.
- Use formal, professional clinical language consistent with NBME exam items.
- Output ONLY the cohesive paragraph, then a single line break, then the exact \
text of the following lead-in (reproduced verbatim, with no additional commentary):

{lead_in}
"""
    return prompt


def get_client() -> genai.Client:
    """Build a Gemini client using the key from Streamlit secrets."""
    # The key is pulled securely from Streamlit secrets (never hard-coded).
    return genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


def get_model_name() -> str:
    """Return the model ID, allowing an override via secrets."""
    return st.secrets.get("GEMINI_MODEL", DEFAULT_MODEL)


def list_available_models() -> list[str]:
    """Return model IDs the API key can use for generateContent."""
    client = get_client()
    available = []
    for model in client.models.list():
        actions = getattr(model, "supported_actions", None) or []
        if "generateContent" in actions:
            # Strip the leading "models/" prefix for readability.
            available.append(model.name.split("/")[-1])
    return sorted(available)


def generate_stem(prompt: str) -> str:
    """Call the Gemini API via the google-genai client and return the text."""
    client = get_client()
    response = client.models.generate_content(
        model=get_model_name(),
        contents=prompt,
    )
    return response.text


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
# Streamlit reruns top-to-bottom on every interaction, so persistent progress
# must live in st.session_state.
if "generated_stem" not in st.session_state:
    st.session_state.generated_stem = ""

# Each vignette field gets its own session_state key so its value survives reruns.
for _field in VIGNETTE_FIELDS:
    _key = f"field_{_field}"
    if _key not in st.session_state:
        st.session_state[_key] = ""


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="NBME Item Generator", page_icon="🩺", layout="centered")

st.title("🩺 NBME Item Generator")
st.caption(
    "A guided tool for writing high-quality NBME-style multiple-choice question "
    "stems. Work through the three steps below."
)


# ===========================================================================
# Step 1 - The Target Planner (Branching Logic)
# ===========================================================================
st.header("Step 1 · The Target Planner")
st.markdown(
    "Decide **where you want the question to go**. The *lead-in* is the "
    "direction the writer wants to take the question—essentially the "
    "knowledge you are trying to assess. Start by choosing the Main Task "
    "Competency, then (if applicable) a Subcomponent."
)

competency = st.radio(
    "Main Task Competency",
    options=list(TAXONOMY.keys()),
    help="The broad category of knowledge or skill you want to assess.",
)

# Show the instructional text for the chosen competency.
st.info(TAXONOMY[competency]["description"])

subcomponent = None
node = TAXONOMY[competency]

# Subcomponent selector is shown ONLY when the competency has subcomponents.
# (Hidden entirely for Foundational (Basic) Science.)
if "subcomponents" in node:
    subcomponent = st.selectbox(
        "Subcomponent",
        options=list(node["subcomponents"].keys()),
        help="Narrows the competency to a specific assessment focus.",
    )
    st.info(node["subcomponents"][subcomponent]["description"])


# ===========================================================================
# Step 2 - The Vignette Builder
# ===========================================================================
st.header("Step 2 · The Vignette Builder")
st.markdown(
    "Fill in the patient vignette template below. **None of these fields are "
    "required**—skip any that are not relevant to your item."
)

field_values: dict[str, str] = {}

# Short, single-line fields use text_input; longer narrative fields use text_area.
short_fields = {"Age & Gender", "Site of care", "Duration of symptoms"}

for field in VIGNETTE_FIELDS:
    key = f"field_{field}"
    if field in short_fields:
        field_values[field] = st.text_input(field, key=key)
    else:
        field_values[field] = st.text_area(field, key=key)

# --- Lead-in Selection (placed at the very bottom of the builder) ----------
st.subheader("Lead-in")
st.markdown(
    "Select the specific lead-in for your item. The options below are filtered "
    "based on your Step 1 selections."
)

lead_in_options = get_lead_ins(competency, subcomponent)
selected_lead_in = st.radio(
    "Select the lead-in (the question that follows the vignette):",
    options=lead_in_options,
)


# ===========================================================================
# Step 3 - API Generation & Output
# ===========================================================================
st.header("Step 3 · Generate & Edit")

if st.button("Generate Question Stem", type="primary"):
    if "GEMINI_API_KEY" not in st.secrets:
        st.error(
            "No Gemini API key found. Add GEMINI_API_KEY to your "
            ".streamlit/secrets.toml file (see the comments at the top of "
            "app.py)."
        )
    else:
        prompt = build_prompt(field_values, selected_lead_in)
        with st.spinner("Contacting Gemini..."):
            try:
                st.session_state.generated_stem = generate_stem(prompt)
            except Exception as exc:  # Surface API/config errors to the user.
                st.error(f"Generation failed: {exc}")
                # A 404 / "not found" usually means the model ID is wrong for
                # this key. Show the IDs that actually work so the user can set
                # GEMINI_MODEL in secrets accordingly.
                if "not found" in str(exc).lower() or "404" in str(exc):
                    try:
                        models = list_available_models()
                        if models:
                            st.info(
                                "Models available to your API key (set one as "
                                "`GEMINI_MODEL` in your secrets):\n\n"
                                + "\n".join(f"- `{m}`" for m in models)
                            )
                    except Exception:
                        pass  # If even listing fails, the first error is enough.

# Render the result in an editable text area so the writer can refine it
# before copying.
st.text_area(
    "Generated question stem (editable):",
    key="generated_stem",
    height=300,
    help="Edit the generated text as needed, then copy it into your item bank.",
)

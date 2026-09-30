import io
import json
import os
import base64
import re
from typing import Any

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from groq import Groq


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Quantity Takeoff",
    page_icon="📐",
    layout="wide",
)


# ============================================================
# CONFIG
# ============================================================

MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


# ============================================================
# AI SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert Construction Quantity Surveyor and Architectural Drawing Analyst.

Your job is to inspect architectural drawing pages and extract ONLY dimensions
and construction elements that can be reliably verified from the supplied drawing.

STRICT ANTI-HALLUCINATION RULES:

1. NEVER invent a dimension.
2. NEVER estimate a dimension from visual proportions.
3. NEVER assume a standard door/window/room size.
4. Only report dimensions that are visibly readable in the drawing or clearly
   supported by drawing text.
5. If a dimension cannot be verified, mark the item as unable_to_verify.
6. Preserve the original dimension text in source_dimension.
7. Convert metric dimensions to millimetres where possible.
8. Do not calculate quantity unless the required dimensions are available.
9. Confidence must represent visual certainty.
10. The drawing IMAGE is the primary evidence. Extracted PDF text is supporting evidence.
11. If scale is visible, report it. Do not infer scale.
12. For rooms, calculate area only when explicit length and width dimensions
    are available.
13. Do not treat notes/specifications as physical dimensions unless clearly
    associated with the drawing element.
14. Return ONLY valid JSON.

Allowed element_type:
room, wall, door, window, beam, column, slab, stair, other

Allowed status:
verified_from_drawing, candidate, unable_to_verify

Quantity rules:

m2 = numbers × length_m × width_m

m3 = numbers × length_m × width_m × height_m

m = numbers × length_m

nos = numbers


Return exactly this JSON structure:

{
  "page": 1,
  "drawing_scale": null,
  "drawing_units": "mm",
  "elements": [
    {
      "description": "Bedroom 1",
      "element_type": "room",
      "numbers": 1,
      "length_mm": 4200,
      "width_mm": 3600,
      "height_mm": null,
      "unit": "m2",
      "quantity": 15.12,
      "source_dimension": "4200 x 3600",
      "confidence": 0.96,
      "status": "verified_from_drawing"
    }
  ],
  "notes": []
}
"""


# ============================================================
# HELPERS
# ============================================================

def get_secret(name: str, default: str = "") -> str:
    """
    Read value from Streamlit Secrets first, then environment variables.
    """
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass

    return os.getenv(name, default)


def get_groq_client():
    api_key = get_secret("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def render_page(page: fitz.Page, dpi: int = 130) -> bytes:
    """
    Render PDF page to JPEG.
    """
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)

    pix = page.get_pixmap(
        matrix=matrix,
        alpha=False,
    )

    return pix.tobytes("jpeg", jpg_quality=82)


def image_to_data_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def safe_float(value: Any):
    try:
        if value is None or value == "":
            return None

        return float(value)

    except Exception:
        return None


def extract_dimension_candidates(text: str):
    """
    Extract obvious dimensions from PDF text.
    This is supporting evidence only.
    """

    patterns = [
        r"\b\d+(?:\.\d+)?\s*(?:mm|cm|m)\b",
        r"\b\d{2,5}\s*[xX×]\s*\d{2,5}\b",
    ]

    results = []

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        results.extend(matches)

    return list(dict.fromkeys(results))


def normalize_ai_element(item: dict, page_number: int):
    """
    Convert AI response into our internal takeoff format.

    IMPORTANT:
    Quantity is recalculated locally rather than blindly trusting
    the AI arithmetic.
    """

    numbers = safe_float(item.get("numbers")) or 1.0

    length_mm = safe_float(item.get("length_mm"))
    width_mm = safe_float(item.get("width_mm"))
    height_mm = safe_float(item.get("height_mm"))

    unit = str(
        item.get("unit") or ""
    ).lower().strip()

    status = str(
        item.get("status") or "unable_to_verify"
    )

    confidence = safe_float(
        item.get("confidence")
    ) or 0.0

    quantity = None

    # Only calculate verified high-confidence items.
    if status == "verified_from_drawing" and confidence >= 0.90:

        if (
            unit == "m2"
            and length_mm
            and width_mm
        ):
            quantity = (
                numbers
                * (length_mm / 1000)
                * (width_mm / 1000)
            )

        elif (
            unit == "m3"
            and length_mm
            and width_mm
            and height_mm
        ):
            quantity = (
                numbers
                * (length_mm / 1000)
                * (width_mm / 1000)
                * (height_mm / 1000)
            )

        elif (
            unit == "m"
            and length_mm
        ):
            quantity = (
                numbers
                * (length_mm / 1000)
            )

        elif unit == "nos":
            quantity = numbers

    return {
        "Page": page_number,
        "Description": str(
            item.get("description") or "Unspecified"
        ),
        "Element Type": str(
            item.get("element_type") or "other"
        ),
        "Nos": numbers,
        "Length (m)": (
            round(length_mm / 1000, 3)
            if length_mm
            else None
        ),
        "Width (m)": (
            round(width_mm / 1000, 3)
            if width_mm
            else None
        ),
        "Height (m)": (
            round(height_mm / 1000, 3)
            if height_mm
            else None
        ),
        "Unit": unit,
        "Quantity": (
            round(quantity, 3)
            if quantity is not None
            else None
        ),
        "Source Dimension": str(
            item.get("source_dimension") or ""
        ),
        "Confidence": round(
            confidence,
            2,
        ),
        "Status": status,
    }


# ============================================================
# AI ANALYSIS
# ============================================================

def analyze_page(
    client,
    page_number: int,
    image_bytes: bytes,
    extracted_text: str,
):

    text_context = (
        extracted_text[:12000]
        if extracted_text
        else "(No machine-readable PDF text found.)"
    )

    user_prompt = f"""
Analyze architectural drawing page {page_number}.

The PDF's machine-readable text is provided below as supporting evidence.

--- PDF TEXT START ---

{text_context}

--- PDF TEXT END ---

IMPORTANT:

- The drawing IMAGE is the primary evidence.
- Read dimensions visible in the drawing.
- Do not estimate dimensions from visual proportions.
- Do not assume standard dimensions.
- If a dimension is not readable, mark it unable_to_verify.
- Preserve original dimension notation in source_dimension.
- Return JSON only.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_to_data_url(
                                image_bytes
                            )
                        },
                    },
                ],
            },
        ],
        temperature=0,
        max_completion_tokens=5000,
        response_format={
            "type": "json_object"
        },
    )

    content = (
        response.choices[0]
        .message
        .content
        or "{}"
    )

    return json.loads(content)


# ============================================================
# EXCEL
# ============================================================

def create_excel(df: pd.DataFrame):
    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl",
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Quantity Takeoff",
        )

        worksheet = writer.book[
            "Quantity Takeoff"
        ]

        worksheet.freeze_panes = "A2"

        worksheet.auto_filter.ref = (
            worksheet.dimensions
        )

        for column in worksheet.columns:

            max_length = 0

            for cell in column:

                try:
                    length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        length,
                    )

                except Exception:
                    pass

            worksheet.column_dimensions[
                column[0].column_letter
            ].width = min(
                max_length + 2,
                35,
            )

        # Summary sheet

        summary = (
            df.groupby(
                [
                    "Element Type",
                    "Unit",
                ],
                dropna=False,
            )["Quantity"]
            .sum(min_count=1)
            .reset_index()
        )

        summary.to_excel(
            writer,
            index=False,
            sheet_name="Summary",
        )

    return output.getvalue()


# ============================================================
# UI
# ============================================================

st.title("📐 AI Quantity Takeoff")

st.caption(
    "Architectural PDF → AI Drawing Scan → "
    "Dimensions → Quantity Takeoff → Excel"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Project Settings")

    project_name = st.text_input(
        "Project Name",
        "My Construction Project",
    )

    max_pages = st.number_input(
        "Maximum pages to scan",
        min_value=1,
        max_value=20,
        value=5,
    )

    dpi = st.selectbox(
        "Drawing image quality",
        [110, 130, 150],
        index=1,
    )

    st.divider()

    st.subheader("AI Settings")

    st.write(
        f"Model: `{MODEL}`"
    )

    if get_secret("GROQ_API_KEY"):

        st.success(
            "Groq API key detected"
        )

    else:

        st.warning(
            "Groq API key not detected"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload Architectural Drawing PDF",
    type=["pdf"],
    help=(
        "For the first test, upload a small "
        "1–5 page architectural drawing."
    ),
)


if uploaded_file:

    pdf_bytes = uploaded_file.getvalue()

    try:

        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        )

    except Exception as error:

        st.error(
            f"Could not open PDF: {error}"
        )

        st.stop()

    st.success(
        f"PDF loaded successfully — "
        f"{len(document)} page(s)"
    )


    # ========================================================
    # PREVIEW
    # ========================================================

    left, right = st.columns(
        [1, 1]
    )

    with left:

        st.subheader(
            "📄 Drawing Preview"
        )

        preview_page = st.number_input(
            "Preview Page",
            min_value=1,
            max_value=len(document),
            value=1,
        )

        page = document[
            preview_page - 1
        ]

        preview_image = render_page(
            page,
            dpi=100,
        )

        st.image(
            preview_image,
            caption=f"Page {preview_page}",
            use_container_width=True,
        )


    # ========================================================
    # SCAN
    # ========================================================

    with right:

        st.subheader(
            "🤖 AI Drawing Scan"
        )

        st.write(
            "The AI will inspect the drawing "
            "and extract only dimensions it "
            "can verify."
        )

        scan_button = st.button(
            "🔎 Scan Drawing with AI",
            type="primary",
            use_container_width=True,
        )


        if scan_button:

            client = get_groq_client()

            if client is None:

                st.error(
                    "Groq API key is missing."
                )

                st.info(
                    "Add GROQ_API_KEY in "
                    "Streamlit → App Settings → Secrets."
                )

                st.stop()


            rows = []
            detected_scales = []
            notes = []

            pages_to_scan = min(
                len(document),
                int(max_pages),
            )

            progress = st.progress(0)


            for index in range(
                pages_to_scan
            ):

                page_number = index + 1

                with st.status(
                    f"Scanning page "
                    f"{page_number}/{pages_to_scan}..."
                ):

                    page = document[
                        index
                    ]

                    extracted_text = (
                        page.get_text(
                            "text"
                        )
                    )

                    candidates = (
                        extract_dimension_candidates(
                            extracted_text
                        )
                    )

                    image_bytes = render_page(
                        page,
                        dpi=int(dpi),
                    )

                    try:

                        result = analyze_page(
                            client,
                            page_number,
                            image_bytes,
                            extracted_text,
                        )

                        if result.get(
                            "drawing_scale"
                        ):

                            detected_scales.append(
                                str(
                                    result[
                                        "drawing_scale"
                                    ]
                                )
                            )


                        for item in result.get(
                            "elements",
                            [],
                        ):

                            if isinstance(
                                item,
                                dict,
                            ):

                                rows.append(
                                    normalize_ai_element(
                                        item,
                                        page_number,
                                    )
                                )


                        if result.get(
                            "notes"
                        ):

                            notes.extend(
                                [
                                    str(x)
                                    for x in result[
                                        "notes"
                                    ]
                                ]
                            )


                    except Exception as error:

                        st.warning(
                            f"Page {page_number} "
                            f"could not be analyzed: "
                            f"{error}"
                        )

                    progress.progress(
                        page_number
                        / pages_to_scan
                    )


            st.session_state[
                "takeoff_rows"
            ] = rows

            st.session_state[
                "detected_scales"
            ] = list(
                dict.fromkeys(
                    detected_scales
                )
            )

            st.session_state[
                "scan_notes"
            ] = notes

            st.session_state[
                "scan_complete"
            ] = True

            st.success(
                f"Scan complete — "
                f"{len(rows)} element(s) found."
            )


# ============================================================
# TAKEOFF TABLE
# ============================================================

if st.session_state.get(
    "scan_complete"
):

    st.divider()

    st.subheader(
        "📊 Quantity Takeoff"
    )

    rows = st.session_state.get(
        "takeoff_rows",
        [],
    )


    if not rows:

        st.info(
            "No verified measurements "
            "were returned. Try a clearer "
            "drawing or fewer pages."
        )

    else:

        dataframe = pd.DataFrame(
            rows
        )


        edited_dataframe = (
            st.data_editor(
                dataframe,
                use_container_width=True,
                num_rows="dynamic",
                column_config={
                    "Confidence":
                        st.column_config.NumberColumn(
                            "Confidence",
                            min_value=0.0,
                            max_value=1.0,
                            step=0.01,
                        ),

                    "Quantity":
                        st.column_config.NumberColumn(
                            "Quantity",
                            format="%.3f",
                        ),
                },

                disabled=[
                    "Page",
                    "Status",
                ],

                key="takeoff_editor",
            )
        )


        st.session_state[
            "edited_dataframe"
        ] = edited_dataframe


        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Total Rows",
                len(
                    edited_dataframe
                ),
            )

        with col2:

            quantity_count = (
                edited_dataframe[
                    "Quantity"
                ]
                .notna()
                .sum()
            )

            st.metric(
                "Rows With Quantity",
                int(
                    quantity_count
                ),
            )


        excel_file = create_excel(
            edited_dataframe
        )


        st.download_button(
            "📥 Download Excel Quantity Takeoff",
            data=excel_file,
            file_name=(
                project_name
                .replace(" ", "_")
                + "_quantity_takeoff.xlsx"
            ),
            mime=(
                "application/vnd.openxmlformats-"
                "officedocument.spreadsheetml.sheet"
            ),
            type="primary",
        )


    # ========================================================
    # SCALE
    # ========================================================

    scales = st.session_state.get(
        "detected_scales",
        [],
    )

    if scales:

        st.info(
            "Detected drawing scale(s): "
            + ", ".join(scales)
        )


    # ========================================================
    # NOTES
    # ========================================================

    notes = st.session_state.get(
        "scan_notes",
        [],
    )

    if notes:

        with st.expander(
            "AI Notes"
        ):

            for note in notes:

                st.write(
                    "•",
                    note,
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "MVP warning: Always verify AI-extracted "
    "dimensions against the original drawing "
    "before using quantities for procurement, "
    "tendering, billing, or contractual work."
)

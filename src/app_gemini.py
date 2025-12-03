import streamlit as st
import pandas as pd
import os
from io import BytesIO
from dotenv import load_dotenv

# PDF generation libs
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

# Cosine similarity engine - Assuming this file now contains the Gemini implementation
# We change the import from 'similarity_engine_openai' to 'similarity_engine'
from similarity_engine_gemini import CSVCosineSimilarity


# ----------------------------------------------------
# ──────────────── APP SETUP ────────────────────────
# ----------------------------------------------------

load_dotenv()

st.set_page_config(page_title="Trade settlement comparison Tool", layout="wide")
st.title("📊 Trade settlement comparison")

# --- Change 1: Use GEMINI_API_KEY ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ GEMINI_API_KEY not found. Please set it in your .env file.")


# ----------------------------------------------------
# ──────────────── CSV UPLOAD SECTION ───────────────
# ----------------------------------------------------

col1, col2 = st.columns(2)

left_df = None
right_df = None

with col1:
    left_file = st.file_uploader("Upload **Actual CSV**", type=["csv"], key="left")
    if left_file:
        left_df = pd.read_csv(left_file)
        left_df.index += 1      # Start index at 1
        st.dataframe(left_df)

with col2:
    right_file = st.file_uploader("Upload **CSV to Compare**", type=["csv"], key="right")
    if right_file:
        right_df = pd.read_csv(right_file)
        right_df.index += 1
        st.dataframe(right_df)

st.divider()


# ----------------------------------------------------
# ───────────── COLUMN SELECTION SECTION ────────────
# ----------------------------------------------------

left_columns = []
right_columns = []

if left_df is not None and right_df is not None:

    st.subheader("📝 Select Text Columns for Comparison")
    col1, col2 = st.columns(2)

    with col1:
        default_left = [c for c in left_df.columns if left_df[c].dtype == "object"]
        left_columns = st.multiselect("Select Actual CSV text columns:",
                                      options=list(left_df.columns),
                                      default=default_left)

    with col2:
        default_right = [c for c in right_df.columns if right_df[c].dtype == "object"]
        right_columns = st.multiselect("Select Compare CSV text columns:",
                                       options=list(right_df.columns),
                                       default=default_right)


# ----------------------------------------------------
# ──────────────── RUN COMPARISON ───────────────────
# ----------------------------------------------------

if st.button("🔍 Compare"):

    if left_df is None or right_df is None:
        st.error("❌ Please upload both CSV files.")
        st.stop()

    if not left_columns or not right_columns:
        st.error("❌ Select at least one text column from each CSV.")
        st.stop()

    # --- Change 2: Pass GEMINI_API_KEY to the engine ---
    engine = CSVCosineSimilarity(gemini_api_key=GEMINI_API_KEY)

    with st.spinner("Comparing rows using Gemini embeddings..."):
        result_df = engine.compute_row_similarity(left_df, right_df,
                                                 left_columns, right_columns)

    st.success("✅ Comparison Completed!")
    st.dataframe(result_df, use_container_width=True)


    # ------------------------------------------------
    # ─────────────── CSV DOWNLOAD ───────────────────
    # ------------------------------------------------

    st.subheader("⬇ Download Reports")

    csv_buffer = BytesIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button("⬇ Download CSV Report",
                       data=csv_buffer.getvalue(),
                       file_name="similarity_report.csv",
                       mime="text/csv")


    # ------------------------------------------------
    # ─────────────── PDF DOWNLOAD ───────────────────
    # ------------------------------------------------

    pdf_buffer = BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(letter),
        leftMargin=0.3 * inch,
        rightMargin=0.3 * inch,
        topMargin=0.3 * inch,
        bottomMargin=0.3 * inch,
    )

    styles = getSampleStyleSheet()

    title = Paragraph("Trade Settlement Similarity Report", styles["Title"])
    elements = [title, Spacer(1, 0.2 * inch)]

    # Table Styles
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=7, leading=9)
    center_small = ParagraphStyle("center_small", parent=small, alignment=1)
    header_style = ParagraphStyle("header", parent=styles["Normal"],
                                  fontSize=8, alignment=1, fontName="Helvetica-Bold")

    # Convert DataFrame to table data
    table_data = [[Paragraph(col, header_style) for col in result_df.columns]]

    for _, row in result_df.iterrows():
        cells = []
        for col, val in row.items():
            txt = "" if pd.isna(val) else str(val)
            if col in ("Left Text", "Right Text"):
                cells.append(Paragraph(txt, small))
            else:
                cells.append(Paragraph(txt, center_small))
        table_data.append(cells)

    col_widths = [
        0.8 * inch, 3.5 * inch,
        0.8 * inch, 3.5 * inch,
        0.7 * inch, 1.2 * inch
    ]

    table = Table(table_data, colWidths=col_widths, repeatRows=1)

    table_style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4A90E2")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ])

    # Row highlighting
    for i in range(1, len(table_data)):
        comment = table_data[i][-1].getPlainText().strip()
        if comment == "Forward to OPs":
            table_style.add("BACKGROUND", (0, i), (-1, i), colors.HexColor("#FFF2CC"))
        elif comment == "OK":
            table_style.add("BACKGROUND", (0, i), (-1, i), colors.HexColor("#D5F5E3"))

    table.setStyle(table_style)
    elements.append(table)

    doc.build(elements)

    pdf_buffer.seek(0)

    st.download_button("⬇ Download PDF Report",
                       data=pdf_buffer.getvalue(),
                       file_name="similarity_report.pdf",
                       mime="application/pdf")
"""
Streamlit app for HITL extraction review and ground truth creation.

Usage:
    streamlit run pipeline/review/streamlit_app.py -- --exp-dir <path> --fund "Fund Name"
"""

import argparse
import re
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from pipeline.review.review_session import (
    ReviewSession,
    build_session_from_results,
    FieldReview,
    get_display_name,
    get_field_sort_key,
)
from pipeline.review.gt_exporter import export_ground_truth


def parse_args():
    """Parse CLI args passed after '--' in streamlit run command."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="Experiment directory path")
    parser.add_argument("--fund", required=True, help="Fund name to review")
    # Streamlit passes extra args; ignore them
    args, _ = parser.parse_known_args()
    return args


def get_session() -> ReviewSession:
    """Get or create the review session."""
    if "session" not in st.session_state:
        args = parse_args()
        exp_dir = args.exp_dir
        fund_name = args.fund

        # Check for existing saved session
        session_path = Path(exp_dir) / f"review_session_{fund_name.lower().replace(' ', '_')[:50]}.json"
        if session_path.exists():
            st.session_state.session = ReviewSession.load(session_path)
        else:
            st.session_state.session = build_session_from_results(exp_dir, fund_name)

    return st.session_state.session


def resolve_filing_path(session: ReviewSession) -> Path | None:
    """Resolve the filing path to an absolute path, trying common locations."""
    if not session.filing_path:
        return None
    p = Path(session.filing_path)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to project root
    resolved = project_root / p
    if resolved.exists():
        return resolved
    return None


@st.cache_data
def load_filing_html(filing_path: str) -> str | None:
    """Load the primary.html from the filing directory."""
    p = Path(filing_path)
    html_file = p / "primary.html" if p.is_dir() else p
    if html_file.exists():
        return html_file.read_text(encoding="utf-8", errors="replace")
    return None


def extract_section_html(full_html: str, section_name: str) -> str | None:
    """Extract a section from the full HTML by searching for the section title.

    Returns a chunk of HTML around where the section title appears.
    """
    if not section_name or not full_html:
        return None

    # Search for the section title in the HTML
    idx = full_html.lower().find(section_name.lower())
    if idx == -1:
        # Try partial match with key words
        words = section_name.split()
        for i in range(len(words), 0, -1):
            partial = " ".join(words[:i])
            idx = full_html.lower().find(partial.lower())
            if idx != -1:
                break

    if idx == -1:
        return None

    # Extract a window of HTML around the match
    # Find enclosing table or section
    context_before = 2000
    context_after = 8000
    start = max(0, idx - context_before)
    end = min(len(full_html), idx + context_after)

    # Try to find enclosing <table> tag
    table_start = full_html.rfind("<table", start, idx)
    if table_start != -1:
        start = table_start
        # Find corresponding </table>
        table_end = full_html.find("</table>", idx)
        if table_end != -1:
            end = table_end + len("</table>")

    return full_html[start:end]


def get_section_name_from_evidence(evidence: str) -> str | None:
    """Extract section name from evidence text like '[Section: Foo Bar]'."""
    m = re.search(r'\[Section:\s*([^\]]+)\]', evidence)
    return m.group(1).strip() if m else None


def clean_garbled_text(text: str) -> str:
    """Fix common text extraction issues where spaces are stripped."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([a-z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-z])', _maybe_insert_space, text)
    return text


def _maybe_insert_space(m: re.Match) -> str:
    """Insert space between digit and letter unless it's an ordinal suffix."""
    digit, letter = m.group(1), m.group(2)
    suffix_starts = {'s', 't', 'n', 'r'}
    if letter in suffix_starts:
        return m.group(0)
    return f"{digit} {letter}"


def format_evidence(text: str) -> str:
    """Add line breaks to evidence text so it reads as paragraphs."""
    # Newline before/after [Section: ...] markers
    text = re.sub(r'(\S)\s*(\[Section:)', r'\1\n\n\2', text)
    text = re.sub(r'(\[Section:[^\]]+\])\s*', r'\1\n', text)

    # Newline before capitalized headers after % or )
    text = re.sub(r'(%\s*)([A-Z][a-z]+ [A-Z])', r'%\n\2', text)
    text = re.sub(r'(\)\s*)([A-Z][a-z]+ [A-Z])', r')\n\2', text)

    # Break after sentence-ending punctuation
    text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\2', text)

    # Break before "Class X Shares" patterns
    text = re.sub(r'\s+(Class\s+\S+\s+Shares)', r'\n\1', text)

    # Break before common section keywords
    keywords = [
        'Annual fund', 'Annual Fund', 'Management fee', 'Management Fee',
        'Distribution fee', 'Distribution Fee', 'Total annual', 'Total Annual',
        'Shareholder transaction', 'Shareholder Transaction',
    ]
    for kw in keywords:
        text = text.replace(f' {kw}', f'\n{kw}')

    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def confidence_color(review: FieldReview) -> str:
    """Return color indicator based on field confidence."""
    if review.confidence == "validated" and review.grounded:
        return "🟢"
    if review.extracted_value is None:
        return "🔴"
    if review.confidence == "rejected":
        return "🔴"
    return "🟡"


def render_field(review: FieldReview, idx: int, filing_html: str | None):
    """Render a single field review card."""
    color = confidence_color(review)
    status = "✅" if review.is_reviewed else "⬜"
    display_name = get_display_name(review.field_key)

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{status} {color} {display_name}**")
            st.caption(f"`{review.field_key}`")
        with col2:
            st.caption(f"Category: {review.category}")

        # Extracted value
        val_display = repr(review.extracted_value) if review.extracted_value is not None else "*null*"
        st.markdown(f"**Extracted Value:** `{val_display}`")

        # Metadata row
        meta_cols = st.columns(3)
        with meta_cols[0]:
            st.caption(f"Confidence: {review.confidence or 'unknown'}")
        with meta_cols[1]:
            grounded_str = "✓" if review.grounded else ("✗" if review.grounded is False else "?")
            st.caption(f"Grounded: {grounded_str}")
        with meta_cols[2]:
            st.caption(f"Source: {review.tier or 'unknown'}")

        # Evidence display
        if review.evidence:
            formatted = format_evidence(clean_garbled_text(review.evidence))
            preview_len = 500
            with st.expander("Evidence (extracted text)", expanded=len(formatted) <= preview_len):
                st.code(formatted, language=None)

            # HTML source view: show the original HTML section with table formatting
            if filing_html:
                section_name = get_section_name_from_evidence(review.evidence)
                key = f"html_{idx}_{review.field_key}"
                with st.expander("View HTML source (tables preserved)"):
                    section_html = None
                    if section_name:
                        section_html = extract_section_html(filing_html, section_name)
                    if section_html:
                        st.html(section_html)
                    else:
                        # Fallback: search for the extracted value in the HTML
                        val_str = str(review.extracted_value) if review.extracted_value else None
                        if val_str:
                            found = extract_section_html(filing_html, val_str)
                            if found:
                                st.html(found)
                            else:
                                st.caption("Could not locate this field in the HTML source.")
                        else:
                            st.caption("No section reference found in evidence.")

        # Full document viewer
        if filing_html:
            with st.expander("Browse full filing"):
                search_term = st.text_input(
                    "Search in document",
                    key=f"search_{idx}_{review.field_key}",
                    placeholder="Enter text to search for...",
                )
                if search_term:
                    # Highlight search term and show surrounding HTML
                    found_html = extract_section_html(filing_html, search_term)
                    if found_html:
                        highlighted = found_html.replace(
                            search_term,
                            f'<mark style="background:yellow;font-weight:bold">{search_term}</mark>',
                        )
                        st.html(highlighted)
                    else:
                        st.warning(f"'{search_term}' not found in filing.")
                else:
                    st.caption("Enter a search term to find relevant sections in the source document.")

        # Reasoning (collapsible)
        if review.reasoning:
            with st.expander("Reasoning", expanded=False):
                st.text(review.reasoning)

        # Decision buttons
        key_prefix = f"field_{idx}_{review.field_key}"

        options = ["accept", "correct", "na"]
        if review.decision and review.decision in options:
            radio_index = options.index(review.decision)
        else:
            radio_index = None

        decision = st.radio(
            "Decision",
            options=options,
            index=radio_index,
            horizontal=True,
            key=f"{key_prefix}_decision",
            format_func=lambda x: {"accept": "✓ Accept", "correct": "✎ Correct", "na": "N/A"}[x],
        )

        corrected_value = None
        if decision == "correct":
            default_val = str(review.corrected_value) if review.corrected_value is not None else ""
            corrected_value = st.text_input(
                "Corrected value",
                value=default_val,
                key=f"{key_prefix}_corrected",
            )

        notes = st.text_input(
            "Notes (optional)",
            value=review.reviewer_notes,
            key=f"{key_prefix}_notes",
        )

        # Apply decision only when user has made a selection
        if decision is not None:
            review.decision = decision
            review.reviewer_notes = notes
            review.reviewed_at = datetime.now().isoformat()
            if decision == "correct" and corrected_value is not None:
                review.corrected_value = corrected_value


def main():
    st.set_page_config(page_title="Extraction Review", layout="wide")

    session = get_session()

    # Resolve filing path and load HTML once
    filing_path = resolve_filing_path(session)
    filing_html = None
    if filing_path:
        filing_html = load_filing_html(str(filing_path))

    # Header
    st.title(f"Extraction Review: {session.fund_name}")
    progress = session.reviewed_count / max(session.total_fields, 1)
    st.progress(progress)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Fields", session.total_fields)
    with col2:
        st.metric("Reviewed", session.reviewed_count)
    with col3:
        st.metric("Accepted", session.accepted_count)
    with col4:
        pct = int(100 * session.accepted_count / max(session.reviewed_count, 1))
        st.metric("Accept Rate", f"{pct}%")

    st.divider()

    # Sidebar
    with st.sidebar:
        # Source document link
        if filing_path:
            if st.button("Open in Browser", use_container_width=True):
                html_file = filing_path / "primary.html" if filing_path.is_dir() else filing_path
                webbrowser.open(html_file.resolve().as_uri())
        elif session.filing_path:
            st.caption(f"Source not found: {session.filing_path}")
        st.divider()

        st.header("Filters")
        categories = ["All"] + session.categories
        selected_cat = st.selectbox("Category", categories)

        st.header("Bulk Actions")
        if st.button("Accept all EXPLICIT+GROUNDED"):
            count = 0
            for review in session.fields.values():
                if not review.is_reviewed and review.grounded and review.extracted_value is not None:
                    review.decision = "accept"
                    review.reviewed_at = datetime.now().isoformat()
                    review.reviewer_notes = "Bulk accepted (grounded)"
                    count += 1
            if count > 0:
                st.success(f"Accepted {count} grounded fields")
                st.rerun()

        if st.button("Mark remaining nulls as N/A"):
            count = 0
            for review in session.fields.values():
                if not review.is_reviewed and review.extracted_value is None:
                    review.decision = "na"
                    review.reviewed_at = datetime.now().isoformat()
                    review.reviewer_notes = "Bulk marked N/A (null extraction)"
                    count += 1
            if count > 0:
                st.success(f"Marked {count} null fields as N/A")
                st.rerun()

        st.divider()

        st.header("Show")
        show_reviewed = st.checkbox("Show reviewed fields", value=True)
        show_unreviewed = st.checkbox("Show unreviewed fields", value=True)

    # Filter fields
    fields_to_show = []
    for review in session.fields.values():
        if selected_cat != "All" and review.category != selected_cat:
            continue
        if review.is_reviewed and not show_reviewed:
            continue
        if not review.is_reviewed and not show_unreviewed:
            continue
        fields_to_show.append(review)

    # Sort by canonical GT field order
    fields_to_show.sort(key=lambda f: get_field_sort_key(f.field_key))

    st.caption(f"Showing {len(fields_to_show)} of {session.total_fields} fields")

    # Render fields
    for idx, review in enumerate(fields_to_show):
        render_field(review, idx, filing_html)

    st.divider()

    # Footer actions
    footer_cols = st.columns(3)

    with footer_cols[0]:
        if st.button("Save Session", type="secondary", use_container_width=True):
            path = session.save()
            st.success(f"Session saved to {path}")

    with footer_cols[1]:
        unreviewed = session.total_fields - session.reviewed_count
        if unreviewed > 0:
            st.warning(f"{unreviewed} fields not yet reviewed")

    with footer_cols[2]:
        if st.button("Export as Ground Truth", type="primary", use_container_width=True):
            if session.reviewed_count == 0:
                st.error("No fields reviewed yet")
            else:
                gt_path = export_ground_truth(session)
                st.success(f"Ground truth exported to {gt_path}")


if __name__ == "__main__":
    main()

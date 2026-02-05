import os
import time
import pandas as pd
import numpy as np
import streamlit as st

# MUST be set before importing sentence_transformers
os.environ['TRANSFORMERS_NO_TF'] = '1'

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================================
# CORE FUNCTIONS (Your existing functions)
# ============================================================================

def compute_ssoc_embeddings(ssoc_reference_df, model_name='all-MiniLM-L6-v2', model=None):
    """
    Pre-compute embeddings for all SSOC categories.
    This should be done once and the results cached for all predictions.
    
    Args:
        ssoc_reference_df: DataFrame with 'Labelled SSOC' and 'Labelled SSOC Title' columns
        model_name: Sentence transformer model to use (default 'all-MiniLM-L6-v2')
        model: Pre-loaded model (optional)
    
    Returns:
        Dictionary containing:
            - 'embeddings': numpy array of category embeddings
            - 'categories': list of SSOC titles
            - 'codes': list of SSOC codes
            - 'model': the loaded model (for reuse)
    """
    start_time = time.time()
    
    # Load model if not provided
    if model is None:
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
    
    # Get unique SSOC categories
    unique_ssoc = ssoc_reference_df[['Labelled SSOC', 'Labelled SSOC Title']].drop_duplicates()
    ssoc_categories = unique_ssoc['Labelled SSOC Title'].tolist()
    ssoc_codes = unique_ssoc['Labelled SSOC'].tolist()
    
    # Encode SSOC categories
    print(f"Encoding {len(ssoc_categories)} SSOC categories...")
    category_embeddings = model.encode(
        ssoc_categories,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    end_time = time.time()
    print(f"SSOC embeddings computed in {end_time - start_time:.4f} seconds")
    
    return {
        'embeddings': category_embeddings,
        'categories': ssoc_categories,
        'codes': ssoc_codes,
        'model': model
    }


def match_single_job_title(job_title, ssoc_embeddings_dict, top_n=3):
    """
    Match a single job title to SSOC categories using pre-computed embeddings.
    
    Args:
        job_title: String containing the job title to match
        ssoc_embeddings_dict: Dictionary returned by compute_ssoc_embeddings()
                              Must contain: 'embeddings', 'categories', 'codes', 'model'
        top_n: Number of top matches to return (default 3)
    
    Returns:
        DataFrame with single row containing: Job Title, Choice 1, Choice 1 Score, Choice 2, ...
        Also returns execution time in seconds
    """
    start_time = time.time()
    
    # Extract from dictionary
    model = ssoc_embeddings_dict['model']
    category_embeddings = ssoc_embeddings_dict['embeddings']
    ssoc_categories = ssoc_embeddings_dict['categories']
    ssoc_codes = ssoc_embeddings_dict['codes']
    
    # Encode the single job title
    title_embedding = model.encode(
        [job_title],
        convert_to_numpy=True
    )
    
    # Compute similarity
    similarities = cosine_similarity(title_embedding, category_embeddings)[0]
    
    # Get top N matches
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    # Build result dictionary
    result = {'Job Title': job_title}
    
    for rank, idx in enumerate(top_indices, start=1):
        result[f'Choice {rank}'] = ssoc_categories[idx]
        result[f'Choice {rank} SSOC'] = ssoc_codes[idx]
        result[f'Choice {rank} Score'] = float(similarities[idx])
    
    # Create single-row DataFrame
    result_df = pd.DataFrame([result])
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return result_df, execution_time


# ============================================================================
# STREAMLIT UI HELPER FUNCTIONS
# ============================================================================

def create_ordered_ssoc_list(top_matches_df, all_categories):
    """
    Create ordered list of SSOC categories: Top 3 first, then rest alphabetically.

    Args:
        top_matches_df: DataFrame from match_single_job_title
        all_categories: List of all SSOC category titles

    Returns:
        List of category titles with top 3 first, then alphabetical
    """
    if top_matches_df is None or top_matches_df.empty:
        # If no matches yet, return all categories alphabetically
        return sorted(all_categories)

    # Extract top 3 matches
    top_3 = []
    for i in range(1, 4):
        choice_col = f'Choice {i}'
        if choice_col in top_matches_df.columns:
            category = top_matches_df[choice_col].iloc[0]
            top_3.append(category)

    # Get remaining categories (not in top 3) and sort alphabetically
    remaining = [cat for cat in all_categories if cat not in top_3]
    remaining_sorted = sorted(remaining)

    # Combine: top 3 first, then alphabetical rest
    return top_3 + remaining_sorted


# ============================================================================
# DATA LOADING & CACHING
# ============================================================================

@st.cache_resource
def load_embeddings(csv_path):
    """
    Load data and compute embeddings (cached for performance).
    This runs once when the app starts.
    """
    # Load your dataset
    df = pd.read_csv(csv_path)
    
    # Compute embeddings
    embeddings_dict = compute_ssoc_embeddings(df)
    
    # Get all unique categories for the dropdown
    unique_ssoc = df[['Labelled SSOC', 'Labelled SSOC Title']].drop_duplicates()
    all_categories = unique_ssoc['Labelled SSOC Title'].tolist()
    
    return embeddings_dict, all_categories


# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="SSOC Job Title Matcher",
        page_icon="💼",
        layout="centered"
    )

    st.title("💼 SSOC Job Title Matcher")

    st.write("")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #888;">
        <p style="color: #666; margin-bottom: 0;">
            ⚠️ Startup performance in this demo is not indicative of real-world performance. This demo runs on free
            community cloud servers and may need to be cold-started leading to higher latency.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    **Enter your job title and click away or hit "enter" to see the best matching SSOC Title.**
    """, unsafe_allow_html=True)

    # ========================================================================
    # DATA LOADING
    # ========================================================================

    CSV_FILE_PATH = "SSOC.csv"

    try:
        with st.spinner("Loading SSOC database..."):
            embeddings_dict, all_categories = load_embeddings(CSV_FILE_PATH)
    except FileNotFoundError:
        st.error(f"❌ Could not find file: {CSV_FILE_PATH}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

    # ========================================================================
    # SESSION STATE INITIALIZATION
    # ========================================================================

    if 'ordered_categories' not in st.session_state:
        st.session_state.ordered_categories = [""] + sorted(all_categories)
    if 'selected_ssoc_index' not in st.session_state:
        st.session_state.selected_ssoc_index = 0
    if 'last_job_title' not in st.session_state:
        st.session_state.last_job_title = ""
    if 'selectbox_key' not in st.session_state:
        st.session_state.selectbox_key = 0
    if 'top_3_matches' not in st.session_state:
        st.session_state.top_3_matches = []
    if 'execution_time' not in st.session_state:
        st.session_state.execution_time = None

    # ========================================================================
    # JOB TITLE INPUT
    # ========================================================================

    job_title = st.text_input(
        "Job Title",
        placeholder="e.g., Software Engineer, Marketing Manager, Data Analyst",
        key="job_title_input"
    )

    # ========================================================================
    # AUTO-MATCH ON INPUT CHANGE (with debouncing)
    # ========================================================================

    if job_title and job_title != st.session_state.last_job_title:
        start_time = time.time()

        # Debounce: wait 300ms
        # time.sleep(0.3)

        # Run matching
        matches_df, _ = match_single_job_title(job_title, embeddings_dict, top_n=3)

        # Extract top 3 matches
        top_3 = []
        for i in range(1, 4):
            choice_col = f'Choice {i}'
            if choice_col in matches_df.columns:
                top_3.append(matches_df[choice_col].iloc[0])

        # Update ordered list (top 3 first, then alphabetical)
        ordered_list = create_ordered_ssoc_list(matches_df, all_categories)
        st.session_state.ordered_categories = ordered_list
        st.session_state.top_3_matches = top_3
        st.session_state.selected_ssoc_index = 0  # Select first item (top match)
        st.session_state.last_job_title = job_title

        # Change the key to force selectbox to reset to new index
        st.session_state.selectbox_key += 1

        # Calculate and store execution time
        end_time = time.time()
        st.session_state.execution_time = end_time - start_time

        # Force rerun to update selectbox
        st.rerun()

    # Handle case where job title is cleared
    elif not job_title and st.session_state.last_job_title:
        st.session_state.ordered_categories = [""] + sorted(all_categories)
        st.session_state.top_3_matches = []
        st.session_state.execution_time = None
        st.session_state.selected_ssoc_index = 0
        st.session_state.last_job_title = ""
        st.session_state.selectbox_key += 1
        st.rerun()

    # ========================================================================
    # SSOC CATEGORY SELECTION
    # ========================================================================

    selected_ssoc = st.selectbox(
        "SSOC Title",
        options=st.session_state.ordered_categories,
        index=st.session_state.selected_ssoc_index,
        key=f"ssoc_selectbox_{st.session_state.selectbox_key}"
    )

    # ========================================================================
    # QUICK SELECTION BUTTONS (2nd and 3rd choice)
    # ========================================================================

    if len(st.session_state.top_3_matches) >= 2:
        st.caption("Are these more right?:")
        cols = st.columns(2)

        # 2nd choice button
        if len(st.session_state.top_3_matches) >= 2:
            with cols[0]:
                if st.button(f"📌 {st.session_state.top_3_matches[1]}", key="choice_2_btn", use_container_width=True):
                    # Update selection to 2nd choice (index 1)
                    st.session_state.selected_ssoc_index = 1
                    st.session_state.selectbox_key += 1
                    st.rerun()

        # 3rd choice button
        if len(st.session_state.top_3_matches) >= 3:
            with cols[1]:
                if st.button(f"📌 {st.session_state.top_3_matches[2]}", key="choice_3_btn", use_container_width=True):
                    # Update selection to 3rd choice (index 2)
                    st.session_state.selected_ssoc_index = 2
                    st.session_state.selectbox_key += 1
                    st.rerun()

    # Display execution time
    if st.session_state.execution_time is not None:
        exec_time_ms = st.session_state.execution_time * 1000  # Convert to milliseconds
        st.caption(f"Execution time: {exec_time_ms:.1f} ms")

    # ========================================================================
    # CONFIRM BUTTON
    # ========================================================================

    st.write("")  # Add some spacing
    if st.button("Confirm Selection", type="primary", use_container_width=True):
        if selected_ssoc and selected_ssoc != "":
            st.success(f"✓ Selected: {selected_ssoc}")
        else:
            st.warning("Please select an SSOC category")

    # ========================================================================
    # INFORMATION/NOTES BOX
    # ========================================================================

    st.write("")
    st.write("")

    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 4px solid #888;">
        <h4 style="margin-top: 0; color: #555;">❗Notes & Disclaimers</h4>
        <p style="color: #666; margin-bottom: 0;">
            Streamlit does not rerender the UI on-the-fly. That is why you only see the SSOC field update after you
            click away from the Job Title field or hit "enter". In a real deployment, the rerendering can be triggered
            after the user stops typing for a preset time even before clicking away. The latency number shown here only covers
            the time it took from the server receivng the job title input (after you click away or "enter") to you seeing the rendered result.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# RUN THE APP
# ============================================================================

if __name__ == "__main__":
    main()
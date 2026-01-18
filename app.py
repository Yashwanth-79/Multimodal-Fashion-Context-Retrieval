import streamlit as st
import pandas as pd
from pathlib import Path
from utils.config import config
from retriever.search_engine import FashionSearchEngine
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Fashion Retrieval System",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f0f2f6;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Load search engine once and cache it"""
    return FashionSearchEngine()

def main():
    
    st.image("utils/Glance-AI-Logo_Gradient-1.webp")
    st.markdown("<h2 style='text-align: center; color: blue;'>🛍️ glance AI Fashion Search</h2>", unsafe_allow_html=True)
    st.markdown("Search for fashion items using natural language (e.g., *'Jeans in the fashion show'*).")
    
    # Initialize engine
    with st.spinner("Loading & setting up... this may take a moment"):
        engine = load_engine()

    # Sidebar controls
    with st.sidebar:
        st.header("Search Settings")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=10)
        use_rerank = st.toggle("Use Attribute Re-ranking", value=True, 
                             help="Enable *Attribute-Aware Two-Stage Retrieval* re-ranking ")
        
        st.divider()
        st.markdown("### 📊 Metrics")
        st.metric("Total Images", len(engine.image_paths))
        
        if 'search_time' in st.session_state:
            st.metric("Last Search Time", f"{st.session_state.search_time:.4f}s")
        
        if 'last_query' in st.session_state:
            st.caption(f"Last query: '{st.session_state.last_query}'")

    # Main search interface
    query = st.text_input("Enter your search query:", placeholder="e.g., A person in formals in office")
    
    if st.button("Search") or query:
        if not query:
            st.warning("Please enter a query first.")
            return
            
        st.session_state.last_query = query
        
        # Perform search
        import time
        start_time = time.time()
        with st.spinner(f"Searching for '{query}'..."):
            # respect the toggle button directly
            explanation = engine.search_with_explanation(query, top_k=top_k, use_rerank=use_rerank)
        st.session_state.search_time = time.time() - start_time
        
        # Display Fusion Insights
        with st.expander("🔍 Query Analysis & Fusion Details", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Parsed Intent:**")
                st.json(explanation['parsed_query'])
            with col2:
                st.markdown("**Fusion Weights:**")
                st.bar_chart(explanation['fusion_weights'])
        
        st.divider()
        
        # Display Results in Grid
        results = explanation['results']
        cols = st.columns(3)  # 3 columns grid
        
        for idx, result in enumerate(results):
            with cols[idx % 3]:
                # Load image
                img_path = Path(result['image_path'])
                if not img_path.is_absolute():
                     # Fallback if path is relative
                     img_path = config.ROOT_DIR / img_path
                
                try:
                    image = Image.open(img_path)
                    st.image(image, use_container_width=True)
                except Exception as e:
                    st.error(f"Could not load image: {e}")
                
                # Expandable Info
                with st.expander(f"ℹ️ Info (Rank #{result['rank']})"):
                    st.markdown(f"**Score:** `{result['score']:.4f}`")
                    
                    meta = result['metadata']
                    
                    # Colors
                    st.markdown("**🎨 Colors:**")
                    st.write(", ".join(meta.get('colors', [])))
                    
                    # Garments
                    st.markdown("**👕 Garments:**")
                    for g in meta.get('garments', []):
                        st.write(f"- {g['garment']} ({g['confidence']:.2f})")
                        
                    # Scenes
                    st.markdown("**🏙️ Scene:**")
                    for s in meta.get('scenes', []):
                        st.write(f"- {s['scene']} ({s['confidence']:.2f})")
                        
                    st.caption(f"Filename: {img_path.name}")

if __name__ == "__main__":
    main()

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import requests
import json
import os
from sentence_transformers import CrossEncoder
import time
import tiktoken
from datetime import datetime

# ====================
# Pricing Database (per 1M tokens)
# ====================
PRICING = {
    # OpenAI Models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "chatgpt-4o-latest": {"input": 2.50, "output": 10.00},
    "gpt-4.1": {"input": 5.00, "output": 15.00},
    "gpt-4.1-mini": {"input": 0.30, "output": 1.20},

    # OpenRouter - Average estimates (varies by model)
    "default_openrouter": {"input": 1.00, "output": 3.00},
}

# ====================
# Helper Functions
# ====================
def estimate_tokens(text):
    """Estimate token count for text"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
    except:
        # Fallback: rough estimate (1 token ≈ 4 characters)
        return len(text) // 4

def calculate_cost(model_name, input_tokens, output_tokens):
    """Calculate cost for API call"""
    # Find pricing
    pricing = None

    # Check if it's a known model
    for key in PRICING:
        if key in model_name.lower():
            pricing = PRICING[key]
            break

    # Default to OpenRouter pricing
    if not pricing:
        pricing = PRICING["default_openrouter"]

    # Calculate cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost

def get_account_balance(provider, api_key):
    """Fetch remaining balance from API provider"""
    try:
        if provider == "OpenAI" and api_key:
            # OpenAI doesn't have a balance endpoint, return placeholder
            return {"balance": "N/A", "error": "OpenAI API doesn't provide balance endpoint"}

        elif provider == "OpenRouter" and api_key:
            response = requests.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                # OpenRouter returns credit info
                limit = data.get('data', {}).get('limit', 0)
                usage = data.get('data', {}).get('usage', 0)
                remaining = limit - usage
                return {
                    "balance": f"${remaining:.2f}",
                    "limit": f"${limit:.2f}",
                    "used": f"${usage:.2f}"
                }
    except Exception as e:
        return {"balance": "Error", "error": str(e)}

    return {"balance": "N/A"}

# ====================
# Page Configuration
# ====================
st.set_page_config(
    page_title="Spurgeon Sermon Q&A",
    page_icon="📖",
    layout="wide",  # Wide layout for stats panel
    initial_sidebar_state="auto"
)

# ====================
# Mobile-First Responsive CSS
# ====================
st.markdown("""
<style>
    /* Mobile-first base styles */
    .main .block-container {
        padding: 1rem 1rem;
        max-width: 100%;
    }

    /* Touch-friendly input */
    .stChatInput {
        font-size: 16px !important; /* Prevents zoom on iOS */
    }

    /* Responsive chat messages */
    .stChatMessage {
        padding: 0.75rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Mobile-optimized buttons */
    .stButton button {
        width: 100%;
        min-height: 44px; /* Apple's minimum touch target */
        font-size: 16px;
        padding: 0.75rem 1rem;
    }

    /* Sidebar optimization - Wider for better filter visibility */
    section[data-testid="stSidebar"] {
        width: 320px !important;
    }

    /* Responsive sliders */
    .stSlider {
        padding: 0.5rem 0;
    }

    /* Source expander touch-friendly */
    .streamlit-expanderHeader {
        font-size: 16px;
        padding: 0.75rem;
        min-height: 44px;
    }

    /* Responsive text inputs */
    .stTextInput input {
        font-size: 16px !important;
        min-height: 44px;
    }

    /* Tablet and up (768px+) */
    @media (min-width: 768px) {
        .main .block-container {
            padding: 2rem 2rem;
            max-width: 900px;
        }

        section[data-testid="stSidebar"] {
            width: 380px !important;
        }

        .stButton button {
            width: auto;
        }
    }

    /* Desktop (1024px+) */
    @media (min-width: 1024px) {
        .main .block-container {
            padding: 3rem 3rem;
            max-width: 1100px;
        }

        section[data-testid="stSidebar"] {
            width: 420px !important;
        }
    }

    /* Prevent horizontal scroll on mobile */
    .main {
        overflow-x: hidden;
    }

    /* Optimize source display for mobile */
    .element-container {
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    /* Hide scrollbar on mobile for cleaner look */
    @media (max-width: 767px) {
        ::-webkit-scrollbar {
            width: 3px;
        }
    }
</style>
""", unsafe_allow_html=True)

# ====================
# Initialize Session State
# ====================
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None

if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None

if 'openrouter_all_models' not in st.session_state:
    st.session_state.openrouter_all_models = []

if 'show_all_models' not in st.session_state:
    st.session_state.show_all_models = False

if 'reranker' not in st.session_state:
    st.session_state.reranker = None

if 'usage_stats' not in st.session_state:
    st.session_state.usage_stats = []  # List of usage records

if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0

# ====================
# Responsive Sidebar Configuration
# ====================
with st.sidebar:
    st.title("⚙️ Settings")

    # Mobile-friendly collapsible sections
    with st.expander("🤖 Model Provider", expanded=False):
        provider = st.radio(
            "Provider",
            options=["LM Studio (Local)", "OpenAI", "OpenRouter"],
            index=0,
            help="Choose your AI model provider"
        )

        if provider == "LM Studio (Local)":
            lm_studio_url = st.text_input(
                "LM Studio URL",
                value="http://192.168.1.182:1234/v1/chat/completions",
                help="URL endpoint for LM Studio API"
            )

            # Fetch available models from LM Studio
            available_models = []
            try:
                base_url = lm_studio_url.replace('/v1/chat/completions', '')
                models_response = requests.get(f"{base_url}/v1/models", timeout=3)
                if models_response.status_code == 200:
                    models_data = models_response.json()
                    available_models = [model['id'] for model in models_data.get('data', [])]
                    # Filter out embedding models
                    available_models = [m for m in available_models if not m.startswith('text-embedding')]
            except:
                pass

            if available_models:
                # Default to google/gemma-3-4b if available, otherwise first model
                default_index = 0
                if "google/gemma-3-4b" in available_models:
                    default_index = available_models.index("google/gemma-3-4b")

                model_name = st.selectbox(
                    "Model",
                    options=available_models,
                    index=default_index,
                    help="Select the loaded model from LM Studio"
                )
            else:
                model_name = st.text_input(
                    "Model Name",
                    value="google/gemma-3-4b",
                    help="Name of the loaded model"
                )

            api_key = None

        elif provider == "OpenAI":
            lm_studio_url = "https://api.openai.com/v1/chat/completions"

            api_key = st.text_input(
                "OpenAI API Key",
                value="",
                type="password",
                help="Your OpenAI API key"
            )

            # Popular OpenAI chat models
            openai_models = [
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo",
                "chatgpt-4o-latest",
                "gpt-4.1",
                "gpt-4.1-mini"
            ]

            model_name = st.selectbox(
                "Model",
                options=openai_models,
                index=0,
                help="Select OpenAI model"
            )

        else:  # OpenRouter
            lm_studio_url = "https://openrouter.ai/api/v1/chat/completions"

            api_key = st.text_input(
                "OpenRouter API Key",
                value="",
                type="password",
                help="Your OpenRouter API key"
            )

            # Fetch all models from OpenRouter (once)
            if not st.session_state.openrouter_all_models:
                try:
                    models_response = requests.get(
                        "https://openrouter.ai/api/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=5
                    )
                    if models_response.status_code == 200:
                        models_data = models_response.json()
                        st.session_state.openrouter_all_models = [model['id'] for model in models_data.get('data', [])]
                except:
                    pass

            all_models = st.session_state.openrouter_all_models

            # Helper functions
            def get_provider(model_id):
                """Extract provider from model ID"""
                if '/' in model_id:
                    return model_id.split('/', 1)[0]
                return 'Unknown'

            def clean_model_name(model_id):
                """Remove provider prefix from model names"""
                if '/' in model_id:
                    return model_id.split('/', 1)[1]
                return model_id

            def get_model_family(model_id):
                """Categorize models by family"""
                model_lower = model_id.lower()
                if 'llama' in model_lower:
                    return 'Llama'
                elif 'qwen' in model_lower:
                    return 'Qwen'
                elif 'gemma' in model_lower:
                    return 'Gemma'
                elif 'deepseek' in model_lower:
                    return 'DeepSeek'
                elif 'grok' in model_lower or 'x-ai' in model_lower:
                    return 'Grok'
                elif 'claude' in model_lower or 'anthropic' in model_lower:
                    return 'Claude'
                elif 'gpt' in model_lower or 'openai' in model_lower:
                    return 'GPT'
                elif 'gemini' in model_lower or 'google' in model_lower:
                    return 'Gemini'
                elif 'mistral' in model_lower:
                    return 'Mistral'
                else:
                    return 'Other'

            if all_models:
                # Filter controls
                st.markdown("**🔍 Filter & Sort**")

                # Three-column filter layout
                col1, col2, col3 = st.columns([2, 2, 1])

                with col1:
                    # Family filter
                    families = sorted(set([get_model_family(m) for m in all_models]))
                    selected_families = st.multiselect(
                        "Model Families",
                        options=families,
                        default=['Llama', 'Qwen', 'Gemma', 'DeepSeek', 'Grok'],
                        help="Filter by model family (architecture)"
                    )

                with col2:
                    # Provider filter
                    providers = sorted(set([get_provider(m) for m in all_models]))
                    selected_providers = st.multiselect(
                        "Providers",
                        options=providers,
                        default=[],  # Empty = show all
                        help="Filter by provider (optional)"
                    )

                with col3:
                    # Sort options
                    sort_by = st.selectbox(
                        "Sort By",
                        options=['Family', 'Name', 'Size', 'Provider'],
                        index=0,
                        help="How to sort the list"
                    )

                # Search box
                search_term = st.text_input(
                    "🔎 Search models",
                    placeholder="Type to filter (e.g., '70b', 'instruct', 'free')...",
                    help="Filter models by keyword"
                )

                # Filter models by family, provider, and search
                filtered_models = []
                for model in all_models:
                    family = get_model_family(model)
                    provider = get_provider(model)

                    # Family filter
                    if selected_families and family not in selected_families:
                        continue

                    # Provider filter (if any selected)
                    if selected_providers and provider not in selected_providers:
                        continue

                    # Search filter
                    if search_term and search_term.lower() not in model.lower():
                        continue

                    filtered_models.append(model)

                # Sort models
                def extract_size(model_id):
                    """Extract model size for sorting"""
                    import re
                    match = re.search(r'(\d+)b', model_id.lower())
                    return int(match.group(1)) if match else 0

                if sort_by == 'Family':
                    filtered_models.sort(key=lambda m: (get_model_family(m), m))
                elif sort_by == 'Name':
                    filtered_models.sort(key=lambda m: clean_model_name(m))
                elif sort_by == 'Size':
                    filtered_models.sort(key=lambda m: extract_size(m), reverse=True)
                elif sort_by == 'Provider':
                    filtered_models.sort(key=lambda m: (get_provider(m), clean_model_name(m)))

                # Create display names - CLEAN names only (no provider prefix)
                model_display_map = {}
                for model in filtered_models:
                    clean_name = clean_model_name(model)
                    # Just show the clean name without any prefix
                    display_name = clean_name
                    model_display_map[display_name] = model

                display_names = list(model_display_map.keys())

                # Larger selectbox with custom CSS
                st.markdown("""
                    <style>
                    div[data-baseweb="select"] > div {
                        min-height: 50px;
                        font-size: 14px;
                        font-family: 'Courier New', monospace;
                    }
                    div[data-baseweb="select"] select {
                        font-size: 14px;
                    }
                    </style>
                """, unsafe_allow_html=True)

                if display_names:
                    selected_display_name = st.selectbox(
                        f"Model ({len(display_names)} available)",
                        options=display_names,
                        index=0,
                        help="Select model from OpenRouter",
                        key="openrouter_model_select"
                    )

                    # Get the actual model ID from the display name
                    model_name = model_display_map[selected_display_name]

                    # Show detailed model info
                    provider = get_provider(model_name)
                    family = get_model_family(model_name)

                    st.markdown(
                        f'<div style="background: #262730; padding: 10px 14px; border-radius: 8px; '
                        f'margin-top: 8px; font-size: 13px; color: #fafafa; border: 1px solid #444;">'
                        f'<b style="color: #4ade80;">✓ Selected:</b> <span style="color: #fff;">{selected_display_name}</span><br>'
                        f'<b style="color: #a8a8a8;">Provider:</b> <span style="color: #60a5fa;">{provider}</span> | '
                        f'<b style="color: #a8a8a8;">Family:</b> <span style="color: #f472b6;">{family}</span><br>'
                        f'<b style="color: #a8a8a8;">Full ID:</b> <code style="background: #1a1a1a; padding: 2px 6px; '
                        f'border-radius: 4px; color: #fbbf24; font-size: 11px;">{model_name}</code>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("No models match your filters. Try adjusting the filters.")
                    model_name = all_models[0] if all_models else "deepseek/deepseek-v3.2-exp"

            else:
                # Fallback if API fetch failed
                st.warning("Could not fetch models from OpenRouter. Using fallback list.")
                fallback_models = [
                    "deepseek/deepseek-v3.2-exp",
                    "qwen/qwen3-max",
                    "x-ai/grok-4-fast",
                    "google/gemma-2-27b-it",
                    "meta-llama/llama-3.3-70b-instruct",
                    "meta-llama/llama-4-maverick"
                ]
                model_name = st.selectbox(
                    "Model",
                    options=fallback_models,
                    index=0,
                    help="Select model from OpenRouter"
                )
        temperature = st.slider(
            "Temperature",
            0.0, 1.0, 0.7, 0.1,
            help="Controls randomness: 0=focused, 1=creative"
        )
        max_tokens = st.slider(
            "Max Tokens",
            100, 4000, 1500, 100,
            help="Maximum response length"
        )

    with st.expander("🔍 RAG Settings", expanded=False):
        # Reranking toggle
        use_reranking = st.checkbox(
            "Enable Reranking",
            value=True,
            help="Use 2-stage retrieval: retrieve more chunks, then rerank to find the most relevant"
        )

        if use_reranking:
            col1, col2 = st.columns(2)
            with col1:
                initial_k = st.slider(
                    "Initial Retrieval",
                    5, 100, 15, 5,
                    help="Number of chunks to retrieve before reranking"
                )
            with col2:
                num_chunks = st.slider(
                    "Final Chunks",
                    1, 20, 5, 1,
                    help="Number of top chunks after reranking"
                )

            st.caption(f"📊 Strategy: Retrieve {initial_k} → Rerank → Select top {num_chunks}")
        else:
            num_chunks = st.slider(
                "Context Chunks",
                1, 50, 4, 1,
                help="Number of sermon excerpts to retrieve"
            )
            initial_k = None

    st.divider()

    # Connection Status - more compact for mobile
    with st.expander("📡 Status", expanded=True):
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Test Connection", use_container_width=True):
                try:
                    test_url = lm_studio_url.replace('/chat/completions', '/models')
                    headers = {}
                    if api_key:  # OpenAI
                        headers["Authorization"] = f"Bearer {api_key}"

                    response = requests.get(test_url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        if "openrouter.ai" in test_url:
                            provider_name = "OpenRouter"
                        elif "openai.com" in test_url:
                            provider_name = "OpenAI"
                        else:
                            provider_name = "LM Studio"
                        st.success(f"✅ {provider_name} OK")
                    else:
                        st.error("❌ Connection failed")
                except Exception as e:
                    st.error("❌ No connection")

        with col2:
            # Vector DB Status (always visible)
            if os.path.exists("./vector_db"):
                st.success("✅ DB Ready")
            else:
                st.error("❌ No DB")

# ====================
# Load Vector Database
# ====================
@st.cache_resource(show_spinner="Loading embedding model (all-MiniLM-L6-v2)...")
def load_vectorstore():
    """Load the vector database and embeddings"""
    try:
        # Load embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Load vector database
        vectorstore = Chroma(
            persist_directory="./vector_db",
            embedding_function=embeddings
        )

        return vectorstore, embeddings
    except Exception as e:
        st.error(f"Error loading vector database: {str(e)}")
        return None, None

# ====================
# Load Reranker Model
# ====================
@st.cache_resource(show_spinner="Loading reranker model (ms-marco-MiniLM)...")
def load_reranker():
    """Load the cross-encoder reranker model"""
    try:
        print("Loading cross-encoder reranker: cross-encoder/ms-marco-MiniLM-L-6-v2")
        reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✓ Reranker loaded successfully")
        return reranker
    except Exception as e:
        print(f"⚠️ Error loading reranker: {str(e)}")
        st.warning(f"Reranker not available: {str(e)}")
        return None

# ====================
# Query Function
# ====================
def query_spurgeon(question, vectorstore, num_chunks, api_url, model_name, temperature, max_tokens, api_key=None, use_reranking=True, initial_k=None):
    """Query the RAG system with optional 2-stage retrieval (retrieve + rerank)"""
    try:
        # 1. STAGE 1: Initial Retrieval
        # Retrieve more chunks if reranking is enabled (default: 3x final amount)
        if use_reranking and initial_k:
            retrieve_k = initial_k
        elif use_reranking:
            retrieve_k = num_chunks * 3  # Retrieve 3x more for reranking
        else:
            retrieve_k = num_chunks

        retrieval_start = time.time()
        with st.spinner(f"🔍 Searching sermon database... (retrieving {retrieve_k} candidates)"):
            candidate_docs = vectorstore.similarity_search(question, k=retrieve_k)
        retrieval_time = time.time() - retrieval_start

        print("\n" + "="*80)
        print(f"STAGE 1 - INITIAL RETRIEVAL: {len(candidate_docs)} candidates in {retrieval_time:.2f}s")
        print("="*80)

        # 2. STAGE 2: Reranking (if enabled)
        if use_reranking and st.session_state.reranker is not None and len(candidate_docs) > num_chunks:
            rerank_start = time.time()
            with st.spinner(f"🎯 Reranking {len(candidate_docs)} chunks to top {num_chunks}..."):
                # Create query-document pairs
                pairs = [[question, doc.page_content] for doc in candidate_docs]

                # Get reranking scores
                scores = st.session_state.reranker.predict(pairs)

                # Sort by scores (descending) and keep top num_chunks
                ranked_indices = scores.argsort()[::-1][:num_chunks]
                relevant_docs = [candidate_docs[i] for i in ranked_indices]

            rerank_time = time.time() - rerank_start

            print(f"\nSTAGE 2 - RERANKING: Top {len(relevant_docs)} selected in {rerank_time:.2f}s")
            print(f"Score range: {scores[ranked_indices[0]]:.4f} (best) to {scores[ranked_indices[-1]]:.4f} (worst)")
            print(f"Total retrieval time: {retrieval_time + rerank_time:.2f}s")
        else:
            relevant_docs = candidate_docs[:num_chunks]
            print(f"\nRERANKING: Disabled - using top {num_chunks} from initial retrieval")

        # 3. Log final selected chunks
        print("\n" + "="*80)
        print(f"FINAL CHUNKS FOR QUESTION: {question}")
        print("="*80)
        for i, doc in enumerate(relevant_docs, 1):
            print(f"\nCHUNK {i}:")
            print(f"  Metadata: {doc.metadata}")
            print(f"  Content Preview: {doc.page_content[:200]}...")
        print("="*80 + "\n")

        # 4. Build context with metadata for each chunk and create citation map
        context_parts = []
        citation_map = []

        for i, doc in enumerate(relevant_docs, 1):
            meta = doc.metadata

            # Build full citation string
            citation_parts = []
            title = meta.get('title', 'Unknown Title')
            sermon_id = meta.get('id', '')
            date = meta.get('date', '')
            page = meta.get('page', '')
            source_file = meta.get('source', '').split('/')[-1] if 'source' in meta else ''

            if sermon_id:
                citation_parts.append(f"Sermon #{sermon_id}")
            if date:
                citation_parts.append(str(date))
            if title and title != 'Unknown Title':
                citation_parts.append(f'"{title}"')
            if page:
                citation_parts.append(f"p. {page}")

            full_citation = ", ".join(citation_parts) if citation_parts else source_file or "Unknown Source"
            citation_map.append(full_citation)

            # Format metadata for the prompt
            citation_info = f"[Source {i} - {full_citation}]:"
            context_parts.append(f"{citation_info}\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        # Build citation reference guide
        citation_guide = "\n".join([f"Source {i}: {cit}" for i, cit in enumerate(citation_map, 1)])

        # 5. Create polished prompt with system message and RAG instructions
        system_message = """You are an expert assistant generating clear, structured, and polished responses for any topic.

Guidelines:
1. **Structure**: Introduction → Analysis/Key Points → Evidence/Examples → Conclusion.
2. **Evidence**: Reference sources or quotes when possible; flag uncertainty; avoid fabricating exact page numbers.
3. **Style**: Use smooth transitions, varied sentence structures, and professional, readable language.
4. **Reasoning**: Ensure logical consistency; distinguish fact, interpretation, and opinion.
5. **Adaptability**: Treat each query independently; works for historical, literary, theological, scientific, or philosophical topics.
6. **Special Condition for Charles Spurgeon**: When answering questions about Charles Spurgeon, use his Autobiography (Vols. 1–4) for historical context and his Sermons (e.g., Sermon #985 "The Weary Resting Place") for theological points. Treat these as authoritative sources for quotes, citations, and theological analysis. Do not invent titles or page numbers—if unsure, summarize faithfully without fabricating details.

Always keep responses clear, structured, and polished while minimizing token use."""

        rag_instructions = f"""Answer the question using the sources below with FULL inline citations.

CRITICAL: You MUST cite using the COMPLETE citation information provided, not just source numbers.

CITATION FORMAT: After any quote or paraphrase, immediately add the full citation in parentheses.
Example: "Faith grasps Christ's righteousness (Sermon #985, 1865, "The Weary Resting Place", p. 214)."

AVAILABLE SOURCES:
{citation_guide}

WHEN YOU CITE:
- Use the EXACT citation text shown above for each source
- Place it in parentheses immediately after the quote or paraphrase
- Do NOT just write "(Source #1)" - write the FULL citation
- Example: (Sermon #123, 1870, "The Blood of Christ", p. 45)

SOURCES WITH CONTENT:
{context}

QUESTION: {question}

Write a clear academic answer with FULL inline citations showing title, sermon number, date, and page."""

        # 6. Send to API (LM Studio, OpenAI, or OpenRouter)
        with st.spinner("🤖 Generating answer..."):
            headers = {"Content-Type": "application/json"}

            # Determine provider and set appropriate headers
            provider_name = "LM Studio"
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
                if "openrouter.ai" in api_url:
                    provider_name = "OpenRouter"
                    # OpenRouter requires these headers
                    headers["HTTP-Referer"] = "http://localhost:8501"  # Streamlit default
                    headers["X-Title"] = "Spurgeon RAG App"
                elif "openai.com" in api_url:
                    provider_name = "OpenAI"

            response = requests.post(
                api_url,
                headers=headers,
                json={
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": rag_instructions}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                timeout=120
            )

            if response.status_code != 200:
                error_detail = response.text if response.text else "Unknown error"
                print(f"{provider_name} Error: Status {response.status_code}, Details: {error_detail}")

                # Add helpful message for OpenRouter data policy errors
                error_msg = f"Error: {provider_name} returned status code {response.status_code}."
                if provider_name == "OpenRouter" and "data policy" in error_detail.lower():
                    error_msg += " Check your OpenRouter privacy settings at https://openrouter.ai/settings/privacy"
                else:
                    error_msg += f" Details: {error_detail}"

                return error_msg, [], None

            result = response.json()
            answer = result['choices'][0]['message']['content']

            # Extract token usage from API response
            usage = result.get('usage', {})

            # Get token counts (all providers return these)
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            total_tokens = usage.get('total_tokens', input_tokens + output_tokens)

            # Get cost from provider
            cost = 0.0

            if 'openrouter.ai' in api_url:
                # OpenRouter: Fetch EXACT cost from generation endpoint
                generation_id = result.get('id')
                print(f"\n🔍 OpenRouter Generation ID: {generation_id}")

                if generation_id and api_key:
                    try:
                        # Wait longer for generation to be indexed (2 seconds is more reliable)
                        print("⏳ Waiting 2 seconds for OpenRouter to index generation...")
                        time.sleep(2)

                        gen_response = requests.get(
                            f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=10
                        )

                        print(f"📡 Generation endpoint status: {gen_response.status_code}")

                        if gen_response.status_code == 200:
                            gen_data = gen_response.json().get('data', {})
                            cost = gen_data.get('total_cost', 0.0)  # Exact cost from OpenRouter

                            print(f"📊 Raw cost value from API: {cost}")
                            print(f"📊 Cost in scientific notation: {cost:.10e}")

                            # Also get native token counts for better accuracy
                            native_input = gen_data.get('native_tokens_prompt', input_tokens)
                            native_output = gen_data.get('native_tokens_completion', output_tokens)

                            print(f"✅ Fetched exact cost from OpenRouter: ${cost:.8f}")
                            print(f"🔢 Native tokens: {native_input} input + {native_output} output")
                        else:
                            print(f"⚠️ Generation endpoint returned {gen_response.status_code}")
                            print(f"Response: {gen_response.text[:200]}")
                            # Fallback to calculation
                            cost = calculate_cost(model_name, input_tokens, output_tokens)
                            print(f"💡 Using calculated cost instead: ${cost:.8f}")

                    except Exception as e:
                        print(f"❌ Could not fetch cost from generation endpoint: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to calculation
                        cost = calculate_cost(model_name, input_tokens, output_tokens)
                        print(f"💡 Using calculated cost instead: ${cost:.8f}")
                else:
                    # No generation ID or API key, calculate manually
                    print("⚠️ No generation ID or API key - calculating cost manually")
                    cost = calculate_cost(model_name, input_tokens, output_tokens)

            elif 'openai.com' in api_url:
                # OpenAI doesn't provide cost, we calculate it
                cost = calculate_cost(model_name, input_tokens, output_tokens)

            else:
                # LM Studio (local) - always free
                cost = 0.0

            # Log cost details
            print(f"\n💰 COST TRACKING:")
            print(f"  Model: {model_name}")
            print(f"  Provider: {provider_name}")
            print(f"  Input tokens: {input_tokens:,}")
            print(f"  Output tokens: {output_tokens:,}")
            print(f"  Total tokens: {total_tokens:,}")
            print(f"  Cost: ${cost:.6f}")
            print()

            # Record usage statistics
            usage_record = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'model': model_name,
                'provider': provider_name,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': total_tokens,
                'cost': cost,  # Direct from API (OpenRouter) or calculated (OpenAI)
                'question': question[:100]  # First 100 chars
            }

            return answer, relevant_docs, usage_record

    except requests.exceptions.Timeout:
        return "Error: Request timed out. Please check if the model is loaded and try again.", [], None
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to API. Please check your connection settings.", [], None
    except Exception as e:
        print(f"Exception in query_spurgeon: {type(e).__name__}: {str(e)}")
        return f"Error: {type(e).__name__}: {str(e)}", [], None

# ====================
# Main UI - Two Column Layout
# ====================
# Create two columns: Main chat (70%) and Stats panel (30%)
main_col, stats_col = st.columns([7, 3])

with main_col:
    # Responsive title
    st.markdown("""
        <h1 style='font-size: clamp(1.5rem, 5vw, 2.5rem); margin-bottom: 0.5rem;'>
            📖 Spurgeon Sermon Q&A
        </h1>
    """, unsafe_allow_html=True)

    # Compact description
    st.caption("Ask questions about Charles Spurgeon's sermons")

    # Load vector store on startup if it exists
    if st.session_state.vectorstore is None:
        if os.path.exists("./vector_db"):
            with st.spinner("Loading database..."):
                vectorstore, embeddings = load_vectorstore()
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.embeddings = embeddings
                    st.success("✅ Database Ready!", icon="✅")
        else:
            st.error("⚠️ Run setup_rag.py first", icon="⚠️")

    # Load reranker model on startup (only once)
    if st.session_state.reranker is None:
        with st.spinner("Loading reranker..."):
            st.session_state.reranker = load_reranker()
            if st.session_state.reranker:
                st.success("✅ Reranker Ready!", icon="🎯")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show model info for assistant responses
            if message["role"] == "assistant" and "model" in message:
                provider = message.get("provider", "Unknown")
                model = message.get("model", "Unknown")

                # Get cost and token info if available
                cost_info = ""
                if "cost" in message and message["cost"] > 0:
                    cost_info = f' · ${message["cost"]:.4f}'
                if "tokens" in message:
                    cost_info += f' · {message["tokens"]:,} tokens'

                # Create a compact model badge
                st.markdown(
                    f'<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); '
                    f'padding: 4px 12px; border-radius: 12px; margin-bottom: 8px; '
                    f'display: inline-block; font-size: 12px; color: white; font-weight: 500;">'
                    f'🤖 {provider} · {model}{cost_info}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask about Spurgeon's sermons..."):
        # Check if vectorstore is loaded
        if st.session_state.vectorstore is None:
            st.error("⚠️ Database not loaded. Run setup_rag.py first.", icon="⚠️")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get response
            answer, sources, usage_record = query_spurgeon(
                prompt,
                st.session_state.vectorstore,
                num_chunks,
                lm_studio_url,  # API URL (works for all providers)
                model_name,
                temperature,
                max_tokens,
                api_key,
                use_reranking,
                initial_k
            )

            # Determine provider for display
            display_provider = "LM Studio"
            if api_key:
                if "openrouter.ai" in lm_studio_url:
                    display_provider = "OpenRouter"
                elif "openai.com" in lm_studio_url:
                    display_provider = "OpenAI"

            # Record usage statistics
            if usage_record:
                st.session_state.usage_stats.append(usage_record)
                st.session_state.total_cost += usage_record['cost']

            # Prepare message metadata
            message_data = {
                "role": "assistant",
                "content": answer,
                "model": model_name,
                "provider": display_provider
            }

            # Add cost and token info if available
            if usage_record:
                message_data["cost"] = usage_record['cost']
                message_data["tokens"] = usage_record['input_tokens'] + usage_record['output_tokens']

            # Add assistant message with model info
            st.session_state.messages.append(message_data)

            with st.chat_message("assistant"):
                # Show model badge with cost info
                cost_display = ""
                if usage_record:
                    cost_display = f' · ${usage_record["cost"]:.4f} · {usage_record["input_tokens"] + usage_record["output_tokens"]:,} tokens'

                st.markdown(
                    f'<div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); '
                    f'padding: 4px 12px; border-radius: 12px; margin-bottom: 8px; '
                    f'display: inline-block; font-size: 12px; color: white; font-weight: 500;">'
                    f'🤖 {display_provider} · {model_name}{cost_display}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(answer)

    # Clear chat button
    if st.session_state.messages:
        if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ====================
# Stats Panel (Right Column)
# ====================
with stats_col:
    st.markdown("### 📊 Usage Stats")

    # Account balance section
    st.markdown("#### 💳 Account Balance")

    # Determine current provider
    current_provider = "LM Studio"
    if api_key:
        if "openrouter.ai" in lm_studio_url:
            current_provider = "OpenRouter"
        elif "openai.com" in lm_studio_url:
            current_provider = "OpenAI"

    if st.button("🔄 Refresh Balance", use_container_width=True):
        if current_provider != "LM Studio":
            balance_info = get_account_balance(current_provider, api_key)
            if "error" not in balance_info:
                if current_provider == "OpenRouter":
                    st.success(f"**Limit**: {balance_info.get('limit', 'N/A')}")
                    st.info(f"**Used**: {balance_info.get('used', 'N/A')}")
                    st.success(f"**Remaining**: {balance_info.get('balance', 'N/A')}")
                else:
                    st.info(f"**Balance**: {balance_info.get('balance', 'N/A')}")
            else:
                st.warning(balance_info['error'])
        else:
            st.info("Local LM Studio (Free)")

    st.divider()

    # Session statistics
    st.markdown("#### 📈 Session Stats")

    # Group usage by model
    model_usage = {}
    for record in st.session_state.usage_stats:
        model = record['model']
        if model not in model_usage:
            model_usage[model] = {
                'count': 0,
                'total_cost': 0.0,
                'input_tokens': 0,
                'output_tokens': 0
            }
        model_usage[model]['count'] += 1
        model_usage[model]['total_cost'] += record['cost']
        model_usage[model]['input_tokens'] += record['input_tokens']
        model_usage[model]['output_tokens'] += record['output_tokens']

    if model_usage:
        # Display per-model stats
        for model, stats in model_usage.items():
            with st.expander(f"**{model}**", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Uses", stats['count'])
                    st.metric("Input Tokens", f"{stats['input_tokens']:,}")
                with col2:
                    st.metric("Cost", f"${stats['total_cost']:.4f}")
                    st.metric("Output Tokens", f"{stats['output_tokens']:,}")

        st.divider()

        # Total statistics
        st.markdown("#### 💰 Total")
        total_queries = len(st.session_state.usage_stats)
        total_cost = st.session_state.total_cost

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Total Cost", f"${total_cost:.4f}")

        # Clear stats button
        if st.button("🗑️ Clear Stats", use_container_width=True):
            st.session_state.usage_stats = []
            st.session_state.total_cost = 0.0
            st.rerun()
    else:
        st.info("No usage data yet. Start chatting to see stats!")

    st.divider()

    # Model info
    st.markdown("#### ⚙️ Current Config")
    st.caption(f"**Provider**: {current_provider}")
    st.caption(f"**Model**: {model_name}")
    st.caption(f"**Max Tokens**: {max_tokens}")
    st.caption(f"**Temperature**: {temperature}")

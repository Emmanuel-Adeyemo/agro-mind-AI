import streamlit as st
import os
import pickle
from pathlib import Path
import tempfile
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import EnsembleRetriever, BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# UI layout tuning
st.set_page_config(
    page_title='QGen AI - Quantitative Genetics Terminal',
    layout='wide',
    initial_sidebar_state='expanded'
)

# custom styling for the right-hand evidence column with a fixed height and scrollbar
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #f8f9fa; }
    .evidence-container {
        max-height: 65vh;
        overflow-y: auto;
        padding: 12px;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .parameter-box {
        background-color: #e9ecef;
        padding: 10px;
        border-radius: 6px;
        margin-bottom: 15px;
    }
    
    /* Floating GitHub Link Styling */
    .github-corner-link {
        position: fixed;
        top: 60px; /* Aligns clean with Streamlit's default top header bar */
        right: 20px;
        z-index: 999999;
        color: #24292e;
        transition: color 0.2s ease-in-out;
    }
    .github-corner-link:hover {
        color: #0366d6; /* Turns professional blue on hover */
    }
    </style>
    
    <a class="github-corner-link" href="https://github.com/Emmanuel-Adeyemo/agro-mind-AI" target="_blank" title="View Source on GitHub">
        <svg height="24" aria-hidden="true" viewBox="0 0 16 16" version="1.1" width="24" data-view-component="true" fill="currentColor">
            <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82a7.42 7.42 0 0 0-4 0c-1.53-1.04-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.16.73.72.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
        </svg>
    </a>
""", unsafe_allow_html=True)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY") or st.secrets.get("PINECONE_API_KEY")


# load and cache pre-baked search indices
@st.cache_resource
def load_core_retrievers():
    ROOT = Path(__file__).resolve().parent
    CHROMADB_DIR = ROOT / 'chromadb'  # Point to your correct folder name
    BM25_PATH = ROOT / 'bm25_index.pkl'

    # load vectors in chromadb
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large', api_key=openai_api_key)

    vector_store = PineconeVectorStore(
        index_name="qgen-ai-index",
        embedding=embeddings,
        pinecone_api_key=pinecone_api_key
    )

    if not BM25_PATH.exists():
        st.error(f"BM25 index file not found! Checked: {BM25_PATH}")
        st.stop()

    # load keyword BM25 matrix
    with open(BM25_PATH, 'rb') as f:
        bm25_retriever = pickle.load(f)

    return vector_store, bm25_retriever


@st.cache_data(show_spinner=False)
def process_new_pdfs(file_bytes, file_name):
    # write uploaded pdf to a temporary disk file for the PyPDFLoader to read
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(file_bytes)
        tmp_file_path = tmp_file.name

    try:
        loader = PyPDFLoader(str(tmp_file_path))
        pages = loader.load()

        cover_page = pages[0].page_content[:3000] if pages else ''

        get_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert digital archivist for an agricultural genomics library.\n"
                "Analyze the provided text from the cover page of a scientific paper. "
                "Extract the primary authors, the year of publication, and the scientific journal.\n\n"
                "CRITICAL OUTPUT FORMATTING:\n"
                "- If there are more than two authors, format exactly as: Lastname et al., Year (Journal)\n"
                "- If there are exactly two authors, format exactly as: Author1 & Author2, Year (Journal)\n"
                "- If there is only one author, format exactly as: Lastname, Year (Journal)\n"
                "- Keep the journal name abbreviated if standard, or use its full title (e.g., Crop Science, Genetics).\n"
                "- Output ONLY the final citation string. Do not include introductory text, markdown quotes, formatting wrappers, or pleasantries."
            )),
            ("human", "Cover Page Text:\n{text}")
        ])

        chain_response = prompt | get_llm | StrOutputParser()
        citation = chain_response.invoke({'text': cover_page})
        clean_citation = citation.strip().replace('"', '').replace("'", "")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        chunks = text_splitter.split_documents(pages)

        for chunk in chunks:
            chunk.metadata['source'] = clean_citation
            chunk.metadata['page'] = chunk.metadata.get('page', 0) + 1

        return chunks
    finally:
        # Clean up the temporary operating system file immediately
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


# initialize backend engine components
core_vector_ret, core_bm25_retriever = load_core_retrievers()
get_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0)

# initialize persistent session state trackers
if 'history' not in st.session_state:
    st.session_state.history = []  # dicts list: {"query": q, "response": r, "sources": s}
if 'active_response' not in st.session_state:
    st.session_state.active_response = None
if 'active_sources' not in st.session_state:
    st.session_state.active_sources = []
if 'active_query' not in st.session_state:
    st.session_state.active_query = ""

# sidebar column (session history and library inventory)
with st.sidebar:
    st.title('QGen AI')
    st.caption('Quantitative Genetics & Plant Breeding RAG Engine')
    st.markdown('---')

    st.subheader('System Status')

    try:
        # get all document meta stored in chromadb
        all_docs = core_vector_ret.get()

        if all_docs and 'metadatas' in all_docs and all_docs['metadatas']:
            # extract unique file paths/names from meta list
            unique_articles = {
                meta.get('source')
                for meta in all_docs['metadatas']
                if meta and meta.get('source')
            }
            total_articles = len(unique_articles)
            total_chunks = len(all_docs['metadatas'])
        else:
            total_articles = 0
            total_chunks = 0

    except Exception:
        # fallback if the chromadb is empty or uninitialized
        total_articles = 0
        total_chunks = 0

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Core Articles", value=f"📚 {total_articles}")
    with col2:
        st.metric(label="Total Chunks", value=f"🧩 {total_chunks}")
    current_k = st.session_state.get('k_depth', 5)
    current_v = st.session_state.get('vector_percentage', 50)
    current_b = 100 - current_v

    st.markdown(
        f"""
            <div class="parameter-box">
                <strong>Current Runtime Profile:</strong><br>
                • Context Depth (k): <code>{current_k} chunks</code><br>
                • Semantic Vector: <code>{current_v}%</code><br>
                • Keyword BM25: <code>{current_b}%</code>
            </div>
            """,
        unsafe_allow_html=True
    )

    with st.expander('Adjust Settings', expanded=False):
        st.caption('Recalibrate retrieval balance and window parameters:')

        # k-value
        k_depth = st.slider(
            'Context Retrieval Depth (k)',
            min_value=2,
            max_value=10,
            value=current_k,
            key='k_depth',
            help='Controls total number of relevant text fragments extracted across data stores to fuel the prompt.'
        )

        # slider
        vector_percentage = st.slider(
            'Semantic Vector Weight',
            min_value=0,
            max_value=100,
            value=current_v,
            step=5,
            key='vector_percentage',
            help='100% = Pure contextual concept match. 0% = Pure alphanumeric string match. 50% = Balanced Hybrid Search.'
        )

        # calc weights for the LangChain pipeline
        v_weight = vector_percentage / 100.0
        k_weight = 1.0 - v_weight

        # semantic vec and bm25 balance
        st.caption(f'Balance Layout: {vector_percentage}% Vector / {100 - vector_percentage}% BM25')

    st.markdown('---')

    # drag and drop pdf
    st.subheader('Article Upload')
    uploaded_file = st.file_uploader(
        'Supplement the database with an external research paper (PDF):',
        type=['pdf'],
        help='This file is safely processed directly inside your isolated server RAM session. '
             '\nIt will be deleted once the session is closed.'
    )

    temp_chunks = []
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        # call cached function for temp pdf
        temp_chunks = process_new_pdfs(file_bytes, uploaded_file.name)
        st.success(f"Loaded: `{temp_chunks[0].metadata['source']}` ({len(temp_chunks)} chunks)")

    st.markdown('---')
    st.subheader('Query History')
    if not st.session_state.history:
        st.caption('No queries run in this active session.')
    else:
        for idx, record in enumerate(reversed(st.session_state.history)):
            if st.button(f"🔍 {record['query'][:28]}...", key=f"hist_{idx}", use_container_width=True):
                st.session_state.active_query = record['query']
                st.session_state.active_response = record['response']
                st.session_state.active_sources = record['sources']

# main app workspace
st.title('Scientific Workspace')

# context layout
if uploaded_file is not None:
    scope = st.radio(
        '**Select Search Context Scope:**',
        options=['Core Database Only', 'Uploaded Paper Only',
                 'Blend Frameworks Together (Cross-Literature Synthesis)'],
        index=2,
        horizontal=True
    )
else:
    st.info(
        'ℹ️ System active on core database index. Upload an external paper in the sidebar to unlock cross-literature blending.')
    scope = 'Core Database Only'

with st.form(key="search_query_form", clear_on_submit=False):
    if st.session_state.active_query:
        st.caption(f'🏁 Active View Focus: *\"{st.session_state.active_query}\"*')

    user_query = st.text_input(
        'Enter scientific inquiry',
        value=st.session_state.active_query,
        placeholder='Compare different statistical methods for multi-environment GxE interaction...',
        key='fresh_query_input'
    )

    # enter should also work for submission
    submit_pipeline = st.form_submit_button(label="Execute Pipeline", type="primary")

if submit_pipeline and user_query:
    st.session_state.active_query = user_query

    with st.spinner('Assembling query and generating response...'):

        core_vector_retriever = core_vector_ret.as_retriever(search_kwargs={'k': k_depth})
        core_bm25_retriever.k = k_depth

        # matrix construction for type of engine
        if scope == 'Core Database Only':
            active_retriever = EnsembleRetriever(
                retrievers=[core_vector_retriever, core_bm25_retriever],
                weights=[v_weight, k_weight]
            )
        elif scope == 'Uploaded Paper Only':
            embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
            temp_db = Chroma.from_documents(temp_chunks, embeddings)
            temp_vector_retriever = temp_db.as_retriever(search_kwargs={'k': k_depth})
            temp_bm25_retriever = BM25Retriever.from_documents(temp_chunks)
            temp_bm25_retriever.k = k_depth

            active_retriever = EnsembleRetriever(
                retrievers=[temp_vector_retriever, temp_bm25_retriever],
                weights=[v_weight, k_weight]
            )
        else:  # blend both together
            embeddings = OpenAIEmbeddings(model='text-embedding-3-large')
            temp_db = Chroma.from_documents(temp_chunks, embeddings)
            temp_vector_retriever = temp_db.as_retriever(search_kwargs={'k': k_depth})
            temp_bm25_retriever = BM25Retriever.from_documents(temp_chunks)
            temp_bm25_retriever.k = k_depth

            # 4-way
            active_retriever = EnsembleRetriever(
                retrievers=[core_vector_retriever, core_bm25_retriever, temp_vector_retriever, temp_bm25_retriever],
                weights=[v_weight * 0.5, k_weight * 0.5, v_weight * 0.5, k_weight * 0.5]
            )

        retrieved_chunks = active_retriever.invoke(user_query)
        st.session_state.active_sources = retrieved_chunks

        # assemble grounding matrix
        context_string = "\n\n".join([
            f"--- START SOURCE CHUNK ({chunk.metadata.get('source', 'Unknown')}, Page {chunk.metadata.get('page', 'Unknown')}) ---\n"
            f"{chunk.page_content}\n"
            f"--- END SOURCE CHUNK ---"
            for chunk in retrieved_chunks
        ])

        #  grounding prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite AI Research Assistant specializing in quantitative genetics.\n\n"
                "Your task is to answer the User Query using ONLY the text segments provided in the Context block.\n\n"
                "CRITICAL GUARDRAILS:\n"

                "1. If the text details multiple concepts (e.g., different statistical models) separately but does not explicitly compare them, "
                "you ARE permitted to use your advanced domain knowledge to connect the dots, synthesize their differences, and perform a rigorous comparative analysis.\n"
                "However, you must clearly distinguish between facts directly pulled from the paper and your own expert synthesis."
                "Do not attempt to merge unrelated data points into a cohesive narrative\n"
                "2. STRICT GROUNDING: Do not use your own external background knowledge to answer. Treat each chunk as an isolated fact."
                "If the context chunks do not explicitly state the mechanism or answer, say 'The provided articles do not contain enough "
                "specific data to answer this query.'\n"
                "3. LITERAL CITATIONS: You must format citations exactly as (filename: [Exact Source Filename], p. [Page Number]). "
                "Do not invent filenames from section headers or text inside the chunks.\n"
                "4. SILENT FILTERING: Ignore context chunks that do not directly address the subject of the query.\n\n"
                "Do not attempt to merge unrelated data points into a cohesive narrative.\n"
                "5. **Handle Missing Specifics Safely:** If the query asks for a specific value or comparison at a certain point and the text only provides it for one trait, explicitly state what is present and specify exactly what is missing."
                "6. **Strict Verbatim Grounding:** Do not guess, smooth over discrepancies, or generalize. Every name, location, and numeric assertion must map cleanly to an explicit statement in the context."
                "Context:\n{context}"

            )),
            ("human", "{query}")
        ])

        # execute synthesis chain
        chain = prompt | get_llm | StrOutputParser()
        response = chain.invoke({'context': context_string, 'query': user_query})
        st.session_state.active_response = response

        # append to history tracking matrix
        st.session_state.history.append({
            'query': user_query,
            'response': response,
            'sources': retrieved_chunks
        })

# side by side window
if st.session_state.active_response:
    st.markdown('---')

    # side-by-side view matrix (60% Response, 40% Evidence Vault)
    left_col, right_col = st.columns([3, 2], gap='medium')

    with left_col:
        st.subheader('QGen AI Response')

        # guardrail indicator logic
        if 'not contain enough specific data' in st.session_state.active_response.lower() or 'insufficient information' in st.session_state.active_response.lower():
            st.warning('⚠️ Data Gap Detected: Grounding block forced a factual containment refusal.')
        else:
            st.success('✅ Fully Grounded: Content successfully synthesized from active citations.')

        st.markdown(st.session_state.active_response)

        # exportable briefing doc
        st.markdown("---")
        st.subheader('Export Research Dossier')
        st.caption('Generate a clean, standalone report containing this synthesis and its verified source trails.')

        # compile raw markdown document
        report_markdown = f'# QGen AI Research Briefing\n\n'
        report_markdown += f'### **Query Inquiry:**\n*{st.session_state.active_query}*\n\n'
        report_markdown += f'--- \n\n## 📋 AI Response\n{st.session_state.active_response}\n\n'
        report_markdown += f'--- \n\n## 📁 Verbatim Evidence Log\n'

        for idx, chunk in enumerate(st.session_state.active_sources):
            report_markdown += f"### [{idx + 1}] {chunk.metadata.get('source', 'Unknown')} — Page {chunk.metadata.get('page', 'Unknown')}\n"
            report_markdown += f"> *{chunk.page_content.strip()}*\n\n"

        # download
        st.download_button(
            label='Download Research Brief (.md)',
            data=report_markdown,
            file_name=f"qgen_dossier_{st.session_state.active_query[:15].strip().lower().replace(' ', '_')}.md",
            mime='text/markdown',
            use_container_width=True
        )

    with right_col:
        st.subheader('Evidence Area')
        st.caption('Verbatim underlying text chunks pulled by search:')

        st.markdown('<div class="evidence-container">', unsafe_allow_html=True)

        for idx, chunk in enumerate(st.session_state.active_sources):
            citation_title = chunk.metadata.get('source', 'Unknown Reference')
            page_num = chunk.metadata.get('page', 'Unknown')

            with st.expander(f'📍 [{idx + 1}] {citation_title} — Page {page_num}', expanded=False):
                st.markdown(f'*{chunk.page_content}*')
                st.caption(f'Score metadata weight tracking enabled.')

        st.markdown('</div>', unsafe_allow_html=True)
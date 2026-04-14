import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import re
import os
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="PDF Smart Searcher", layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
    }
    
    h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #000080 !important;
        font-weight: bold;
    }
    
    [data-testid="stTextInput"] input {
        background-color: #ffffff !important;
        color: #000080 !important;
        caret-color: #000080 !important;
        border: 1px solid #ccd1d9 !important;
        border-radius: 5px;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f0f7ff !important;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
        color: #000080 !important;
        border: 1px dashed #000080 !important;
        padding: 20px;
        border-radius: 10px;
    }
    
    [data-testid="stFileUploader"] section div, 
    [data-testid="stFileUploader"] section label, 
    [data-testid="stFileUploader"] section small {
        color: #000080 !important;
    }
    
    [data-testid="stFileUploader"] button {
        background-color: #e3f2fd !important;
        color: #000080 !important;
        border: 1px solid #bbdefb !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #000080 !important;
    }
    
    div[data-testid="stNotification"], 
    div[data-testid="stNotification"] *, 
    section[data-testid="stSidebar"] div[data-testid="stNotification"] *,
    .stAlert p {
        color: #000080 !important;
    }
    
    [data-testid="stTextInput"] label p {
        color: #000080 !important;
        font-weight: bold;
    }

    mark {
        background-color: #fff176 !important;
        color: #000000 !important;
        font-weight: bold;
    }
    
    /* Expander card - Chữ Navy cho kết quả */
    .stExpander div, .stExpander p, .stExpander span {
        color: #000080 !important;
    }
    .stExpander {
        border: 1px solid #e3f2fd !important;
        background-color: #ffffff !important;
    }

    /* Nút đóng/mở Sidebar - Ép màu Navy cực mạnh */
    [data-testid="stSidebarCollapseButton"], 
    [data-testid="stSidebarCollapseButton"] button,
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"],
    .collapsedControl,
    .st-emotion-cache-6qob1r {
        background-color: #000080 !important;
        color: white !important;
    }
    /* Đảm bảo mũi tên/biểu tượng luôn hiện màu trắng */
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stSidebarCollapseButton"] path,
    button[aria-label="Close sidebar"] svg, 
    button[aria-label="Open sidebar"] svg,
    button[aria-label="Close sidebar"] path,
    button[aria-label="Open sidebar"] path {
        fill: white !important;
        stroke: white !important; /* Thêm stroke để chắc chắn hiện mũi tên */
    }


    [data-testid="StyledFullScreenButton"],
    button[title="View fullscreen"] {
        display: none !important;
    }

    /* Thu hẹp khoảng cách các thành phần trong sidebar */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }

    [data-testid="stFileUploader"] *, 
    [data-testid="stFileUploaderFileName"],
    section[data-testid="stSidebar"] .stText {
        color: #000080 !important;
    }
    </style>



    """, unsafe_allow_html=True)

st.title("🔍 PDF Smart Searcher")

DB_FAISS_PATH = "vectorstore/db_faiss"
INDEXED_FILES_PATH = "vectorstore/indexed_files.json"

if "db" not in st.session_state:
    st.session_state.db = None
if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = {}

def load_indexed_files():
    if os.path.exists(INDEXED_FILES_PATH):
        try:
            with open(INDEXED_FILES_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_indexed_files(indexed_files):
    os.makedirs("vectorstore", exist_ok=True)
    with open(INDEXED_FILES_PATH, 'w', encoding='utf-8') as f:
        json.dump(indexed_files, f, ensure_ascii=False, indent=2)

def get_file_hash(file_path):
    import hashlib
    file_stat = os.stat(file_path)
    return f"{file_stat.st_size}_{file_stat.st_mtime}"

def load_existing_db():
    if os.path.exists(DB_FAISS_PATH):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            return db
        except:
            return None
    return None

def highlight_keywords(text, keywords):
    keyword_list = [kw.strip() for kw in keywords.split() if kw.strip()]
    highlighted_text = text
    for keyword in keyword_list:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        highlighted_text = pattern.sub(
            lambda m: f"<mark>{m.group()}</mark>",
            highlighted_text
        )
    return highlighted_text

def check_all_keywords_present_exact(text, keywords):
    text_lower = text.lower()
    keyword_list = [kw.strip().lower() for kw in keywords.split() if kw.strip()]
    if not keyword_list:
        return False
    for keyword in keyword_list:
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if not re.search(pattern, text_lower):
            return False
    return True

def calculate_proximity_score(text, keywords):
    text_lower = text.lower()
    keyword_list = [kw.strip().lower() for kw in keywords.split() if kw.strip()]
    if len(keyword_list) <= 1:
        return 100.0
    indices = []
    for kw in keyword_list:
        idx = text_lower.find(kw)
        if idx != -1:
            indices.append(idx)
    if len(indices) < len(keyword_list):
        return 0.0
    spread = max(indices) - min(indices)
    return 1000.0 / (spread + 10.0) 

def extract_sentences_with_all_keywords(text, keywords):
    keyword_list = [kw.strip().lower() for kw in keywords.split() if kw.strip()]
    if not keyword_list:
        return []
    sentences = re.split(r'(?<=[.!?。！？\n])\s+', text)
    matching_sentences = []
    for sentence in sentences:
        if check_all_keywords_present_exact(sentence, keywords):
            matching_sentences.append(sentence.strip())
    return matching_sentences

def strict_keyword_search_from_db(query, db):
    keyword_list = [kw.strip().lower() for kw in query.split() if kw.strip()]
    if not keyword_list:
        return []
    candidates = db.similarity_search(query, k=200)
    scored_results = []
    seen_contents = set()
    for doc in candidates:
        content = doc.page_content
        content_lower = content.lower()
        if check_all_keywords_present_exact(content, query):
            if content not in seen_contents:
                keyword_score = sum(content_lower.count(kw) for kw in keyword_list)
                proximity_score = calculate_proximity_score(content, query)
                total_score = (keyword_score * 10) + proximity_score
                scored_results.append((total_score, doc))
                seen_contents.add(content)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_results]

def build_index_from_files(uploaded_files):
    documents = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    indexed_files = load_indexed_files()
    new_files_to_process = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        temp_path = f"temp_{file_name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        file_hash = get_file_hash(temp_path)
        if file_name in indexed_files and indexed_files[file_name] == file_hash:
            os.remove(temp_path)
        else:
            new_files_to_process.append((temp_path, file_name, file_hash))
    if not new_files_to_process and len(uploaded_files) > 0:
        if os.path.exists(DB_FAISS_PATH):
            db = load_existing_db()
            progress_bar.empty()
            status_text.empty()
            return db
    for idx, (temp_path, file_name, file_hash) in enumerate(new_files_to_process):
        try:
            status_text.text(f"🚀 Đang đọc cực nhanh: {file_name}")
            loader = PyMuPDFLoader(temp_path)
            documents.extend(loader.load())
            os.remove(temp_path)
            indexed_files[file_name] = file_hash
            progress_bar.progress((idx + 1) / len(new_files_to_process))
        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            if os.path.exists(temp_path): os.remove(temp_path)
    if len(documents) == 0:
        db = load_existing_db()
        progress_bar.empty()
        status_text.empty()
        save_indexed_files(indexed_files)
        return db
    status_text.text("✂️ Đang chia nhỏ văn bản...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100, separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""])
    chunks = text_splitter.split_documents(documents)
    status_text.text(f"🧠 Đang tạo embedding ({len(chunks)} chunks)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(DB_FAISS_PATH) and len(documents) > 0:
        try:
            existing_db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
            new_db = FAISS.from_documents(chunks, embeddings)
            existing_db.merge_from(new_db)
            vector_db = existing_db
        except:
            vector_db = FAISS.from_documents(chunks, embeddings)
    else:
        vector_db = FAISS.from_documents(chunks, embeddings)
    os.makedirs("vectorstore", exist_ok=True)
    vector_db.save_local(DB_FAISS_PATH)
    save_indexed_files(indexed_files)
    progress_bar.empty()
    status_text.empty()
    return vector_db

with st.sidebar:
    st.header("📤 Tải lên PDF")

    st.divider()
    st.session_state.indexed_files = load_indexed_files()
    if st.session_state.indexed_files:
        st.caption(f"📊 Số file đã index: {len(st.session_state.indexed_files)}")
        with st.expander("📋 Danh sách file đã index"):
            for file_name in st.session_state.indexed_files.keys():
                st.text(f"✅ {file_name}")
    st.divider()
    uploaded_files = st.file_uploader("Chọn file PDF để phân tích", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
    if uploaded_files:
        if st.button("🚀 Bắt đầu xây dựng Index", use_container_width=True):
            with st.spinner("⏳ Đang xử lý..."):
                db = build_index_from_files(uploaded_files)
                if db:
                    st.session_state.db = db
                    st.session_state.indexed = True
                    st.session_state.indexed_files = load_indexed_files()
                    st.success(f"✅ Thành công! Hiện có {len(st.session_state.indexed_files)} file")
                    st.rerun()
    st.divider()
    if not st.session_state.indexed and os.path.exists(DB_FAISS_PATH):
        st.session_state.db = load_existing_db()
        st.session_state.indexed_files = load_indexed_files()
        if st.session_state.db:
            st.session_state.indexed = True
    if st.session_state.indexed and st.session_state.db:
        st.success("✅ Cơ sở dữ liệu đã sẵn sàng")
        if st.button("🗑️ Xóa tất cả dữ liệu", use_container_width=True):
            import shutil
            if os.path.exists("vectorstore"):
                shutil.rmtree("vectorstore")
            st.session_state.indexed = False
            st.session_state.db = None
            st.session_state.indexed_files = {}
            st.rerun()
    else:
        st.warning("⚠️ Chưa có dữ liệu. Vui lòng tải lên file PDF")

if st.session_state.indexed and st.session_state.db:
    query = st.text_input("🔍 Nhập từ khóa hoặc nội dung cần tìm:")
    if query:
        keyword_list = [kw.strip().lower() for kw in query.split() if kw.strip()]
        if len(keyword_list) == 0:
            st.warning("⚠️ Vui lòng nhập từ khóa!")
        else:
            with st.spinner("🔍 Đang tìm kiếm..."):
                db = st.session_state.db
                results = strict_keyword_search_from_db(query, db)
            if results:
                st.markdown(f"### => Tìm thấy {len(results)} kết quả")
                result_count = 0
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'Không xác định').split('\\')[-1]
                    page = doc.metadata.get('page', 'N/A')
                    matching_sentences = extract_sentences_with_all_keywords(doc.page_content, query)
                    if len(matching_sentences) > 0:
                        result_count += 1
                        with st.expander(f"**#{result_count}** 📄 {source} | Trang {page} ({len(matching_sentences)} câu khớp)", expanded=(result_count == 1)):
                            for idx, sentence in enumerate(matching_sentences, 1):
                                highlighted_sentence = highlight_keywords(sentence, query)
                                st.markdown(f"**{idx}.** {highlighted_sentence}", unsafe_allow_html=True)
                            st.caption(f"Nguồn: {source} | Trang: {page}")
            else:
                st.warning(f"⚠️ Không tìm thấy tài liệu nào chứa TẤT CẢ từ khóa: {', '.join([f'**{kw}**' for kw in keyword_list])}")
else:
    st.info("📤 Vui lòng tải lên file PDF ở sidebar để bắt đầu!")
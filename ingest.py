import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Tạo các folder cần thiết nếu chưa có
for folder in ["data", "vectorstore"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def build_index():
    documents = []
    pdf_files = [f for f in os.listdir("data") if f.endswith(".pdf")]
    
    if not pdf_files:
        print("❌ Thư mục 'data' đang trống. Hãy bỏ file PDF vào nhé!")
        return

    for file in pdf_files:
        print(f"📄 Đang đọc file: {file}...")
        loader = PyMuPDFLoader(os.path.join("data", file))
        documents.extend(loader.load())

    # Chia nhỏ văn bản - Tối ưu cho tìm kiếm chính xác
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Nhỏ hơn để chunk cụ thể
        chunk_overlap=100,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    print(f"📊 Tổng số chunks: {len(chunks)}")

    # Embedding - Model mạnh hơn cho semantic search
    print("🧠 AI đang mã hóa nội dung (Embedding)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("vectorstore/db_faiss")
    print("✅ Xong! AI đã học xong toàn bộ file PDF.")
    print(f"💾 Lưu {len(chunks)} chunks vào vectorstore")

if __name__ == "__main__":
    build_index()
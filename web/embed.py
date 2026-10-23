#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Embedding dữ liệu RAG và lưu trữ vào PostgreSQL (PGVector).
Sử dụng mô hình Hugging Face 'AITeamVN/Vietnamese_Embedding' (gốc BAAI/bge-m3)
chạy trực tiếp trên GPU CUDA.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Cấu hình encoding UTF-8 cho Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Tải biến môi trường từ .env (nếu có)
load_dotenv()

# Cấu hình mặc định
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
DEFAULT_CHUNKS_PATH = "chunks.json"
DEFAULT_BATCH_SIZE = 32

# Cấu hình PostgreSQL
PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DATABASE = os.getenv("PG_DATABASE", "integrix_rag")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgrespassword")


def get_device() -> str:
    """Kiểm tra và trả về thiết bị tính toán tối ưu (GPU hoặc CPU)."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[+] Sử dụng GPU: {gpu_name} ({vram:.1f} GB VRAM) - Tăng tốc CUDA")
        return "cuda"
    print("[!] Không tìm thấy GPU CUDA, chuyển sang chạy trên CPU")
    return "cpu"


def load_model(model_name: str = DEFAULT_MODEL, device: Optional[str] = None) -> SentenceTransformer:
    """Tải mô hình embedding từ Hugging Face."""
    device = device or get_device()
    print(f"[*] Đang tải mô hình '{model_name}' trên thiết bị '{device}'...")
    start = time.time()
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = 2048  # Tận dụng context length 2048 của bge-m3
    print(f"[+] Mô hình đã sẵn sàng ({time.time() - start:.2f}s) | Max tokens: {model.max_seq_length}")
    return model


def get_db_connection(host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE, user=PG_USER, password=PG_PASSWORD):
    """Kết nối tới PostgreSQL và kích hoạt extension pgvector."""
    import psycopg
    from pgvector.psycopg import register_vector

    conn_str = f"host={host} port={port} dbname={dbname} user={user} password={password}"
    try:
        conn = psycopg.connect(conn_str, autocommit=True)
        # Đảm bảo extension vector đã được tạo trước khi đăng ký kiểu dữ liệu
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        return conn
    except Exception as e:
        print(f"[LỖI] Không thể kết nối PostgreSQL ({host}:{port}/{dbname}): {e}")
        print("  Gợi ý: Nếu chưa bật database, bạn có thể khởi động bằng:")
        print("         docker compose up -d")
        return None


def init_database_schema(conn, vector_dim: int = 1024) -> None:
    """Khởi tạo extension vector, bảng document_chunks và chỉ mục HNSW."""
    with conn.cursor() as cur:
        # 1. Kích hoạt extension pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # 2. Tạo bảng lưu trữ chunks & embeddings
        cur.execute(f"""
        CREATE TABLE IF NOT EXISTS document_chunks (
            id VARCHAR(100) PRIMARY KEY,
            file_name VARCHAR(255) NOT NULL,
            doc_title TEXT NOT NULL,
            section_title TEXT NOT NULL,
            content TEXT NOT NULL,
            char_count INT NOT NULL,
            metadata JSONB,
            embedding VECTOR({vector_dim}),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # 3. Tạo chỉ mục HNSW tối ưu tìm kiếm Cosine Similarity
        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
        ON document_chunks USING hnsw (embedding vector_cosine_ops);
        """)
    print("[+] Cơ sở dữ liệu và bảng 'document_chunks' đã được khởi tạo chuẩn hóa.")


def compute_embeddings(chunks: List[dict], model: SentenceTransformer, batch_size: int = DEFAULT_BATCH_SIZE) -> np.ndarray:
    """Tạo vector embedding cho toàn bộ chunks (ghép Doc Title + Section Title + Content để tối ưu ngữ nghĩa)."""
    texts = [
        f"{c.get('doc_title', '')} | {c.get('section_title', '')}\n{c.get('content', '')}".strip()
        for c in chunks
    ]
    print(f"[*] Đang sinh embedding cho {len(texts)} chunks (batch_size={batch_size})...")
    start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # Chuẩn hóa để tính Cosine Distance chính xác
    )
    elapsed = time.time() - start
    print(f"[+] Hoàn tất sinh embedding: {len(embeddings)} vectors ({elapsed:.2f}s, ~{len(texts)/max(elapsed, 0.001):.1f} chunks/s).")
    return np.array(embeddings, dtype=np.float32)


def upsert_to_pgvector(chunks: List[dict], embeddings: np.ndarray, conn) -> None:
    """Lưu / Cập nhật (Upsert) danh sách chunks và vector vào PostgreSQL."""
    print(f"[*] Đang lưu {len(chunks)} records vào PostgreSQL...")
    start = time.time()
    query = """
    INSERT INTO document_chunks (id, file_name, doc_title, section_title, content, char_count, metadata, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        file_name = EXCLUDED.file_name,
        doc_title = EXCLUDED.doc_title,
        section_title = EXCLUDED.section_title,
        content = EXCLUDED.content,
        char_count = EXCLUDED.char_count,
        metadata = EXCLUDED.metadata,
        embedding = EXCLUDED.embedding,
        updated_at = CURRENT_TIMESTAMP;
    """

    records = [
        (
            c["id"],
            c["file_name"],
            c["doc_title"],
            c["section_title"],
            c["content"],
            c["char_count"],
            json.dumps(c.get("metadata", {})),
            embeddings[i],
        )
        for i, c in enumerate(chunks)
    ]

    with conn.cursor() as cur:
        cur.executemany(query, records)

    print(f"[+] Đã lưu thành công {len(records)} chunks vào PGVector ({time.time() - start:.2f}s).")


def search_similar_chunks(
    query: str,
    model: SentenceTransformer,
    conn,
    top_k: int = 5,
    lang_filter: Optional[str] = None,
) -> None:
    """Tìm kiếm các chunk có độ tương đồng ngữ nghĩa cao nhất bằng khoảng cách Cosine (<=>)."""
    print(f"\n🔍 Truy vấn: \"{query}\"")
    query_vector = model.encode([query], normalize_embeddings=True)[0]

    sql = """
    SELECT id, file_name, doc_title, section_title, content, metadata,
           1 - (embedding <=> %s) AS similarity
    FROM document_chunks
    """
    params = [query_vector]

    if lang_filter:
        sql += " WHERE metadata->>'lang' = %s "
        params.append(lang_filter)

    sql += " ORDER BY embedding <=> %s LIMIT %s;"
    params.extend([query_vector, top_k])

    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        print("  Không tìm thấy kết quả phù hợp.")
        return

    print("=" * 80)
    for rank, (cid, fname, doc_title, sec_title, content, meta, sim) in enumerate(rows, 1):
        print(f"Top {rank} | Độ tương đồng: {sim * 100:.2f}% | ID: {cid} | File: {fname}")
        print(f"  📌 Tiêu đề: {doc_title} -> {sec_title}")
        print(f"  📄 Nội dung chi tiết:\n{content.strip()}")
        print("-" * 80)


def resolve_chunks_file(file_arg: str) -> Path:
    """Tìm đường dẫn file chunks.json từ đối số hoặc thư mục hiện tại/web."""
    p = Path(file_arg)
    if p.is_file():
        return p
    candidates = [
        Path(__file__).parent / p.name,
        Path("web") / p.name,
        Path(p.name),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return p


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sinh Embedding với AITeamVN/Vietnamese_Embedding và lưu vào PGVector."
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_CHUNKS_PATH,
        help=f"Đường dẫn file chunks.json (mặc định: '{DEFAULT_CHUNKS_PATH}')",
    )
    parser.add_argument(
        "--model", "-m",
        default=DEFAULT_MODEL,
        help=f"Tên mô hình Hugging Face (mặc định: '{DEFAULT_MODEL}')",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Kích thước batch khi encode (mặc định: 32)",
    )
    parser.add_argument(
        "--save-local",
        default=None,
        help="Lưu vector ra file cục bộ (.npz) để sao lưu hoặc dùng offline mà không cần kết nối DB ngay",
    )
    parser.add_argument(
        "--sync-local",
        default=None,
        help="Nạp vector đã lưu từ file .npz và đẩy thẳng vào PGVector (không cần encode lại)",
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Thử nghiệm tìm kiếm ngữ nghĩa với câu truy vấn",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Số lượng kết quả trả về khi tìm kiếm (mặc định: 5)",
    )
    parser.add_argument(
        "--lang",
        default=None,
        choices=["vi", "en"],
        help="Lọc ngôn ngữ khi tìm kiếm ('vi' hoặc 'en')",
    )

    args = parser.parse_args()

    # Trường hợp 0: Đồng bộ từ file vector .npz đã lưu trước đó vào PostgreSQL
    if args.sync_local:
        npz_path = Path(args.sync_local)
        if not npz_path.is_file():
            print(f"[LỖI] Không tìm thấy file: {npz_path}")
            return
        input_file = resolve_chunks_file(args.input)
        chunks_data = json.loads(input_file.read_text(encoding="utf-8"))
        embeddings = np.load(npz_path)["embeddings"]
        print(f"[+] Đã tải {len(embeddings)} vectors từ '{npz_path.name}' và {len(chunks_data)} chunks.")

        conn = get_db_connection()
        if not conn:
            return
        init_database_schema(conn, vector_dim=embeddings.shape[1])
        upsert_to_pgvector(chunks_data, embeddings, conn)
        conn.close()
        print("[+] Đồng bộ hoàn tất!")
        return

    # Tải mô hình embedding
    model = load_model(args.model)

    # Trường hợp 1: Chế độ tìm kiếm truy vấn (Query Search)
    if args.query:
        conn = get_db_connection()
        if not conn:
            print("[LỖI] Không thể tìm kiếm vì chưa kết nối được cơ sở dữ liệu PostgreSQL.")
            return
        search_similar_chunks(args.query, model, conn, top_k=args.top_k, lang_filter=args.lang)
        conn.close()
        return

    # Trường hợp 2: Chế độ Embedding và Lưu trữ (Ingestion)
    input_file = resolve_chunks_file(args.input)
    if not input_file.is_file():
        print(f"[LỖI] Không tìm thấy file dữ liệu chunks: {args.input}")
        print("  Gợi ý: Hãy tạo chunks trước bằng lệnh: python chunk.py --all -o chunks.json")
        return

    chunks_data = json.loads(input_file.read_text(encoding="utf-8"))
    print(f"[+] Đã đọc {len(chunks_data)} chunks từ: {input_file.resolve()}")

    # Sinh embeddings
    embeddings = compute_embeddings(chunks_data, model, batch_size=args.batch_size)

    # Lưu cục bộ nếu có yêu cầu
    if args.save_local:
        out_local = Path(args.save_local)
        np.savez_compressed(out_local, embeddings=embeddings)
        print(f"[+] Đã lưu file vector cục bộ: {out_local.resolve()}")

    # Kết nối PostgreSQL và lưu vào PGVector
    print("\n[*] Đang kết nối tới PostgreSQL để lưu trữ...")
    conn = get_db_connection()
    if conn:
        init_database_schema(conn, vector_dim=embeddings.shape[1])
        upsert_to_pgvector(chunks_data, embeddings, conn)
        conn.close()
        print("\n Hoàn tất toàn bộ quy trình Embedding và Lưu trữ vào PGVector!")
        print("  Bạn có thể thử tìm kiếm bằng lệnh:")
        print(f"  python {Path(__file__).name} --query \"Giải pháp giao thông thông minh của Integrix\"")
    else:
        print("\n[!] Không thể lưu vào PGVector do chưa kết nối được PostgreSQL.")
        if not args.save_local:
            # Tự động sao lưu tạm vào file .npz để không lãng phí công encode
            auto_save = input_file.with_name("cached_embeddings.npz")
            np.savez_compressed(auto_save, embeddings=embeddings)
            print(f"  -> Đã tự động lưu vector tạm vào: {auto_save.resolve()}")
            print("  -> Sau khi bật PostgreSQL (`docker compose up -d`), bạn có thể đẩy dữ liệu lên bất cứ lúc nào.")


if __name__ == "__main__":
    main()

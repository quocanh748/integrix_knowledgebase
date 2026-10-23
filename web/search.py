#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Hybrid Search kết hợp:
1. Dense Search: Vector Search (PostgreSQL PGVector + AITeamVN/Vietnamese_Embedding)
2. Sparse Search: Keyword Search (BM25)
3. Hợp nhất kết quả bằng Reciprocal Rank Fusion (RRF) với hằng số chuẩn k = 60.
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import psycopg
import torch
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Cấu hình UTF-8 cho Windows CMD / PowerShell
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

load_dotenv()

# Cấu hình mặc định
DEFAULT_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding")
DEFAULT_RRF_K = 60  # Hằng số chuẩn kinh điển của Reciprocal Rank Fusion
DEFAULT_TOP_K = 5
DEFAULT_CANDIDATES = 30  # Số ứng viên lấy từ mỗi nhánh (Dense & Sparse) trước khi hợp nhất

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = int(os.getenv("PG_PORT", 5432))
PG_DATABASE = os.getenv("PG_DATABASE", "integrix_rag")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgrespassword")


@dataclass
class SearchResult:
    id: str
    file_name: str
    doc_title: str
    section_title: str
    content: str
    metadata: dict
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: float = 0.0


def tokenize(text: str) -> List[str]:
    """Tách từ đơn giản bằng regex hỗ trợ đầy đủ tiếng Việt có dấu và tiếng Anh."""
    return re.findall(r"\w+", text.lower())


class HybridSearchEngine:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int = DEFAULT_CANDIDATES,
    ):
        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        # 1. Kết nối PostgreSQL
        conn_str = f"host={PG_HOST} port={PG_PORT} dbname={PG_DATABASE} user={PG_USER} password={PG_PASSWORD}"
        try:
            self.conn = psycopg.connect(conn_str, autocommit=True)
            register_vector(self.conn)
        except Exception as e:
            print(f"[LỖI] Không thể kết nối cơ sở dữ liệu PGVector: {e}")
            sys.exit(1)

        # 2. Tải toàn bộ tài liệu từ DB để xây dựng chỉ mục BM25
        print("[*] Đang khởi tạo chỉ mục BM25 từ cơ sở dữ liệu...")
        self.doc_store: Dict[str, dict] = {}
        corpus_tokens = []
        self.doc_ids: List[str] = []

        with self.conn.cursor() as cur:
            cur.execute("""
            SELECT id, file_name, doc_title, section_title, content, metadata
            FROM document_chunks;
            """)
            rows = cur.fetchall()

        for cid, fname, doc_title, sec_title, content, meta in rows:
            self.doc_store[cid] = {
                "id": cid,
                "file_name": fname,
                "doc_title": doc_title,
                "section_title": sec_title,
                "content": content,
                "metadata": meta if isinstance(meta, dict) else json.loads(meta or "{}"),
            }
            self.doc_ids.append(cid)
            # Kết hợp tiêu đề và nội dung để BM25 tính điểm chính xác
            searchable_text = f"{doc_title} {sec_title} {content}"
            corpus_tokens.append(tokenize(searchable_text))

        self.bm25 = BM25Okapi(corpus_tokens)
        print(f"[+] Chỉ mục BM25 sẵn sàng với {len(self.doc_ids)} chunks.")

        # 3. Tải mô hình embedding cho Dense Search
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Đang tải mô hình Dense Search '{model_name}' trên '{device}'...")
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 2048
        print(f"[+] Hệ thống Hybrid Search đã sẵn sàng hoạt động!\n")

    def dense_search(self, query: str, top_k: int, lang_filter: Optional[str] = None) -> List[Tuple[str, int, float]]:
        """Dense Search: Tìm kiếm ngữ nghĩa bằng vector embedding trong PGVector."""
        query_vector = self.model.encode([query], normalize_embeddings=True)[0]

        sql = """
        SELECT id, 1 - (embedding <=> %s) AS similarity
        FROM document_chunks
        """
        params = [query_vector]

        if lang_filter:
            sql += " WHERE metadata->>'lang' = %s "
            params.append(lang_filter)

        sql += " ORDER BY embedding <=> %s LIMIT %s;"
        params.extend([query_vector, top_k])

        with self.conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [(row[0], rank, float(row[1])) for rank, row in enumerate(rows, 1)]

    def sparse_search(self, query: str, top_k: int, lang_filter: Optional[str] = None) -> List[Tuple[str, int, float]]:
        """Sparse Search: Khớp từ khóa chính xác bằng thuật toán BM25."""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        scored_docs = []

        for idx, score in enumerate(scores):
            cid = self.doc_ids[idx]
            if lang_filter and self.doc_store[cid]["metadata"].get("lang") != lang_filter:
                continue
            if score > 0:
                scored_docs.append((cid, float(score)))

        # Sắp xếp điểm BM25 giảm dần
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [(cid, rank, score) for rank, (cid, score) in enumerate(scored_docs[:top_k], 1)]

    def hybrid_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        lang_filter: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Thực thi Hybrid Search theo đúng kiến trúc:
        1. Gọi đồng thời Sparse (BM25) và Dense (Vector).
        2. Hợp nhất bằng Reciprocal Rank Fusion: RRF = 1 / (k + rank_dense) + 1 / (k + rank_sparse).
        3. Trả về top-k chunks có điểm RRF cao nhất.
        """
        # Lấy top ứng viên từ mỗi bảng xếp hạng
        dense_results = self.dense_search(query, top_k=self.candidate_k, lang_filter=lang_filter)
        sparse_results = self.sparse_search(query, top_k=self.candidate_k, lang_filter=lang_filter)

        dense_map = {cid: (rank, score) for cid, rank, score in dense_results}
        sparse_map = {cid: (rank, score) for cid, rank, score in sparse_results}

        # Tập hợp tất cả ứng viên xuất hiện ở ít nhất 1 bảng xếp hạng
        all_candidate_ids = set(dense_map.keys()) | set(sparse_map.keys())
        merged_results: List[SearchResult] = []

        for cid in all_candidate_ids:
            doc = self.doc_store[cid]
            d_rank, d_score = dense_map.get(cid, (None, None))
            s_rank, s_score = sparse_map.get(cid, (None, None))

            # Tính điểm RRF
            rrf_score = 0.0
            if d_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + d_rank)
            if s_rank is not None:
                rrf_score += 1.0 / (self.rrf_k + s_rank)

            merged_results.append(
                SearchResult(
                    id=cid,
                    file_name=doc["file_name"],
                    doc_title=doc["doc_title"],
                    section_title=doc["section_title"],
                    content=doc["content"],
                    metadata=doc["metadata"],
                    dense_rank=d_rank,
                    sparse_rank=s_rank,
                    dense_score=d_score,
                    sparse_score=s_score,
                    rrf_score=rrf_score,
                )
            )

        # Sắp xếp theo điểm RRF giảm dần
        merged_results.sort(key=lambda x: x.rrf_score, reverse=True)
        return merged_results[:top_k]

    def close(self):
        if self.conn:
            self.conn.close()


def display_results(results: List[SearchResult], mode: str = "hybrid") -> None:
    """In chi tiết toàn bộ nội dung của các chunk trả về."""
    if not results:
        print("  Không tìm thấy kết quả phù hợp.")
        return

    print("=" * 80)
    for rank, res in enumerate(results, 1):
        dense_info = f"#{res.dense_rank} ({res.dense_score*100:.1f}%)" if res.dense_rank else "Không có"
        sparse_info = f"#{res.sparse_rank} (điểm={res.sparse_score:.2f})" if res.sparse_rank else "Không có"

        print(f"Top {rank} | RRF Score: {res.rrf_score:.5f} | ID: {res.id} | File: {res.file_name}")
        print(f"  📊 Xếp hạng thành phần: Vector Rank: {dense_info} | BM25 Rank: {sparse_info}")
        print(f"  📌 Tiêu đề: {res.doc_title} -> {res.section_title}")
        print(f"  📄 Nội dung chi tiết:\n{res.content.strip()}")
        print("-" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Search (BM25 + PGVector) kết hợp Reciprocal Rank Fusion (RRF k=60)."
    )
    parser.add_argument(
        "--query", "-q",
        default=None,
        help="Câu truy vấn tìm kiếm",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Số lượng kết quả trả về (mặc định: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=DEFAULT_RRF_K,
        help=f"Hằng số k cho Reciprocal Rank Fusion (mặc định: {DEFAULT_RRF_K})",
    )
    parser.add_argument(
        "--lang",
        choices=["vi", "en"],
        default=None,
        help="Lọc kết quả theo ngôn ngữ ('vi' hoặc 'en')",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Chế độ tìm kiếm tương tác (gõ câu hỏi liên tục không cần nạp lại mô hình)",
    )

    args = parser.parse_args()

    # Khởi tạo Search Engine (chỉ nạp 1 lần)
    engine = HybridSearchEngine(rrf_k=args.rrf_k)

    try:
        if args.interactive:
            print("=" * 80)
            print(" Chế độ Hybrid Search Tương tác (gõ 'exit' hoặc 'quit' để thoát)")
            print("=" * 80)
            while True:
                try:
                    q = input("\nNhập câu hỏi tìm kiếm: ").strip()
                    if not q:
                        continue
                    if q.lower() in ["exit", "quit", "q"]:
                        print("Tạm biệt!")
                        break
                    start = time.time()
                    results = engine.hybrid_search(q, top_k=args.top_k, lang_filter=args.lang)
                    print(f"\n🔍 Kết quả tìm kiếm ({time.time() - start:.3f}s):")
                    display_results(results)
                except KeyboardInterrupt:
                    print("\nThoát chương trình.")
                    break
        else:
            query = args.query or "Các Lĩnh vực Integrix Đồng hành bằng Giải pháp Thông minh"
            print(f"🔍 Truy vấn Hybrid Search: \"{query}\" (RRF k={args.rrf_k}, Top-k={args.top_k})\n")
            start = time.time()
            results = engine.hybrid_search(query, top_k=args.top_k, lang_filter=args.lang)
            print(f"[+] Thời gian tìm kiếm: {time.time() - start:.3f}s")
            display_results(results)
    finally:
        engine.close()


if __name__ == "__main__":
    main()

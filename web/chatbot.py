import json
import os
import sys
import time
import google.generativeai as genai
from search import HybridSearchEngine

api_key = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=api_key)

llm = genai.GenerativeModel(
    "gemini-3-flash-preview",
    generation_config={"temperature": 0.0}
)

engine = HybridSearchEngine()


def print_chunk_metadata(results):
    """In chi tiết metadata của các chunk tìm được ra màn hình console."""
    print("\n" + "=" * 80)
    print(f"METADATA CỦA CÁC CHUNK TÌM THẤY (TOP {len(results)}):")
    print("=" * 80)
    for idx, item in enumerate(results, 1):
        dense_info = f"#{item.dense_rank} ({item.dense_score*100:.1f}%)" if item.dense_rank else "Không có"
        sparse_info = f"#{item.sparse_rank} (điểm={item.sparse_score:.2f})" if item.sparse_rank else "Không có"
        meta_str = json.dumps(item.metadata, ensure_ascii=False) if isinstance(item.metadata, dict) else str(item.metadata)

        print(f"[{idx}] ID: {item.id} | File: {item.file_name}")
        print(f"    • Tiêu đề (#)   : {item.doc_title}")
        print(f"    • Mục (##)      : {item.section_title}")
        print(f"    • Xếp hạng      : Vector {dense_info} | BM25 {sparse_info} | RRF Score: {item.rrf_score:.5f}")
        print(f"    • Metadata      : {meta_str}")
        print("-" * 80)
    print()


def ask_chatbot(user_query: str, top_k: int = 4, show_metadata: bool = True) -> str:
    total_start = time.time()

    # 1. Đo thời gian Hybrid Search
    search_start = time.time()
    search_results = engine.hybrid_search(user_query, top_k=top_k, status_filter="active")
    search_duration = time.time() - search_start

    if show_metadata:
        print_chunk_metadata(search_results)

    # 2. Xây dựng Context
    context_text = ""
    for idx, item in enumerate(search_results, 1):
        context_text += f"\n--- [Tài liệu tham khảo {idx}] ---\n"
        context_text += f"ID: {item.id} | File: {item.file_name}\n"
        context_text += f"Tiêu đề: {item.doc_title} -> {item.section_title}\n"
        context_text += f"Metadata: {json.dumps(item.metadata, ensure_ascii=False)}\n"
        context_text += f"Nội dung:\n{item.content}\n"

    prompt = f"""Bạn là trợ lý ảo chuyên nghiệp của Integrix Technologies.
Hãy trả lời câu hỏi của người dùng dựa DUY NHẤT vào các tài liệu tham khảo dưới đây.
Nếu tài liệu không chứa đủ thông tin để trả lời, hãy trung thực trả lời: "Tôi không tìm thấy thông tin này trong tài liệu của Integrix", không được tự bịa thông tin.

=== TÀI LIỆU THAM KHẢO ===
{context_text}

=== CÂU HỎI CỦA NGƯỜI DÙNG ===
{user_query}
"""

    # 3. Đo thời gian sinh câu trả lời của LLM (khóa temperature=0.0 để loại bỏ tính ngẫu nhiên)
    llm_start = time.time()
    response = llm.generate_content(
        prompt,
        generation_config={"temperature": 0.0}
    )
    llm_duration = time.time() - llm_start
    total_duration = time.time() - total_start

    # 4. Trích xuất số lượng Token (để tính chi phí)
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0)
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0)
        total_tokens = getattr(response.usage_metadata, "total_token_count", 0)
    else:
        # Dự phòng nếu API không trả về usage_metadata
        try:
            input_tokens = llm.count_tokens(prompt).total_tokens
            output_tokens = llm.count_tokens(response.text).total_tokens
            total_tokens = input_tokens + output_tokens
        except Exception:
            pass

    tokens_per_sec = (output_tokens / llm_duration) if llm_duration > 0 and output_tokens > 0 else 0

    # In bảng thống kê hiệu năng & token
    print("=" * 80)
    print("⚡ THỐNG KÊ TỐC ĐỘ (LATENCY) & TOKEN (CHI PHÍ):")
    print(f"  • Thời gian tìm kiếm (Search)   : {search_duration:.3f} giây")
    print(f"  • Thời gian LLM sinh text       : {llm_duration:.3f} giây (~{tokens_per_sec:.1f} tokens/giây)")
    print(f"  • Tổng thời gian phản hồi       : {total_duration:.3f} giây")
    print("  " + "-" * 50)
    print(f"  • Input (Prompt) Tokens         : {input_tokens:,} tokens (dùng tính chi phí đầu vào)")
    print(f"  • Output (Completion) Tokens    : {output_tokens:,} tokens (dùng tính chi phí đầu ra)")
    print(f"  • Tổng Tokens tiêu thụ          : {total_tokens:,} tokens")
    print("=" * 80)

    return response.text


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-i", "--interactive"]:
        print("=" * 80)
        print("🤖 Integrix Chatbot - Chế độ tương tác (Gõ 'exit' hoặc 'quit' để thoát)")
        print("=" * 80)
        while True:
            try:
                q = input("\n💬 Nhập câu hỏi: ").strip()
                if not q:
                    continue
                if q.lower() in ["exit", "quit", "q"]:
                    print("Tạm biệt!")
                    break
                answer = ask_chatbot(q)
                print(f"\n💡 TRẢ LỜI:\n{answer}")
            except KeyboardInterrupt:
                print("\nThoát chương trình.")
                break
    else:
        query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "THÔNG TIN LIÊN HỆ của Integrix?"
        print(f"\n❓ Câu hỏi: {query}\n")
        answer = ask_chatbot(query)
        print(f"\n💡 TRẢ LỜI:\n{answer}")

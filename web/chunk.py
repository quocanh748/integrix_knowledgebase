#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script chia nhỏ (chunk) file Markdown theo tiêu đề H2 (##)
và trích xuất tiêu đề H1 (#) làm Metadata của tài liệu.
Hỗ trợ xem trước từng file hoặc xuất hàng loạt toàn bộ tài liệu ra JSON cho RAG/PGVector.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Cấu hình encoding UTF-8 cho Windows console tránh lỗi UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class MarkdownChunk:
    chunk_index: int
    file_name: str
    doc_title: str
    section_title: str
    content: str
    status: str = "active"

    @property
    def char_count(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict:
        stem = Path(self.file_name).stem
        lang = "vi" if self.file_name.startswith("vi") else "en"
        return {
            "id": f"{stem}_{self.chunk_index:03d}",
            "file_name": self.file_name,
            "doc_title": self.doc_title,
            "section_title": self.section_title,
            "content": self.content,
            "char_count": self.char_count,
            "metadata": {
                "lang": lang,
                "source": "integrix_web",
                "file_stem": stem,
                "status": self.status,
            },
        }


def clean_markdown_text(text: str) -> str:
    """Loại bỏ ký tự in đậm (**), in nghiêng (*, _) ở đầu và cuối chuỗi tiêu đề."""
    return re.sub(r"^[\*_]+|[\*_]+$", "", text.strip()).strip()


def chunk_by_h2(
    text: str,
    file_name: str = "",
    default_title: str = "Tài liệu không có tiêu đề",
    status: str = "active",
) -> Tuple[str, List[MarkdownChunk]]:
    """
    Chia nhỏ Markdown theo ##:
    1. Trích xuất # đầu tiên làm tiêu đề metadata.
    2. Đoạn mở đầu trước H2 đầu tiên tạo thành chunk 'Tổng quan / Overview'.
    3. Mỗi khối H2 tạo một chunk độc lập (tự động gộp các H2 rỗng liên tiếp).
    """
    lines = text.splitlines()
    doc_title = ""
    content_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not doc_title and (m := re.match(r"^#\s+([^#].*)$", stripped)):
            doc_title = clean_markdown_text(m.group(1))
            continue
        content_lines.append(line)

    doc_title = doc_title or default_title
    remaining = "\n".join(content_lines).strip()
    if not remaining:
        return doc_title, []

    parts = re.split(r"^(##\s+[^\n]+)", remaining, flags=re.MULTILINE)
    chunks: List[MarkdownChunk] = []
    chunk_idx = 1

    if intro := parts[0].strip():
        chunks.append(MarkdownChunk(chunk_idx, file_name, doc_title, "Tổng quan / Overview", intro, status=status))
        chunk_idx += 1

    titles, raw_headings = [], []
    for i in range(1, len(parts), 2):
        heading_line = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        titles.append(clean_markdown_text(re.sub(r"^##\s*", "", heading_line)))
        raw_headings.append(heading_line)

        # Nếu chưa có nội dung và phía sau còn H2 khác -> gộp vào tiêu đề tiếp theo
        if not body and (i + 2 < len(parts)):
            continue

        title = " - ".join(titles)
        content = "\n".join(raw_headings) + (f"\n\n{body}" if body else "")
        chunks.append(MarkdownChunk(chunk_idx, file_name, doc_title, title, content, status=status))
        chunk_idx += 1
        titles, raw_headings = [], []

    return doc_title, chunks


def resolve_file(file_arg: str) -> Path:
    """Tìm đường dẫn file markdown phù hợp từ đối số dòng lệnh hoặc tự động quét."""
    p = Path(file_arg)
    if p.is_file():
        return p

    candidates = [
        Path(__file__).parent / "integrix_docs" / p.name,
        Path("integrix_docs") / p.name,
        *Path(".").glob("*.md"),
        *Path(".").glob("**/*.md"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return p


def chunk_directory(target_dir: str = "integrix_docs", default_status: str = "active") -> List[dict]:
    """Quét và chia nhỏ toàn bộ file Markdown trong thư mục thành danh sách dictionary."""
    root = Path(target_dir).resolve()
    if not root.is_dir():
        script_rel = Path(__file__).parent / target_dir
        if script_rel.is_dir():
            root = script_rel.resolve()
        else:
            print(f"[LỖI] Không tìm thấy thư mục: {target_dir}")
            return []

    md_files = sorted(root.glob("*.md"))
    all_chunks = []
    print(f"[*] Đang quét và chia nhỏ {len(md_files)} file Markdown trong '{root.name}' (status: {default_status})...")

    for p in md_files:
        try:
            content = p.read_text(encoding="utf-8")
        except Exception:
            content = p.read_text(encoding="utf-8-sig", errors="ignore")

        doc_title, chunks = chunk_by_h2(content, file_name=p.name, default_title=p.stem, status=default_status)
        for c in chunks:
            all_chunks.append(c.to_dict())

    print(f"[+] Hoàn tất: Tạo thành công {len(all_chunks)} chunks từ {len(md_files)} file.")
    return all_chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chia nhỏ file Markdown theo tiêu đề H2 (##) và giữ H1 (#) làm Metadata."
    )
    parser.add_argument(
        "file_path",
        nargs="?",
        default=None,
        help="Đường dẫn 1 file .md cần kiểm tra (mặc định: 'en_home.md')",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Quét toàn bộ tài liệu trong thư mục và xuất ra file JSON",
    )
    parser.add_argument(
        "--dir",
        default="integrix_docs",
        help="Thư mục chứa tài liệu khi dùng --all (mặc định: 'integrix_docs')",
    )
    parser.add_argument(
        "--output", "-o",
        default="chunks.json",
        help="Đường dẫn file JSON xuất ra khi dùng --all (mặc định: 'chunks.json')",
    )
    parser.add_argument(
        "--status", "-s",
        default="active",
        choices=["active", "archived", "draft"],
        help="Trạng thái hiệu lực của tài liệu ('active', 'archived', 'draft') (mặc định: 'active')",
    )
    args = parser.parse_args()

    # Chế độ 1: Xuất hàng loạt ra JSON
    if args.all:
        chunks_data = chunk_directory(args.dir, default_status=args.status)
        if not chunks_data:
            return

        out_path = Path(args.output)
        if not out_path.is_absolute() and not out_path.parent.name:
            out_path = Path(__file__).parent / args.output

        out_path.write_text(json.dumps(chunks_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[+] Đã lưu toàn bộ dữ liệu vào: {out_path.resolve()}\n")
        return

    # Chế độ 2: Xem trước 1 file cụ thể
    target_file = args.file_path or "en_home.md"
    file_path = resolve_file(target_file)
    if not file_path.is_file():
        print(f"[LỖI] Không tìm thấy file: {target_file}")
        print("Cách chạy:\n  1. Xem trước 1 file: python chunk.py <đường_dẫn_file>\n  2. Xuất toàn bộ ra JSON: python chunk.py --all -o chunks.json")
        return

    content = file_path.read_text(encoding="utf-8")
    doc_title, chunks = chunk_by_h2(content, file_name=file_path.name, default_title=file_path.stem, status=args.status)

    print("=" * 80)
    print(f"📄 TẬP TIN       : {file_path.name}")
    print(f"📌 TIÊU ĐỀ (#)   : {doc_title}")
    print(f"📊 TỔNG DUNG LƯỢNG: {len(content):,} ký tự | SỐ LƯỢNG CHUNK: {len(chunks)} | TRẠNG THÁI: {args.status}")
    print("=" * 80)

    for chunk in chunks:
        print(f"\n--- [CHUNK {chunk.chunk_index}/{len(chunks)}] ({chunk.char_count} ký tự) ---")
        print(f"[METADATA]\n  • Tập tin           : {chunk.file_name}\n  • Doc Title     (#) : {chunk.doc_title}\n  • Section Title (##): {chunk.section_title}\n  • Trạng thái        : {chunk.status}")
        print(f"[NỘI DUNG]\n{chunk.content}\n{'-' * 80}")

    print(f"\n Hoàn tất hiển thị {len(chunks)} chunks của file '{file_path.name}'.\n")


if __name__ == "__main__":
    main()

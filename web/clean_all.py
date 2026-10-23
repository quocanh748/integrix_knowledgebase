#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script làm sạch nội dung các file Markdown (.md), Text (.txt)... của Công ty Integrix:
1. Xóa phần thông tin liên hệ / footer (cả bản Tiếng Việt & Tiếng Anh).
2. Tự động xóa các icon trang trí (💎, 🔹, 🔷, 🌐), in đậm thừa và dấu gạch đầu dòng '-'.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Set, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

DEFAULT_EXTENSIONS = {".md", ".markdown"}
TARGET_ICONS_CLASS = r"[💎🔹🔷🌐]"

FOOTER_START_HEADINGS = {
    "công ty integrix",
    "integrix technologies",
}

TARGET_PHRASES = [
    "CÔNG TY INTEGRIX",
    "INTEGRIX TECHNOLOGIES",
    "THÔNG TIN LIÊN HỆ",
    "CONTACT INFORMATION",
    "CONTACT US",
    "ĐĂNG KÝ THÔNG TIN",
    "STAY CONNECTED",
    "Email: info@integrixtech.com",
    "Tel: +84.8.46834749 hay +84.8.INTEGRIX",
    "Tel: +84.8.46834749 or +84.8.INTEGRIX",
    "© 2026 Integrix Technologies. All rights reserved.",
    "Website: www.integrixtech.com",
    "MST: 0319579945",
    "Tax code: 0319579945",
    "Nền tảng tích hợp ứng dụng AI, giải pháp thông minh, phần mềm phát triển theo yêu cầu và dịch vụ kỹ thuật chuyên nghiệp.",
    "Chúng tôi luôn sát cánh cùng bạn trong quá trình số hóa.",
    "Theo dõi Integrix Technologies trên Facebook và LinkedIn để cập nhật những thông tin mới nhất.",
    "Đăng ký để nhận các cập nhật mới nhất về tích hợp thông minh và công nghệ AI tiên tiến.",
    "AI-Powered Integrated Platforms, Intelligent Solutions, Custom-Built Software, and Professional Services.",
    "We're here to support your digital journey.",
    "Follow Integrix Technologies on Facebook and LinkedIn for the latest updates.",
    "Subscribe to get updates on smart integration and AI innovation.",
]

REGEX_PATTERNS = [
    re.compile(r"^#{1,6}\s*\*?\*?💎?\s*\*?\*?(?:CÔNG TY INTEGRIX|INTEGRIX TECHNOLOGIES|THÔNG TIN LIÊN HỆ|CONTACT INFORMATION|CONTACT US|ĐĂNG KÝ THÔNG TIN|STAY CONNECTED)\*?\*?\s*$", re.I),
    re.compile(r"^Email:\s*(?:\[info@integrixtech\.com\]\([^)]+\)|info@integrixtech\.com)\s*$", re.I),
    re.compile(r"^Tel:\s*\+84\.8\.46834749\s+(?:hay|or)\s+\+84\.8\.INTEGRIX\s*$", re.I),
    re.compile(r"^©\s*2026\s+Integrix\s+Technologies\.\s*All\s+rights\s+reserved\.\s*$", re.I),
    re.compile(r"^Website:\s*(?:\[www\.integrixtech\.com\]\([^)]+\)|(?:https?://)?www\.integrixtech\.com/?)\s*$", re.I),
    re.compile(r"^(?:MST|Tax\s+code):\s*0319579945\s*$", re.I),
]

ICON_BOLD_PATTERNS = [
    (re.compile(rf"\*\*\s*{TARGET_ICONS_CLASS}+\s*\*\*\s*"), ""),
    (re.compile(rf"\*\*\s*{TARGET_ICONS_CLASS}+\s*(.+?)\s*\*\*"), r"\1"),
    (re.compile(rf"{TARGET_ICONS_CLASS}+\s*\*\*\s*(.+?)\s*\*\*"), r"\1"),
    (re.compile(rf"\*\*\s*{TARGET_ICONS_CLASS}+\s*"), ""),
    (re.compile(rf"{TARGET_ICONS_CLASS}+\s*\*\*"), ""),
    (re.compile(rf"{TARGET_ICONS_CLASS}+\s*"), ""),
]


def normalize_markdown_line(line: str) -> str:
    """Loại bỏ ký tự định dạng Markdown, icon và rút gọn link [text](url) -> text."""
    cleaned = line.strip().replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(rf"^[#*_\->\s{TARGET_ICONS_CLASS[1:-1]}]+", "", cleaned)
    cleaned = re.sub(rf"[*_\-#\s{TARGET_ICONS_CLASS[1:-1]}]+$", "", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


NORMALIZED_TARGETS = {normalize_markdown_line(p).lower() for p in TARGET_PHRASES}


def should_remove_line(line: str) -> bool:
    """Kiểm tra dòng có thuộc phần footer Integrix cần xóa hay không."""
    stripped = line.strip()
    if not stripped:
        return False
    if normalize_markdown_line(stripped).lower() in NORMALIZED_TARGETS:
        return True
    return any(rgx.match(stripped) for rgx in REGEX_PATTERNS)


def clean_inline_markers(
    line: str,
    clean_icons: bool = True,
    clean_leading_dash: bool = True,
) -> Tuple[str, bool]:
    """Làm sạch icon trang trí và dấu gạch đầu dòng list ('-')."""
    new_line = line

    def strip_dash(text: str) -> str:
        text = re.sub(r"^(\s*)-\s+", r"\1", text)
        return re.sub(r"^\s*-\s*$", "", text)

    if clean_leading_dash:
        new_line = strip_dash(new_line)

    if clean_icons:
        for pat, repl in ICON_BOLD_PATTERNS:
            new_line = pat.sub(repl, new_line)

    if clean_leading_dash:
        new_line = strip_dash(new_line)

    if new_line != line:
        new_line = new_line.rstrip()

    return new_line, (new_line != line)


def clean_file_content(
    content: str,
    truncate_from_first_heading: bool = False,
    clean_icons: bool = True,
    clean_leading_dash: bool = True,
) -> Tuple[str, int, int]:
    """Xóa dòng footer và làm sạch icon / dấu gạch đầu dòng."""
    lines = content.splitlines()
    new_lines: List[str] = []
    removed_count = reformatted_count = 0

    for line in lines:
        if truncate_from_first_heading and normalize_markdown_line(line).lower() in FOOTER_START_HEADINGS:
            removed_count += len(lines) - len(new_lines)
            break

        if should_remove_line(line):
            removed_count += 1
            continue

        had_text = bool(line.strip())
        cleaned, changed = clean_inline_markers(line, clean_icons, clean_leading_dash)
        if changed:
            reformatted_count += 1
            if had_text and not cleaned.strip():
                continue

        new_lines.append(cleaned)

    if removed_count == 0 and reformatted_count == 0:
        return content, 0, 0

    cleaned_text = re.sub(r"\n{3,}", "\n\n", "\n".join(new_lines)).strip() + "\n"
    return cleaned_text, removed_count, reformatted_count


def process_directory(
    target_dir: str = "integrix_docs",
    extensions: Set[str] = DEFAULT_EXTENSIONS,
    recursive: bool = True,
    truncate_footer: bool = False,
    clean_icons: bool = True,
    clean_leading_dash: bool = True,
    dry_run: bool = False,
) -> None:
    """Quét thư mục và xử lý tất cả các file phù hợp."""
    root_path = Path(target_dir).resolve()
    if not root_path.is_dir():
        script_rel = Path(__file__).parent / target_dir
        if script_rel.is_dir():
            root_path = script_rel.resolve()
        else:
            print(f"[LỖI] Thư mục không tồn tại: {target_dir}")
            return

    files = root_path.rglob("*") if recursive else root_path.glob("*")
    script_path = Path(__file__).resolve()
    scanned = modified = lines_removed = lines_reformatted = 0

    for file_path in files:
        if not file_path.is_file() or file_path.resolve() == script_path:
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue

        scanned += 1
        try:
            raw = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                raw = file_path.read_text(encoding="utf-8-sig")
            except Exception as e:
                print(f"[BỎ QUA] Không đọc được file {file_path.name}: {e}")
                continue
        except Exception as e:
            print(f"[LỖI] Không thể mở file {file_path}: {e}")
            continue

        cleaned, rem, ref = clean_file_content(
            raw,
            truncate_from_first_heading=truncate_footer,
            clean_icons=clean_icons,
            clean_leading_dash=clean_leading_dash,
        )

        if rem > 0 or ref > 0:
            modified += 1
            lines_removed += rem
            lines_reformatted += ref
            rel_path = file_path.relative_to(root_path)
            tag = "[XEM TRƯỚC]" if dry_run else "[ĐÃ XỬ LÝ]"
            verb = "sẽ xóa" if dry_run else "đã xóa"
            print(f"{tag} {rel_path}: {verb} {rem} dòng footer, làm sạch ký tự ở {ref} dòng")
            if not dry_run:
                file_path.write_text(cleaned, encoding="utf-8")

    mode = "CHẾ ĐỘ XEM TRƯỚC (DRY-RUN)" if dry_run else "HOÀN TẤT"
    print("-" * 80)
    print(f"[{mode}] Đã quét: {scanned} file | Đã làm sạch: {modified} file | Đã xóa: {lines_removed} dòng footer | Đã làm sạch ký tự: {lines_reformatted} dòng")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Xóa footer Integrix (VI & EN) và làm sạch các icon 💎, 🔹, 🔷, 🌐, dấu '-' khỏi các file."
    )
    parser.add_argument("directory", nargs="?", default="integrix_docs", help="Thư mục cần xử lý (mặc định: 'integrix_docs')")
    parser.add_argument("--ext", nargs="*", default=list(DEFAULT_EXTENSIONS), help="Đuôi file cần xử lý (mặc định: .md .markdown)")
    parser.add_argument("--no-recursive", action="store_true", help="Chỉ quét thư mục gốc, không quét thư mục con")
    parser.add_argument("--truncate-footer", action="store_true", help="Cắt bỏ toàn bộ nội dung từ tiêu đề footer đến hết file")
    parser.add_argument("--keep-icons", action="store_true", help="Giữ nguyên các icon 💎, 🔹, 🔷, 🌐")
    parser.add_argument("--keep-dash", action="store_true", help="Giữ nguyên dấu gạch đầu dòng '-' ở đầu dòng")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ kiểm tra và in kết quả dự kiến, không ghi đè file")
    args = parser.parse_args()

    exts = set() if args.ext and "all" in [e.lower() for e in args.ext] else {
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.ext
    }

    process_directory(
        target_dir=args.directory,
        extensions=exts,
        recursive=not args.no_recursive,
        truncate_footer=args.truncate_footer,
        clean_icons=not args.keep_icons,
        clean_leading_dash=not args.keep_dash,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()

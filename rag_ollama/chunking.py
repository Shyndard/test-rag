from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    chunk_index: int
    content: str


def _split_by_h2(markdown_text: str) -> list[str]:
    lines = markdown_text.splitlines()
    blocks: list[list[str]] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append(current)

    rendered = ["\n".join(b).strip() for b in blocks]
    return [b for b in rendered if b]


def _split_to_max_chars(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    buf: list[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if chunk:
            parts.append(chunk)
        buf = []
        buf_len = 0

    for p in paragraphs:
        add_len = len(p) + (2 if buf else 0)
        if buf and buf_len + add_len > max_chars:
            flush()

            # overlap from the previous chunk tail
            if overlap_chars > 0 and parts:
                tail = parts[-1][-overlap_chars:]
                if tail.strip():
                    buf = [tail]
                    buf_len = len(tail)

        if len(p) > max_chars:
            # hard-split very long paragraph
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                parts.append(p[start:end].strip())
                start = end - overlap_chars if overlap_chars > 0 else end
            buf = []
            buf_len = 0
        else:
            buf.append(p)
            buf_len += add_len

    flush()
    return [x for x in parts if x]


def chunk_markdown(
    markdown_text: str,
    *,
    source: str,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Chunk markdown using H2 sections, then enforce size limits."""
    h2_blocks = _split_by_h2(markdown_text)
    if not h2_blocks:
        h2_blocks = [markdown_text.strip()]

    chunks: list[Chunk] = []
    idx = 0
    for block in h2_blocks:
        for part in _split_to_max_chars(block, max_chars=max_chars, overlap_chars=overlap_chars):
            cleaned = part.strip()
            if cleaned:
                chunks.append(Chunk(source=source, chunk_index=idx, content=cleaned))
                idx += 1

    return chunks

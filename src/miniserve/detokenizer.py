"""Incremental detokenization for streaming.

Decoding token-by-token is wrong for byte-level BPE / SentencePiece tokenizers: one character
can span several tokens (emoji, CJK), and some tokenizers drop the leading space when a token is
decoded on its own. We use the windowed approach popularised by Hugging Face TGI and vLLM:
decode a small window that includes already-emitted context (``prefix_offset``), emit only the
new suffix, and hold back output while it ends in U+FFFD (an incomplete UTF-8 sequence).
"""

from __future__ import annotations

from miniserve.backends.base import Tokenizer


class IncrementalDetokenizer:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self._tok = tokenizer
        self._ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0

    def add(self, token_id: int) -> str:
        """Append one token; return the newly *stable* text (may be empty)."""
        self._ids.append(token_id)
        prefix_text = self._tok.decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = self._tok.decode(self._ids[self._prefix_offset :])
        if len(new_text) > len(prefix_text) and not new_text.endswith("�"):
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)
            return new_text[len(prefix_text) :]
        return ""

    def flush(self) -> str:
        """Return any held-back text at end of generation (may contain U+FFFD)."""
        if self._read_offset == len(self._ids):
            return ""
        prefix_text = self._tok.decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = self._tok.decode(self._ids[self._prefix_offset :])
        self._prefix_offset = self._read_offset
        self._read_offset = len(self._ids)
        return new_text[len(prefix_text) :]

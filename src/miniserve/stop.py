"""Stop-string handling for streamed text.

A stop string can arrive split across several deltas ("<|end" + "|>"), so text that *might* be
the beginning of a stop string is held back until it is either completed (generation stops and
the stop string is not emitted, matching OpenAI semantics) or ruled out (it is released).
Only the longest suffix that is a proper prefix of some stop string is held, so ordinary text is
not delayed. Invariant: emitted text never contains any part of a stop string that later
completes.
"""

from __future__ import annotations


class StopChecker:
    def __init__(self, stops: tuple[str, ...]) -> None:
        self.stops = tuple(s for s in stops if s)
        self._held = ""

    def process(self, delta: str) -> tuple[str, bool]:
        """Feed newly stable text; return (text safe to emit, whether a stop string matched)."""
        buf = self._held + delta
        cut = min((i for i in (buf.find(s) for s in self.stops) if i != -1), default=-1)
        if cut != -1:
            self._held = ""
            return buf[:cut], True
        hold = 0
        for s in self.stops:
            for k in range(min(len(s) - 1, len(buf)), hold, -1):
                if buf.endswith(s[:k]):
                    hold = k
                    break
        self._held = buf[len(buf) - hold :]
        return buf[: len(buf) - hold], False

    def flush(self) -> str:
        """Release held text when generation ends for another reason."""
        held, self._held = self._held, ""
        return held

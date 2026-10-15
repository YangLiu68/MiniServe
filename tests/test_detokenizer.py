import random

import pytest

from miniserve.backends.toy import ByteTokenizer
from miniserve.detokenizer import IncrementalDetokenizer


def stream(tok, ids):
    d = IncrementalDetokenizer(tok)
    deltas = [d.add(i) for i in ids]
    return deltas, d.flush()


def test_multibyte_characters_are_held_back_until_complete():
    tok = ByteTokenizer()
    deltas, tail = stream(tok, tok.encode("a🙂b"))  # 🙂 is 4 UTF-8 bytes
    assert deltas == ["a", "", "", "", "🙂", "b"]
    assert tail == ""
    assert all("�" not in d for d in deltas)


def test_concatenation_matches_full_decode_for_valid_text():
    tok = ByteTokenizer()
    rng = random.Random(0)
    alphabet = list("abc xyz") + ["é", "ß", "世", "界", "🙂", "👩‍💻", "\n", "Ω"]
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 30)))
        deltas, tail = stream(tok, tok.encode(text))
        assert "".join(deltas) + tail == text


def test_first_token_keeps_its_leading_space_with_a_metaspace_decoder():
    """Regression: SentencePiece-style decoders drop the leading space of the first token of a
    decode call, so "is" + "▁Paris" streamed as "isParis". Seeding the window with the prompt's
    last tokens fixes it."""
    tokenizers = pytest.importorskip("tokenizers")
    vocab = {"[UNK]": 0, "▁The": 1, "▁capital": 2, "▁is": 3, "▁Paris": 4, ",": 5, "▁a": 6}
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    raw.decoder = tokenizers.decoders.Metaspace()

    class Tok:
        eos_token_ids = frozenset()

        def decode(self, ids):
            return raw.decode(ids)

    prompt, output = [1, 2, 3], [4, 5, 6]
    d = IncrementalDetokenizer(Tok(), prompt)
    assert "".join(d.add(i) for i in output) + d.flush() == " Paris, a"
    d = IncrementalDetokenizer(Tok())  # the old behaviour, for contrast
    assert "".join(d.add(i) for i in output) == "Paris, a"


def test_prompt_context_is_never_emitted():
    tok = ByteTokenizer()
    for prompt in ("", "x", "héllo wörld 🙂", "🙂"):
        for text in ("a", "🙂b", " ok"):
            d = IncrementalDetokenizer(tok, tok.encode(prompt))
            assert "".join(d.add(i) for i in tok.encode(text)) + d.flush() == text


def test_truncated_sequence_is_flushed_at_end():
    tok = ByteTokenizer()
    ids = tok.encode("ok🙂")[:-2]  # generation stopped mid-character
    deltas, tail = stream(tok, ids)
    assert "".join(deltas) == "ok"
    assert tail == "�"

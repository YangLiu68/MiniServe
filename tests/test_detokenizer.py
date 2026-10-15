import random

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


def test_truncated_sequence_is_flushed_at_end():
    tok = ByteTokenizer()
    ids = tok.encode("ok🙂")[:-2]  # generation stopped mid-character
    deltas, tail = stream(tok, ids)
    assert "".join(deltas) == "ok"
    assert tail == "�"

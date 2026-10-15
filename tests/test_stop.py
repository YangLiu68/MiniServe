import random

from miniserve.stop import StopChecker


def feed(stops, pieces):
    sc = StopChecker(tuple(stops))
    emitted, stopped = [], False
    for p in pieces:
        out, stopped = sc.process(p)
        emitted.append(out)
        if stopped:
            break
    if not stopped:
        emitted.append(sc.flush())
    return emitted, stopped


def test_stop_split_across_deltas():
    emitted, stopped = feed(["<|end|>"], ["Hello", " <|", "en", "d|>", "never"])
    assert stopped and "".join(emitted) == "Hello "
    assert emitted[1] == " "  # "<|" held back, " " released immediately


def test_false_alarm_is_released():
    emitted, stopped = feed(["STOP"], ["ST", "O", "X done"])
    assert not stopped and "".join(emitted) == "STOX done"
    assert emitted[:2] == ["", ""]


def test_earliest_of_several_stops_wins():
    emitted, stopped = feed(["bb", "a"], ["xxbbya"])
    assert stopped and "".join(emitted) == "xx"


def test_randomized_matches_reference_truncation():
    rng = random.Random(0)
    alphabet = "ab<|>xyz é🙂"
    for _ in range(2000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        stops = [
            "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 4)))
            for _ in range(rng.randint(1, 3))
        ]
        cuts = sorted(rng.sample(range(len(text) + 1), min(len(text) + 1, rng.randint(0, 6))))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)], strict=True)]
        emitted, stopped = feed(stops, pieces)
        hits = [i for i in (text.find(s) for s in stops) if i != -1]
        expected = text[: min(hits)] if hits else text
        assert "".join(emitted) == expected, (text, stops, pieces)
        assert stopped == bool(hits)
        # nothing emitted before the stop ever contains a complete stop string
        assert not any(s in "".join(emitted) for s in stops)

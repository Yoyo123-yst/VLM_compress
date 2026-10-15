"""Token-position rules for the four attack objectives.

B, C, and D all read one frozen clean FastV id sequence. They do not share a
boolean that secretly changes the set. D never falls back to C.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class CoreSpanUnavailable(RuntimeError):
    """Raised when a core-content loss is asked to train without a real span."""


@dataclass(frozen=True)
class Selection:
    name: str
    positions: tuple[int, ...]
    token_ids: tuple[int, ...]
    pieces: tuple[str, ...]
    reason: str
    same_positions_as_noeos: bool
    available: bool


def control_id_set(tokenizer) -> set[int]:
    found: set[int] = set()
    for attr in ("eos_token_id", "pad_token_id", "bos_token_id", "unk_token_id"):
        value = getattr(tokenizer, attr, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            found.update(int(item) for item in value)
        else:
            found.add(int(value))
    specials = getattr(tokenizer, "all_special_ids", None) or []
    found.update(int(item) for item in specials)
    return found


def is_control_token(token_id: int, piece: str, control_ids: set[int]) -> bool:
    if int(token_id) in control_ids:
        return True
    if piece.strip() == "":
        return True
    # Pure control pieces have no letters or digits. Punctuation inside an
    # answer stays in the fixed-noEOS set; EOS/PAD/specials are already ids.
    if piece.startswith("<|") and piece.endswith("|>"):
        return True
    return False


def decode_pieces(tokenizer, token_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(token_id)], skip_special_tokens=False) for token_id in token_ids]


def find_contiguous(reference: list[int], target: list[int]) -> list[int] | None:
    if not target or len(target) > len(reference):
        return None
    width = len(target)
    for start in range(len(reference) - width + 1):
        if reference[start : start + width] == list(target):
            return list(range(start, start + width))
    return None


def find_piece_span(pieces: list[str], phrase: str) -> list[int] | None:
    if not phrase:
        return None
    spans = []
    cursor = 0
    for index, piece in enumerate(pieces):
        spans.append((cursor, cursor + len(piece), index))
        cursor += len(piece)
    text = "".join(pieces)
    start = text.find(phrase)
    if start < 0:
        start = text.lower().find(phrase.lower())
    if start < 0:
        return None
    end = start + len(phrase)
    positions = [index for left, right, index in spans if not (right <= start or left >= end)]
    return positions or None


def match_core_phrase(tokenizer, reference: list[int], pieces: list[str], phrase: str) -> tuple[list[int], str] | None:
    target = tokenizer(phrase, add_special_tokens=False)
    raw = target["input_ids"] if isinstance(target, dict) else target.input_ids
    if raw and isinstance(raw[0], list):
        raw = raw[0]
    target_ids = [int(item) for item in raw]
    contiguous = find_contiguous(reference, target_ids)
    if contiguous is not None:
        return contiguous, "token_id_contiguous"
    located = find_piece_span(pieces, phrase)
    if located is not None:
        return located, "piece_char_span"
    return None


def choose_core_span(tokenizer, reference: list[int], phrases: list[str]) -> tuple[list[int], str, str] | None:
    """Longest registered phrase that actually occurs. No non-EOS fallback."""
    pieces = decode_pieces(tokenizer, reference)
    ranked = sorted({phrase.strip() for phrase in phrases if phrase and phrase.strip()}, key=len, reverse=True)
    for phrase in ranked:
        matched = match_core_phrase(tokenizer, reference, pieces, phrase)
        if matched is None:
            continue
        positions, how = matched
        if not positions:
            continue
        return positions, phrase, how
    return None


def fixed_all_selection(tokenizer, reference: list[int]) -> Selection:
    pieces = decode_pieces(tokenizer, reference)
    positions = tuple(range(len(reference)))
    return Selection(
        name="fixed_all",
        positions=positions,
        token_ids=tuple(int(token) for token in reference),
        pieces=tuple(pieces),
        reason="frozen_clean_fastv_all_tokens_including_eos",
        same_positions_as_noeos=False,
        available=True,
    )


def fixed_noeos_selection(tokenizer, reference: list[int], control_ids: set[int] | None = None) -> Selection:
    pieces = decode_pieces(tokenizer, reference)
    controls = control_ids if control_ids is not None else control_id_set(tokenizer)
    positions = [index for index, (token, piece) in enumerate(zip(reference, pieces)) if not is_control_token(token, piece, controls)]
    if not positions:
        raise RuntimeError("fixed-noEOS removed every token")
    return Selection(
        name="fixed_noeos",
        positions=tuple(positions),
        token_ids=tuple(int(reference[index]) for index in positions),
        pieces=tuple(pieces[index] for index in positions),
        reason="frozen_clean_fastv_excluding_eos_pad_and_control",
        same_positions_as_noeos=False,
        available=True,
    )


def core_content_selection(tokenizer, reference: list[int], phrases: list[str], noeos_positions: tuple[int, ...]) -> Selection:
    pieces = decode_pieces(tokenizer, reference)
    matched = choose_core_span(tokenizer, reference, phrases)
    if matched is None:
        return Selection(
            name="core_content",
            positions=tuple(),
            token_ids=tuple(),
            pieces=tuple(),
            reason="CORE_SPAN_UNAVAILABLE",
            same_positions_as_noeos=False,
            available=False,
        )
    positions, phrase, how = matched
    # A registered span that merely reprints every non-EOS token is recorded,
    # not rewritten. Callers must not treat that equality as independent evidence.
    same = tuple(positions) == tuple(noeos_positions)
    return Selection(
        name="core_content",
        positions=tuple(positions),
        token_ids=tuple(int(reference[index]) for index in positions),
        pieces=tuple(pieces[index] for index in positions),
        reason=f"preregistered_core_phrase:{phrase}:{how}",
        same_positions_as_noeos=same,
        available=True,
    )


def dynamic_sequence_selection(tokenizer, generated: list[int]) -> Selection:
    pieces = decode_pieces(tokenizer, generated)
    return Selection(
        name="dynamic_sequence",
        positions=tuple(range(len(generated))),
        token_ids=tuple(int(token) for token in generated),
        pieces=tuple(pieces),
        reason="current_perturbed_natural_fastv_sequence",
        same_positions_as_noeos=False,
        available=bool(generated),
    )


def assert_core_not_silent_noeos(core: Selection, noeos: Selection) -> None:
    if core.available:
        return
    if core.positions:
        raise RuntimeError("unavailable core span still carried positions")
    if core.positions == noeos.positions and noeos.positions:
        raise RuntimeError("core span silently reused the noEOS set")


_ALNUM = re.compile(r"[A-Za-z0-9]")


def content_piece(piece: str) -> bool:
    return bool(_ALNUM.search(piece))

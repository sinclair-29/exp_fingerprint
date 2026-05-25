from __future__ import annotations

import re
import string
from dataclasses import dataclass

import torch


def _is_ascii_printable(text: str) -> bool:
    return all(ch in string.printable and ch not in "\x0b\x0c" for ch in text)


class AllowAllFilter:
    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        return ids


class AsciiFilter:
    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        keep = []
        for row in ids:
            if _is_ascii_printable(tokenizer.decode(row.tolist(), skip_special_tokens=False)):
                keep.append(row)
        return torch.stack(keep) if keep else ids[:0]


class RetokenizationConsistencyFilter:
    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        keep = []
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=False)
        for row, text in zip(ids, decoded):
            retok = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(ids.device)
            if torch.equal(row, retok):
                keep.append(row)
        return torch.stack(keep) if keep else ids[:0]


class TrapNumberFilter:
    NUMBER_WORDS = {
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
        "seventeen", "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty",
        "sixty", "seventy", "eighty", "ninety", "hundred", "thousand", "million",
        "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
        "ninth", "tenth", "jan", "january", "feb", "february", "mar", "march", "apr",
        "april", "may", "jun", "june", "jul", "july", "aug", "august", "sep",
        "sept", "september", "oct", "october", "nov", "november", "dec", "december",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "mon", "tue", "wed", "thu", "fri", "sat", "sun", "number", "digit", "digits",
        "numeric", "integer", "roman", "iv", "v", "vi", "vii", "viii", "ix", "x",
    }

    def __init__(self, tokenizer):
        self.disallowed_ids = self._build_disallowed(tokenizer)

    def _build_disallowed(self, tokenizer) -> set[int]:
        disallowed = set(tokenizer.all_special_ids or [])
        for token_id in range(getattr(tokenizer, "vocab_size", len(tokenizer))):
            text = tokenizer.decode([token_id], skip_special_tokens=False)
            compact = re.sub(r"[^a-z0-9]+", "", text.casefold())
            if not compact:
                continue
            if any(ch.isdigit() for ch in compact):
                disallowed.add(token_id)
            if compact in self.NUMBER_WORDS:
                disallowed.add(token_id)
            if re.fullmatch(r"[ivxlcdm]+", compact):
                disallowed.add(token_id)
        return disallowed

    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        if not self.disallowed_ids:
            return ids
        bad = torch.tensor(sorted(self.disallowed_ids), device=ids.device, dtype=ids.dtype)
        keep_mask = ~torch.isin(ids, bad).any(dim=1)
        return ids[keep_mask]


class ProFLingoWordFragmentFilter:
    def _token_ok(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if not _is_ascii_printable(text):
            return False
        if any(ch in text for ch in "\n\r\t"):
            return False
        return bool(re.fullmatch(r"[A-Za-z][A-Za-z_\-]{0,18}", stripped))

    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        keep = []
        for row in ids:
            if all(self._token_ok(tokenizer.decode([int(token_id)], skip_special_tokens=False)) for token_id in row):
                keep.append(row)
        return torch.stack(keep) if keep else ids[:0]


@dataclass
class TargetKeywordExclusionFilter:
    target_keywords: list[str]

    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        keywords = [keyword.casefold() for keyword in self.target_keywords if keyword]
        if not keywords:
            return ids
        keep = []
        for row in ids:
            text = tokenizer.decode(row.tolist(), skip_special_tokens=False).casefold()
            if not any(keyword in text for keyword in keywords):
                keep.append(row)
        return torch.stack(keep) if keep else ids[:0]


class CompositeFilter:
    def __init__(self, *filters):
        self.filters = [flt for flt in filters if flt is not None]

    def filter_ids(self, ids: torch.Tensor, tokenizer) -> torch.Tensor:
        out = ids
        for flt in self.filters:
            if out.numel() == 0:
                return out
            out = flt.filter_ids(out, tokenizer)
        return out

"""Convert common LaTeX math into readable Unicode.

The local LLM frequently emits LaTeX (``\\sqrt{\\frac{1}{n}\\sum_i ...}``), often
without ``$`` delimiters. Terminals can't render LaTeX and the Chainlit UI only
renders it when ``latex`` is enabled *and* the math is delimited. Rewriting to
Unicode (``√((1)/(n)Σᵢ ...)``) makes formulas readable on every surface without
relying on a math renderer.

This is a best-effort converter scoped to the math that shows up in AI/ML course
material (sums, fractions, roots, sub/superscripts, hats, Greek). It is not a
full LaTeX engine — anything it can't map is left as readable plain text.
"""
from __future__ import annotations

import re

_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "rho": "ρ", "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

_SYMBOLS = {
    "cdot": "·", "times": "×", "div": "÷", "pm": "±", "mp": "∓",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "propto": "∝", "sim": "∼", "ll": "≪", "gg": "≫",
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "Σ", "prod": "Π",
    "int": "∫", "in": "∈", "notin": "∉", "subset": "⊂", "subseteq": "⊆",
    "forall": "∀", "exists": "∃", "rightarrow": "→", "to": "→", "gets": "←",
    "leftarrow": "←", "Rightarrow": "⇒", "Leftarrow": "⇐", "mapsto": "↦",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "angle": "∠", "perp": "⊥",
    "ldots": "…", "dots": "…", "cdots": "⋯", "odot": "⊙", "otimes": "⊗",
    "oplus": "⊕", "star": "⋆", "circ": "∘", "bullet": "•", "prime": "′",
}

# Accent commands → Unicode combining mark (placed after the base character).
_ACCENTS = {
    "hat": "̂", "bar": "̅", "vec": "⃗", "tilde": "̃",
    "dot": "̇", "ddot": "̈", "overline": "̅",
}

# Font/wrapper commands whose single argument should pass through unchanged.
_PASSTHROUGH = {
    "text", "mathrm", "mathbf", "mathit", "mathcal", "mathbb", "boldsymbol",
    "operatorname", "mathsf", "mathtt", "textbf", "textit", "rm", "bf",
}

_SUP = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶",
    "7": "⁷", "8": "⁸", "9": "⁹", "+": "⁺", "-": "⁻", "=": "⁼", "(": "⁽",
    ")": "⁾", "n": "ⁿ", "i": "ⁱ", "T": "ᵀ",
}
_SUB = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄", "5": "₅", "6": "₆",
    "7": "₇", "8": "₈", "9": "₉", "+": "₊", "-": "₋", "=": "₌", "(": "₍",
    ")": "₎", "a": "ₐ", "e": "ₑ", "i": "ᵢ", "j": "ⱼ", "k": "ₖ", "l": "ₗ",
    "m": "ₘ", "n": "ₙ", "o": "ₒ", "p": "ₚ", "r": "ᵣ", "s": "ₛ", "t": "ₜ",
    "u": "ᵤ", "v": "ᵥ", "x": "ₓ",
}

# Word-like macros to strip (lookahead avoids eating e.g. \leftarrow → arrow).
_STRIP_WORDS = (
    "left|right|bigl|bigr|big|displaystyle|limits|nolimits|quad|qquad"
)
# Spacing macros (non-letter) to strip literally.
_STRIP_SPACING = [r"\,", r"\;", r"\!", r"\:"]


def _read_arg(s: str, i: int) -> tuple[str, int]:
    """Read a command argument starting at index ``i``.

    Returns ``(raw_argument, next_index)``. A ``{...}`` group returns its
    balanced contents; a ``\\command`` returns that token; otherwise a single
    character. Leading spaces are skipped.
    """
    n = len(s)
    while i < n and s[i] == " ":
        i += 1
    if i >= n:
        return "", i
    if s[i] == "{":
        depth, j = 1, i + 1
        while j < n and depth:
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
            j += 1
        return s[i + 1 : j - 1], j
    if s[i] == "\\":
        m = re.match(r"\\([a-zA-Z]+|.)", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _to_script(raw: str, table: dict[str, str]) -> str:
    """Map every char of ``raw`` to its sub/superscript form, or fall back."""
    converted = _convert(raw)
    if converted and all(ch in table for ch in converted):
        return "".join(table[ch] for ch in converted)
    marker = "^" if table is _SUP else "_"
    return f"{marker}({converted})" if len(converted) != 1 else f"{marker}{converted}"


def _convert(s: str) -> str:
    out: list[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if not m:  # escaped symbol like \{ \% \_ \\
                out.append(s[i + 1] if i + 1 < n else "")
                i += 2
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in ("frac", "dfrac", "tfrac"):
                a, i = _read_arg(s, i)
                b, i = _read_arg(s, i)
                out.append(f"({_convert(a)})/({_convert(b)})")
            elif name == "sqrt":
                if i < n and s[i] == "[":  # optional root index \sqrt[3]{x}
                    j = s.find("]", i)
                    idx = s[i + 1 : j] if j != -1 else ""
                    i = j + 1 if j != -1 else i
                    out.append(_to_script(idx, _SUP))
                a, i = _read_arg(s, i)
                out.append(f"√({_convert(a)})")
            elif name in _ACCENTS:
                a, i = _read_arg(s, i)
                base = _convert(a)
                out.append(base + _ACCENTS[name] if base else _ACCENTS[name])
            elif name in _PASSTHROUGH:
                a, i = _read_arg(s, i)
                out.append(_convert(a))
            elif name in _GREEK:
                out.append(_GREEK[name])
            elif name in _SYMBOLS:
                out.append(_SYMBOLS[name])
            else:
                out.append(name)  # unknown command: drop backslash, keep name
        elif c == "^":
            raw, i = _read_arg(s, i + 1)
            out.append(_to_script(raw, _SUP))
        elif c == "_":
            raw, i = _read_arg(s, i + 1)
            out.append(_to_script(raw, _SUB))
        elif c in "{}":
            i += 1  # drop stray grouping braces
        else:
            out.append(c)
            i += 1
    return "".join(out)


def latex_to_unicode(text: str) -> str:
    """Render LaTeX math fragments in ``text`` as readable Unicode.

    Handles both ``$...$`` / ``\\(...\\)`` delimited math and bare LaTeX commands
    (which the local model often emits undelimited). Non-math text is untouched.
    """
    if "\\" not in text and "^" not in text and "_" not in text and "$" not in text:
        return text
    text = re.sub(rf"\\(?:{_STRIP_WORDS})(?![a-zA-Z])", "", text)
    for token in _STRIP_SPACING:
        text = text.replace(token, "")
    # Strip math delimiters but keep their contents.
    text = re.sub(r"\$\$(.+?)\$\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\$(.+?)\$", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.+?)\\\)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.+?)\\\]", r"\1", text, flags=re.DOTALL)
    converted = _convert(text)
    return re.sub(r"[ \t]{2,}", " ", converted)

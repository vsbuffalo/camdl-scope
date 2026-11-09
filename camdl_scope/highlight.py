"""Syntax highlighting for camdl model files and fit TOML.

A Pygments ``RegexLexer`` for the camdl DSL, with token lists ported from the
authoritative camdl sources — the tree-sitter ``queries/highlights.scm`` and the
skylighting ``camdl.xml`` used by the Quarto book. It is regex-based, so it
won't make the parser's context distinctions (a compartment name vs a plain
variable), but it colors the parts a reader scans for: block keywords, types,
built-in functions, distributions, comments, strings, numbers, and operators.

Pygments handles the fit TOML natively. Both render to inline HTML; the CSS is
emitted once via :data:`HIGHLIGHT_CSS`.
"""

from __future__ import annotations

from pygments import highlight as _highlight
from pygments.formatters import HtmlFormatter
from pygments.lexer import RegexLexer, words
from pygments.lexers import get_lexer_by_name
from pygments.token import (
    Comment, Keyword, Name, Number, Operator, Punctuation, String, Text,
    Whitespace,
)

# ── token lists (merged from highlights.scm + camdl.xml) ────────────────────
_BLOCK_KW = (
    "time_unit", "description", "origin", "dimensions", "compartments",
    "parameters", "tables", "functions", "forcing", "transitions",
    "observations", "interventions", "events", "ode", "output", "simulate",
    "init", "timepoints", "stratify", "let", "scenarios", "balance",
)
_CLAUSE_KW = (
    "from", "to", "where", "in", "by", "values", "only", "at", "at_day",
    "every", "until", "tag", "transfer", "add", "consecutive", "extends",
    "set", "scale", "enable", "disable", "compose", "label", "likelihood",
    "format",
)
_COND_KW = ("if", "then", "else")
_OP_WORD = ("and", "or", "not")
_TYPES = ("real", "integer", "rate", "probability", "positive", "count")
_BUILTINS = (
    "sum", "date", "add_calendar_days", "add_calendar_weeks",
    "add_calendar_months", "add_calendar_years", "date_range", "read",
    "read_levels", "read_long", "defines", "incidence", "cumulative",
    "prevalence", "overdispersed", "deterministic", "exp", "log", "min",
    "max", "mod", "abs", "sqrt", "floor", "ceil", "round", "sinusoidal",
    "piecewise", "interpolated", "periodic",
)
_DISTS = (
    "poisson", "neg_binomial", "normal", "binomial", "beta_binomial",
    "bernoulli", "log_normal", "half_normal", "beta", "gamma", "exponential",
    "uniform", "diagnostic_test",
)


class CamdlLexer(RegexLexer):
    name = "camdl"
    aliases = ["camdl"]
    filenames = ["*.camdl"]

    tokens = {
        "root": [
            (r"\s+", Whitespace),
            (r"#\[", Name.Decorator, "attribute"),     # #[lineage] attribute
            (r"#.*$", Comment.Single),                  # # line comment
            (r'"[^"]*"', String.Double),
            (r"'[A-Za-z_]\w*", Name.Attribute),         # 'days unit literal
            (words(_COND_KW, suffix=r"\b"), Keyword),
            (words(_BLOCK_KW, suffix=r"\b"), Keyword),
            (words(_CLAUSE_KW, suffix=r"\b"), Keyword),
            (words(_OP_WORD, suffix=r"\b"), Operator.Word),
            (words(_TYPES, suffix=r"\b"), Keyword.Type),
            (words(_DISTS, suffix=r"\b"), Name.Builtin),
            (words(_BUILTINS, suffix=r"\b"), Name.Builtin),
            (r"null\b", Keyword.Constant),
            (r"-->|==|!=|<=|>=|[-+*/×^~@<>=]", Operator),
            (r"[{}\[\]()]", Punctuation),
            (r"[,:]", Punctuation),
            (r"\d+\.\d+(?:[eE][+-]?\d+)?|\.\d+|\d+(?:[eE][+-]?\d+)?", Number),
            (r"[A-Za-z_]\w*", Name),
            (r".", Text),
        ],
        "attribute": [
            (r"[^\]]+", Name.Decorator),
            (r"\]", Name.Decorator, "#pop"),
        ],
    }


# A light, readable style to sit on the white dashboard.
_FORMATTER = HtmlFormatter(style="friendly", cssclass="codehl")
HIGHLIGHT_CSS = _FORMATTER.get_style_defs(".codehl")

try:
    _TOML_LEXER = get_lexer_by_name("toml")
except Exception:  # pragma: no cover - alias should always exist
    _TOML_LEXER = get_lexer_by_name("ini")


def highlight_camdl(text: str) -> str:
    """Render camdl source to inline highlighted HTML (content is escaped)."""
    return _highlight(text, CamdlLexer(), _FORMATTER)


def highlight_toml(text: str) -> str:
    """Render TOML to inline highlighted HTML (content is escaped)."""
    return _highlight(text, _TOML_LEXER, _FORMATTER)

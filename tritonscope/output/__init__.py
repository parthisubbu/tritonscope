"""Output formatters."""
from tritonscope.output.json_formatter import JSONFormatter

try:
    from tritonscope.output.terminal_formatter import TerminalFormatter
except ImportError:
    # rich not installed; optional dependency
    TerminalFormatter = None

try:
    from tritonscope.output.markdown_formatter import MarkdownFormatter
except ImportError:
    MarkdownFormatter = None

__all__ = ["JSONFormatter", "TerminalFormatter", "MarkdownFormatter"]

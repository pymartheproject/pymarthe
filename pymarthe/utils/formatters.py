"""
formatters.py — Utility functions for formatting filenames and numeric/string data.
"""

# =========================
# === CONSTANT FORMATTERS ===
# =========================

# ---- ZPC formatters ---- #
ZPCFMT = "{0}_zpc_l{1:02d}_z{2:03d}"
ZPCFMT_LITE = "{0}zpc{1:02d}z{2:03d}"

# ---- Pilot Point formatters ---- #
PPFMT = "{0}_l{1:02d}_z{2:02d}_{3:03d}"
PPFMT_LITE = "{0}{1:02d}z{2:02d}p{3:03d}"


# =========================
# === ZPC FORMATTERS ===
# =========================

def zpc_fmt(name: str, lay: int, zone: int) -> str:
    """Format a standard ZPC name."""
    return ZPCFMT.format(name, int(lay) + 1, abs(int(zone)))


def zpc_fmt_lite(name: str, lay: int, zone: int) -> str:
    """Format a compact ZPC name (≤12 characters), used for PESTHP."""
    return ZPCFMT_LITE.format(name, int(lay) + 1, abs(int(zone)))


# =========================
# === PILOT POINT FORMATTERS ===
# =========================

def pp_fmt(name: str, lay: int, zone: int, ppid: int) -> str:
    """Format a pilot point name with underscores."""
    return PPFMT.format(name, int(lay) + 1, int(zone), int(ppid))


def pp_fmt_lite(name: str, lay: int, zone: int, ppid: int) -> str:
    """Format a compact pilot point name (no underscores)."""
    return PPFMT_LITE.format(name, int(lay) + 1, int(zone), int(ppid))


def input_file_fmt(name: str, lay: int, zone: int, ext: str, fmt_lite: bool = False) -> str:
    """
    Format kriging factor file name.
    """
    if fmt_lite:
        return f"{name}{lay+1:02d}z{zone:02d}{ext}"
    else:
        return f"{name}_pp_l{lay+1:02d}_z{zone:02d}{ext}"


# =========================
# === NUMERIC FORMATTERS ===
# =========================

def float_fmt(x: float) -> str:
    """Format a float in scientific notation (10 decimals, width = 20, left-aligned)."""
    return f"{float(x):<20.10E} "


def int_fmt(x: int) -> str:
    """Format an integer with width = 10, left-aligned."""
    return f"{int(x):<10d} "


# =========================
# === STRING FORMATTER ===
# =========================

def str_fmt(item) -> str:
    """
    Format a string into a fixed-width (20 characters, left-aligned) field.
    Accepts both bytes and str; any other type is converted to str.
    """
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for non-UTF8 encoded bytes
            item = item.decode("latin-1", errors="replace")
    return f"{str(item):<20s} "

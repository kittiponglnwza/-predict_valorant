"""
ui/utils.py — Shared utility functions for Football AI Nexus Engine
"""
import io
import contextlib
import pandas as pd


def silent(fn, *args, **kwargs):
    """Run a function while suppressing stdout output."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = fn(*args, **kwargs)
    return result


def make_styled_table(display_df, pts_col, max_pts):
    """Shared Pandas Styler for league table display."""
    def _row_bg(row):
        p = row.name
        if p == 1:    bg = 'rgba(255,215,0,0.07)'
        elif p <= 4:  bg = 'rgba(0,176,255,0.06)'
        elif p <= 6:  bg = 'rgba(249,115,22,0.06)'
        elif p <= 7:  bg = 'rgba(168,85,247,0.05)'
        elif p >= 18: bg = 'rgba(239,68,68,0.07)'
        else:         bg = 'transparent'
        return [f'background-color:{bg}'] * len(row)

    def _color_zone(val):
        m = {'👑': '#FFD700', '⚽': '#00B0FF', '🌍': '#F97316', '🏅': '#A855F7', '🔻': '#EF4444'}
        for emoji, color in m.items():
            if emoji in str(val):
                return f'color:{color};font-weight:600'
        return 'color:#475569'

    styled = (
        display_df.style
        .apply(_row_bg, axis=1)
        .applymap(_color_zone, subset=['Zone'])
        .bar(subset=[pts_col], color='rgba(0,176,255,0.2)', vmin=0, vmax=max_pts)
    )
    return styled


def zone_label(pos):
    """Return zone label emoji + text for a league position."""
    if pos == 1:   return "👑 Champion"
    if pos <= 4:   return "⚽ Champions League"
    if pos <= 6:   return "🌍 Europa League"
    if pos <= 7:   return "🏅 Conference"
    if pos >= 18:  return "🔻 Relegation"
    return "➖ Mid-table"


def find_team_col(df):
    """Find the column in a DataFrame that contains team names."""
    for c in ['Team', 'team', 'index', 'Club', 'club', 'HomeTeam']:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return df.columns[0]
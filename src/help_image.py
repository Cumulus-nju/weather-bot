"""Help image generator — renders command reference as a clean PNG."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime

from config import OUTPUT_DIR

# Chinese font
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# Colours
C_HEADER = "#1a5276"
C_SECTION = "#2e86c1"
C_BODY = "#212f3d"
C_ACCENT = "#e74c3c"
C_BG = "#fdfefe"
C_BORDER = "#d5dbdb"


def _section(ax, y, title: str, items: list[tuple[str, str]], color: str = C_SECTION):
    """Draw a command section at vertical position y.

    *items*: list of (command, description) tuples.
    Returns the new y position (bottom of section).
    """
    row_h = 1.05
    header_h = 1.4
    pad = 0.4

    # Section background
    n_rows = len(items)
    total_h = header_h + n_rows * row_h + pad * 2
    rect = mpatches.FancyBboxPatch(
        (0.15, y - total_h), 7.7, total_h,
        boxstyle="round,pad=0.15", facecolor="#f0f3f5",
        edgecolor=C_BORDER, linewidth=0.5, zorder=0
    )
    ax.add_patch(rect)

    # Section title
    ax.text(0.5, y - header_h + 0.35, title, fontsize=11, fontweight="bold",
            color=color, va="center", ha="left")

    # Items
    y_pos = y - header_h - pad
    for cmd, desc in items:
        ax.text(0.8, y_pos, cmd, fontsize=9, fontweight="bold",
                color=C_HEADER, va="center", ha="left")
        ax.text(3.5, y_pos, desc, fontsize=9, color=C_BODY,
                va="center", ha="left")
        y_pos -= row_h

    return y - total_h - 0.25


def generate_help_image() -> str:
    """Generate the help reference image. Returns path to PNG."""
    fig, ax = plt.subplots(figsize=(8, 12), dpi=150)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 12)
    ax.axis("off")
    fig.patch.set_facecolor(C_BG)

    y = 11.5

    # ── Title ──
    ax.text(4, y, "Cumulus 天气绘图机器人", fontsize=16, fontweight="bold",
            color=C_HEADER, ha="center", va="center")
    y -= 0.8
    ax.text(4, y, "长三角站点观测 · ECMWF IFS / GFS 数值预报", fontsize=9,
            color="#5d6d7e", ha="center", va="center")
    y -= 1.0

    # ── Real-time section ──
    y = _section(ax, y, "实时站点观测 — 长三角 (116-123°E, 27-35°N)", [
        ("/温度", "2m温度分析 + 站点标注"),
        ("/降水", "降水量分布图"),
        ("/风场", "10m风场 (含风羽)"),
        ("/气压", "海平面气压分析"),
        ("/湿度", "相对湿度分析"),
        ("/综合", "四合一综合大图"),
    ])

    # ── ECMWF section ──
    y = _section(ax, y, "ECMWF IFS 数值预报 — 全中国 0.25°", [
        ("/EC 温度", "2m温度 [分析场/预报]"),
        ("/EC 降水", "累计降水 [默认24h预报]"),
        ("/EC 风场", "10m风场 (含风羽)"),
        ("/EC 气压", "海平面气压"),
        ("/EC 湿度", "相对湿度"),
        ("/EC 综合", "四合一综合分析"),
        ("/EC h500", "500hPa 位势高度"),
        ("/EC t850", "850hPa 温度"),
        ("/EC 高空风场", "850hPa 风场"),
    ])

    # ── GFS section ──
    y = _section(ax, y, "GFS 数值预报 — 全中国 0.25°", [
        ("/GFS 温度", "2m温度 [分析场/预报]"),
        ("/GFS 降水", "累计降水 [默认24h预报]"),
        ("/GFS 风场", "10m风场 (含风羽)"),
        ("/GFS 气压", "海平面气压"),
        ("/GFS 湿度", "相对湿度"),
        ("/GFS 综合", "四合一综合分析"),
        ("/GFS h500", "500hPa 位势高度"),
        ("/GFS t850", "850hPa 温度"),
        ("/GFS 高空风场", "850hPa 风场"),
    ])

    # ── Data management section ──
    y = _section(ax, y, "数据管理", [
        ("/预报", "天气形势分析 + 苏南三天预报"),
        ("/数据更新", "手动下载最新NWP数据"),
        ("/帮助", "显示本帮助图"),
    ], color="#1e8449")

    # ── Footer ──
    y -= 0.3
    ax.text(4, y, "时效说明: 不填=分析场(降水默认+24h)  0/24/48/72=预报",
            fontsize=8, color="#85929e", ha="center")
    y -= 0.5
    ax.text(4, y, "数据源: QWeather · ECMWF IFS · NOAA GFS  |  南京大学气象爱好者",
            fontsize=7.5, color="#aeb6bf", ha="center")

    # Save
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"help_{int(datetime.now().timestamp())}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    return str(path)

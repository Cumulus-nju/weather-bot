"""weather-bot — QQ group weather mapping bot for Yangtze River Delta.

Usage:
  python main.py

Requires NapCatQQ running with reverse WebSocket pointing to ws://127.0.0.1:8080/onebot/v11/ws
"""

import sys
from pathlib import Path

# Ensure the project root is on sys.path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import nonebot
from nonebot.adapters.onebot.v11 import Adapter as OneBotV11Adapter

# ---------------------------------------------------------------------------
# nonebot2 init
# ---------------------------------------------------------------------------

nonebot.init(
    command_start={"/", "。"},     # support both /cmd and 。cmd
    command_sep={" ", "\n"},       # split args by space or newline
    superusers={""},               # set your QQ number for admin commands
)

# Register the OneBot V11 adapter (used by NapCatQQ)
driver = nonebot.get_driver()
driver.register_adapter(OneBotV11Adapter)

# Load our bot plugin
nonebot.load_plugins("bot")

# ---------------------------------------------------------------------------
# Startup health check
# ---------------------------------------------------------------------------

@driver.on_startup
async def on_startup():
    from src.pipeline import get_pipeline
    from config import STATION_DATA_SOURCE, DEFAULT_EXTENT

    print("=" * 50)
    print("  weather-bot — 长三角天气绘图机器人")
    print(f"  数据源: {STATION_DATA_SOURCE}")
    print(f"  区域: {DEFAULT_EXTENT}")
    print(f"  等待 NapCatQQ WebSocket 连接...")
    print("=" * 50)

    # Pre-warm: build the grid
    _ = get_pipeline()
    print("  Pipeline 初始化完成，可以接收命令。")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nonebot.run()

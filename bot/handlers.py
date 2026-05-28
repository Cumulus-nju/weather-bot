"""nonebot2 command handlers for weather-bot.

Responds to QQ group commands:
  /温度  /降水  /风场  /气压  /湿度  /综合  /帮助
"""

import asyncio
import logging
import traceback

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, Message, MessageSegment
from nonebot.matcher import Matcher
from nonebot.params import CommandArg

from src.pipeline import get_pipeline

logger = logging.getLogger("weather-bot.handlers")

# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

temperature   = on_command("温度", aliases={"气温"}, priority=5, block=True)
precipitation = on_command("降水", aliases={"降雨", "雨量", "下雨"}, priority=5, block=True)
wind          = on_command("风场", aliases={"风", "风速", "风向"}, priority=5, block=True)
pressure      = on_command("气压", priority=5, block=True)
humidity      = on_command("湿度", priority=5, block=True)
comprehensive = on_command("综合", aliases={"天气", "全部"}, priority=5, block=True)
help_cmd      = on_command("帮助", aliases={"help", "菜单", "说明"}, priority=10, block=True)

# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------

VAR_MAP = {
    "温度": "temperature",
    "气温": "temperature",
    "降水": "precipitation",
    "降雨": "precipitation",
    "雨量": "precipitation",
    "下雨": "precipitation",
    "风场": "wind",
    "风": "wind",
    "风速": "wind",
    "风向": "wind",
    "气压": "pressure",
    "湿度": "humidity",
    "综合": "comprehensive",
    "天气": "comprehensive",
    "全部": "comprehensive",
}

# ---------------------------------------------------------------------------
# Shared pipeline runner
# ---------------------------------------------------------------------------

_semaphore = asyncio.Semaphore(4)  # max concurrent pipeline runs


async def run_pipeline_and_reply(
    bot: Bot, event: GroupMessageEvent, variable: str, display_name: str
):
    """Run the weather pipeline in a thread and send the image back."""
    async with _semaphore:
        # Acknowledge
        await bot.send(event, f"正在生成{display_name}图，请稍候...")

        try:
            pipeline = get_pipeline()
            path = await asyncio.to_thread(pipeline.generate, variable)

            # Send image — use file:/// for Windows absolute path
            import pathlib
            abs_path = pathlib.Path(path).resolve().as_posix()
            msg = (
                MessageSegment.text(f"【{display_name}】长三角实时分析\n")
                + MessageSegment.image(f"file:///{abs_path}")
            )
            await bot.send(event, msg)

        except Exception as e:
            logger.error(f"Pipeline failed for {variable}: {traceback.format_exc()}")
            err_msg = str(e)[:200] if str(e) else "未知错误"
            await bot.send(event, f"生成失败: {err_msg}\n请稍后再试或联系管理员。")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

@temperature.handle()
async def handle_temperature(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "temperature", "温度")


@precipitation.handle()
async def handle_precipitation(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "precipitation", "降水")


@wind.handle()
async def handle_wind(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "wind", "风场")


@pressure.handle()
async def handle_pressure(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "pressure", "气压")


@humidity.handle()
async def handle_humidity(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "humidity", "湿度")


@comprehensive.handle()
async def handle_comprehensive(bot: Bot, event: GroupMessageEvent):
    await run_pipeline_and_reply(bot, event, "comprehensive", "综合")


@help_cmd.handle()
async def handle_help(bot: Bot, event: GroupMessageEvent):
    await bot.send(event,
                   "Cumulus 天气绘图机器人\n"
                   "━━━━━━━━━━━━━━━━━━\n"
                   "【实时 长三角】\n"
                   "/温度 /降水 /风场 /气压 /湿度 /综合\n\n"
                   "【ECMWF 全中国】\n"
                   "/EC 温度|降水|风场|气压|湿度|综合 [时效]\n"
                   "/EC h500|t850|高空风场\n\n"
                   "【GFS 全中国】\n"
                   "/GFS 温度|降水|风场|气压|湿度|综合 [时效]\n"
                   "/GFS h500|t850|高空风场\n\n"
                   "时效: 不填=分析场 24/48/72=预报\n"
                   "━━━━━━━━━━━━━━━━━━\n"
                   "/预报 /数据更新 /帮助")


# ---------------------------------------------------------------------------
# /今日云朵 — random cloud of the day
# ---------------------------------------------------------------------------

import random, re
from pathlib import Path

CLOUD_DIR = Path.home() / "Desktop" / "云图"

_cloud_cache: list[tuple[str, str]] | None = None


def _get_clouds() -> list[tuple[str, str]]:
    global _cloud_cache
    if _cloud_cache is None:
        clouds: list[tuple[str, str]] = []
        for f in sorted(CLOUD_DIR.glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".jfif", ".gif", ".webp"):
                name = re.sub(r"[\d\s.]+$", "", f.stem)
                clouds.append((name, str(f)))
        _cloud_cache = clouds
    return _cloud_cache


cloud_cmd = on_command("今日云朵", aliases={"云朵", "云"}, priority=10, block=True)


@cloud_cmd.handle()
async def handle_cloud(bot: Bot, event: GroupMessageEvent):
    clouds = _get_clouds()
    if not clouds:
        await bot.send(event, "云图文件夹为空，请检查 Desktop/云图 目录。")
        return

    name, path = random.choice(clouds)
    abs_path = Path(path).resolve().as_posix()
    msg = MessageSegment.text(f"今日云朵：{name} ☁\n") + MessageSegment.image(f"file:///{abs_path}")
    await bot.send(event, msg)

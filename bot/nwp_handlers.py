"""NWP command handlers for ECMWF IFS / GFS."""

import asyncio
import logging
import traceback

from nonebot import on_command
from nonebot.adapters.onebot.v11 import Bot, GroupMessageEvent, MessageSegment

logger = logging.getLogger("weather-bot.nwp")

# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------

ec_cmd  = on_command("EC", priority=5, block=True)
gfs_cmd = on_command("GFS", priority=5, block=True)

# ---------------------------------------------------------------------------
# Variable mapping
# ---------------------------------------------------------------------------

VAR_MAP = {
    "温度": "temperature", "气温": "temperature",
    "降水": "precipitation", "降雨": "precipitation",
    "风场": "wind", "风": "wind", "风速": "wind", "风向": "wind",
    "气压": "pressure",
    "湿度": "humidity",
    "综合": "comprehensive", "全部": "comprehensive",
    # Upper-air
    "h500": "h500", "500hPa": "h500",
    "t850": "t850", "850hPa": "t850",
    "高空风场": "wind_850", "高空风850": "wind_850",
}

VAR_DISPLAY = {
    "temperature": "温度", "precipitation": "降水",
    "wind": "风场", "pressure": "气压",
    "humidity": "湿度", "comprehensive": "综合",
    "h500": "500hPa位势高度", "t850": "850hPa温度",
    "wind_850": "850hPa风场", "wind_200": "200hPa急流",
}

FHOUR_MAP = {"0": 0, "00": 0, "24": 24, "48": 48, "72": 72}

SOURCE_LABEL = {"ecmwf": "ECMWF IFS", "gfs": "GFS"}
SOURCE_CMD  = {"ecmwf": "EC", "gfs": "GFS"}


def _parse_args(args: list[str]):
    if not args:
        return (None, None, None)
    first = args[0]
    if first in ("帮助", "help", "?", "？"):
        return (None, None, None)
    vk = VAR_MAP.get(first)
    if vk is None:
        return (first, None, None)
    dn = VAR_DISPLAY.get(vk, first)
    fh = None
    if len(args) > 1:
        fs = args[1].lstrip("+")
        fh = FHOUR_MAP.get(fs)
        if fh is None:
            try:
                fh = int(fs)
            except ValueError:
                pass
    return (vk, dn, fh)


# ---------------------------------------------------------------------------
# Shared runner
# ---------------------------------------------------------------------------

_nwp_semaphore = asyncio.Semaphore(2)


async def _run_nwp(bot, event, source, variable, display_name, forecast_hour):
    async with _nwp_semaphore:
        sl = SOURCE_LABEL[source]
        fh_label = ""
        if forecast_hour is not None and forecast_hour > 0:
            fh_label = f" +{forecast_hour}h"
        elif forecast_hour == 0:
            fh_label = " 分析场"

        await bot.send(event,
                       f"正在获取{sl} {display_name}{fh_label}数据，请稍候...")

        try:
            # Deferred import to avoid any module-level side effects
            from src.nwp import get_nwp_pipeline
            pipeline = get_nwp_pipeline()
            path = await asyncio.to_thread(
                pipeline.generate, source, variable, forecast_hour)
            import pathlib
            abs_path = pathlib.Path(path).resolve().as_posix()
            msg = (MessageSegment.text(f"【{display_name}】{sl}{fh_label}\n")
                   + MessageSegment.image(f"file:///{abs_path}"))
            await bot.send(event, msg)
        except Exception as e:
            logger.error(f"NWP failed [{source}/{variable}]: {traceback.format_exc()}")
            err_msg = str(e)[:300] if str(e) else "未知错误"
            await bot.send(event,
                           f"{sl} 数据获取失败: {err_msg}\n"
                           f"可能原因: 网络超时 / 数据尚未到达\n"
                           f"发送 /{SOURCE_CMD[source]} 帮助 查看用法。")


# ---------------------------------------------------------------------------
# Handlers — EXACT same signature as working handlers.py
# ---------------------------------------------------------------------------

@ec_cmd.handle()
async def handle_ec(bot: Bot, event: GroupMessageEvent):
    text = event.get_plaintext().strip()
    parts = text.split(None, 1)
    arg_text = parts[1] if len(parts) > 1 else ""
    args = arg_text.split() if arg_text else []

    logger.info(f"EC 命令: args={args}")
    var_key, display_name, fhour = _parse_args(args)

    if var_key is None and display_name is None:
        await bot.send(event, _help_text("EC", "ecmwf"))
        return
    if display_name is None:
        await bot.send(event,
                       f"未知变量: {var_key}\n"
                       f"可用: 温度 | 降水 | 风场 | 气压 | 湿度 | 综合\n"
                       f"发送 /EC 帮助 查看完整帮助。")
        return

    await _run_nwp(bot, event, "ecmwf", var_key, display_name, fhour)


@gfs_cmd.handle()
async def handle_gfs(bot: Bot, event: GroupMessageEvent):
    text = event.get_plaintext().strip()
    parts = text.split(None, 1)
    arg_text = parts[1] if len(parts) > 1 else ""
    args = arg_text.split() if arg_text else []

    logger.info(f"GFS 命令: args={args}")
    var_key, display_name, fhour = _parse_args(args)

    if var_key is None and display_name is None:
        await bot.send(event, _help_text("GFS", "gfs"))
        return
    if display_name is None:
        await bot.send(event,
                       f"未知变量: {var_key}\n"
                       f"可用: 温度 | 降水 | 风场 | 气压 | 湿度 | 综合\n"
                       f"发送 /GFS 帮助 查看完整帮助。")
        return

    await _run_nwp(bot, event, "gfs", var_key, display_name, fhour)


# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

def _help_text(cmd: str, source: str) -> str:
    sl = SOURCE_LABEL.get(source, source)
    return (
        f"{chr(127757)} {sl} 0.25° 数值预报\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"用法:\n"
        f"/{cmd} <变量> [时效]\n\n"
        f"变量:\n"
        f"  温度 — 2m温度\n"
        f"  降水 — 累计降水量\n"
        f"  风场 — 10m风场(含风羽)\n"
        f"  气压 — 海平面气压\n"
        f"  湿度 — 2m相对湿度\n"
        f"  综合 — 四合一综合分析\n\n"
        f"时效(可选):\n"
        f"  (不填) — 分析场(温度/风/气压/湿度)\n"
        f"           降水默认24h预报\n"
        f"  0  — 分析场\n"
        f"  24 — 未来24h预报\n"
        f"  48 — 未来48h预报\n"
        f"  72 — 未来72h预报\n\n"
        f"示例:\n"
        f"  /{cmd} 温度     → 分析场 2m温度\n"
        f"  /{cmd} 温度 48  → 未来48h 温度预报\n"
        f"  /{cmd} 降水     → 未来24h累计降水\n"
        f"  /{cmd} 综合     → 分析场四合一大图\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"数据来源: {sl} 0.25°\n"
        f"区域: 全中国 (73-136°E, 16-55°N)\n"
        f"缓存自动清理: 7天"
    )


# ---------------------------------------------------------------------------
# /数据更新 — manually download all NWP data (surface + upper-air)
# ---------------------------------------------------------------------------

update_cmd = on_command("数据更新", priority=5, block=True)


@update_cmd.handle()
async def handle_update(bot: Bot, event: GroupMessageEvent):
    await bot.send(event, "开始更新NWP数据（ECMWF + GFS，含高空变量）...\n各步时效: 0/24/48/72h")

    def _update():
        from src.nwp import ECMWFSource, GFSSource

        results: list[str] = []
        for src_cls, label in [(ECMWFSource, "ECMWF"), (GFSSource, "GFS")]:
            try:
                src = src_cls()
                src.cleanup()
                date, hour = src.get_latest_cycle()
                for step in [0, 24, 48, 72]:
                    try:
                        if not src.is_cached(date, hour, step):
                            logger.info(f"[{label}] Downloading f{step:03d} ...")
                            src.get_dataset(date, hour, step)
                        else:
                            logger.info(f"[{label}] f{step:03d} already cached")
                        results.append(f"{label} +{step}h ✅")
                    except Exception as e:
                        results.append(f"{label} +{step}h ❌ {str(e)[:80]}")
                        logger.warning(f"[{label}] f{step:03d} failed: {e}")
            except Exception as e:
                results.append(f"{label} 周期检测失败: {str(e)[:80]}")

        if not results:
            return "更新完成！（所有数据已在缓存中）"

        ok = sum(1 for r in results if "✅" in r)
        fail = sum(1 for r in results if "❌" in r)
        status = "\n".join(results[-8:])  # last 8 entries
        summary = f"更新完成！成功{ok}项，失败{fail}项。\n{status}"
        if ok == 0:
            summary = f"更新失败！所有下载均未成功。\n{status}\n\n请检查网络连接后重试。"
        return summary

    try:
        result = await asyncio.to_thread(_update)
        await bot.send(event, result)
    except Exception as e:
        await bot.send(event, f"更新过程出错: {str(e)[:200]}")


# ---------------------------------------------------------------------------
# /预报 — generate professional weather forecast text
# ---------------------------------------------------------------------------

forecast_cmd = on_command("预报", priority=5, block=True)


@forecast_cmd.handle()
async def handle_forecast(bot: Bot, event: GroupMessageEvent):
    await bot.send(event, "正在生成天气形势分析，请稍候...")

    try:
        from src.forecast import ForecastEngine

        engine = ForecastEngine()
        text = await asyncio.to_thread(engine.generate)
        await bot.send(event, text)
    except Exception as e:
        logger.error(f"Forecast failed: {traceback.format_exc()}")
        await bot.send(event, f"预报生成失败: {str(e)[:200]}")

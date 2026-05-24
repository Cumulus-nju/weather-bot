"""nonebot2 plugin package — weather-bot command handlers."""

from .handlers import (
    temperature, precipitation, wind, pressure, humidity, comprehensive, help_cmd,
)
from .nwp_handlers import ec_cmd, gfs_cmd  # register /EC and /GFS commands

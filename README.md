# weather-bot

QQ 群气象绘图机器人 — 长三角实时站点观测 + 全中国 ECMWF/GFS 数值预报。

南京大学气象爱好者

---

## 功能

### 实时站点观测（长三角，116-123°E, 27-35°N）

| 命令 | 功能 |
|------|------|
| `/温度` | 2m 温度分析图 |
| `/降水` | 降水量分布图 |
| `/风场` | 10m 风场图（含风羽） |
| `/气压` | 海平面气压分析图 |
| `/湿度` | 相对湿度分析图 |
| `/综合` | 四合一综合分析图 |
| `/帮助` | 显示帮助信息 |

数据来源：和风天气 QWeather 实时站点 + Barnes 插值 / ML 多变量精炼

### ECMWF IFS 数值预报（全中国，73-136°E, 16-55°N）

| 命令 | 功能 |
|------|------|
| `/EC 温度 [时效]` | 2m 温度 |
| `/EC 降水 [时效]` | 累计降水量 |
| `/EC 风场 [时效]` | 10m 风场 |
| `/EC 气压 [时效]` | 海平面气压 |
| `/EC 湿度 [时效]` | 相对湿度 |
| `/EC 综合 [时效]` | 四合一综合图 |

时效（可选）：不填默认分析场（降水默认 24h 预报），`24`/`48`/`72` 为预报

示例：`/EC 温度` `/EC 降水 48` `/EC 综合 24`

### GFS 数值预报（全中国）

| 命令 | 功能 |
|------|------|
| `/GFS 温度 [时效]` | 2m 温度 |
| `/GFS 降水 [时效]` | 累计降水量 |
| `/GFS 风场 [时效]` | 10m 风场 |
| `/GFS 气压 [时效]` | 海平面气压 |
| `/GFS 湿度 [时效]` | 相对湿度 |
| `/GFS 综合 [时效]` | 四合一综合图 |

用法同 ECMWF，示例：`/GFS 风场` `/GFS 温度 72`

---

## 架构

```
NapCatQQ (QQ 协议端)
    │  reverse WebSocket
    ▼
nonebot2 (OneBot V11, port 8080)
    │
    ├── bot/handlers.py       # 站点观测命令
    └── bot/nwp_handlers.py   # EC/GFS 命令
           │
      src/nwp.py              # NWP 下载/缓存/解码/绘图
      src/pipeline.py          # 站点数据 + ML 插值
      src/plotter.py           # Cartopy 绘图（共享）
```

---

## 快速开始

### 1. 环境要求

- **Python 3.9+**
- NapCatQQ Shell（QQ 协议端，需 Windows）
- 一个 QQ 小号（推荐，避免大号被封）

### 2. 安装依赖

```bash
cd weather-bot
pip install -r requirements.txt
```

如果 `cfgrib` 安装失败（Windows 上 eccodes 依赖问题），可以尝试：

```bash
pip install cfgrib eccodes --only-binary cfgrib
```

### 3. 配置

复制 `.env.example` 为 `.env`，填写：

```env
# 数据源: qweather（和风天气，免费 1000次/天）
STATION_DATA_SOURCE=qweather
QWATHER_API_KEY=你的和风天气key
QWATHER_API_HOST=你的和风天气host

# nonebot2 + NapCat 连接
HOST=127.0.0.1
PORT=8080
ONEBOT_ACCESS_TOKEN=weather-bot-token
```

**和风天气 API 获取**：https://dev.qweather.com/ （免费版 1000次/天，足够用）

### 4. 配置 NapCatQQ

1. 下载 NapCatQQ Shell：https://github.com/NapNeko/NapCatQQ
2. 在 NapCat 的 `config/napcat.json` 中设置 WebSocket 连接：

```json
{
  "wsReverse": {
    "enable": true,
    "urls": ["ws://127.0.0.1:8080/onebot/v11/ws"]
  }
}
```

3. 设置 `ONEBOT_ACCESS_TOKEN` 与 `.env` 中一致

### 5. 启动

**先启动 NapCatQQ**（双击 `launcher-user.bat`，扫码登录 QQ），然后：

```bash
cd weather-bot
python main.py
```

看到 `Pipeline 初始化完成，可以接收命令。` 即可在 QQ 群中发送命令。

---

## ECMWF 数据说明

2025 年 10 月起 ECMWF 全部实时预报数据免费开放（CC-BY-4.0），无需注册。首次请求会从欧洲 S3 下载（30-60 秒），之后使用本地缓存（6 小时 TTL），秒出图。

数据自动缓存在 `data/nwp/`，7 天后自动清理。

---

## 目录结构

```
weather-bot/
├── main.py                  # 入口
├── config.py                # 全局配置
├── requirements.txt         # Python 依赖
├── .env.example             # 环境变量模板
├── bot/                     # nonebot2 插件
│   ├── handlers.py          # /温度 /降水 等命令
│   └── nwp_handlers.py      # /EC /GFS 命令
├── src/                     # 核心模块
│   ├── nwp.py               # NWP 下载/缓存/调度
│   ├── pipeline.py          # 站点数据 + 插值
│   ├── plotter.py           # Cartopy 绘图
│   ├── interpolation.py     # Barnes/IDW/克里金
│   ├── station_data.py      # QWeather 数据源
│   ├── ml_model.py          # CNN 精炼器
│   ├── background.py        # 背景场
│   └── assimilation.py      # OI 最优插值
└── data/
    ├── stations_yangtze_delta.json  # 站点列表
    └── training/            # ML 模型和训练数据
```

---

## License

MIT

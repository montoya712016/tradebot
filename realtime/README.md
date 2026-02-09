# Realtime Trading Module

Runtime components used by the production bot.

Structure
---------
- `bot/` - decision/execution loop (`realtime.bot.sniper`) and settings.
- `market_data/` - websocket, REST backfill queue, MySQL adapters, rolling windows.

Dashboard
---------
The web dashboard server and static assets live under `modules/realtime/`.

Run
---
```python
from realtime.bot.sniper import LiveDecisionBot
from realtime.bot.settings import LiveSettings

settings = LiveSettings()
bot = LiveDecisionBot(settings)
bot.start()
```

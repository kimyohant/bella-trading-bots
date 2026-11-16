# bottrade

บอทเทรดแบบ **paper trading** สำหรับ **Polymarket** (prediction market) + **pump.fun** (Solana memecoin)
โครงสร้างแบบ multi-agent ที่แยกแต่ละหน้าที่ชัดเจน และ **ขยาย agent เพิ่มได้** โดยไม่ต้องแก้โค้ดเดิม

> ⚠️ **ไม่มีการส่งคำสั่งซื้อขายจริงในโปรเจกต์นี้** — ทุกออเดอร์ถูก simulate ผ่าน `PaperBroker`
> ด้วยราคาจริงจากตลาด (slippage + fee จริง) เพื่อวัดกลยุทธ์ก่อนเอาเงินจริงเข้า

---

## 1. Pipeline

```
feed ──► scanner ──► sentiment ──► whale ──► signal ──► risk ──► executor ──► portfolio
         (universe)  (lexicon/news) (flow)   (edge)    (size/gate) (fills)     (equity/report)
```

| agent | หน้าที่ | ไฟล์ |
|---|---|---|
| `scanner` | กรองตลาดที่เทรดได้ + quota ต่อ venue | `agents/scanner.py` |
| `sentiment` | ให้คะแนนข้อความ → ปรับ probability (lexicon / RSS / LLM) | `agents/sentiment.py` |
| `whale` | วัดแรงซื้อ-ขายจาก trade tape / taker counts | `agents/whale.py` |
| `signal` | รวมสัญญาณเป็น "fair price" แล้วคัดเฉพาะที่มี edge | `agents/signal.py` |
| `risk` | exit (stop/target/time/decay) + hard gate + position sizing | `agents/risk.py` |
| `executor` | ส่งออเดอร์เข้า broker เท่านั้น | `agents/executor.py` |
| `portfolio` | mark-to-market + บันทึก equity ลง sqlite | `agents/portfolio.py` |

แต่ละ agent รับ `Context` เดียวกัน (blackboard) — agent ใหม่แค่อ่าน/เขียน field ใน `Context`

---

## 2. เริ่มใช้ใน 3 คำสั่ง

```bash
git clone https://github.com/kimyohant/bella-trading-bots.git
cd bella-trading-bots

python -m bottrade scan --source live --limit 10     # ดูตลาดจริงตอนนี้
python -m bottrade run  --source mock --ticks 60 --report   # รัน demo แบบ offline
python -m bottrade run  --source live --duration 1800 --report  # รันกับข้อมูลจริง 30 นาที
python -m bottrade serve --source mock --open                    # เปิด web dashboard
```

ไม่ต้องติดตั้งอะไรเลย — core ใช้ **stdlib ล้วน** (Python 3.11+ เพราะใช้ `tomllib`)
ถ้าอยากให้เร็วขึ้น/สวยขึ้นค่อยลง optional deps:

```bash
pip install -r requirements.txt   # httpx (connection pooling) + rich
```

คำสั่งอื่น:

```bash
python -m bottrade backtest --demo --ticks 300     # backtest ด้วยข้อมูลสังเคราะห์
python -m bottrade backtest --csv myprices.csv     # backtest ด้วยข้อมูลจริงของคุณ
python -m bottrade report --db state/bottrade.sqlite3
python -m bottrade agents                          # ดู agent ที่ลงทะเบียนไว้
```

---

## 3. โหมดการรัน

| `--source` | ใช้เมื่อไหร่ | ต้องมีเน็ต |
|---|---|---|
| `mock` | พัฒนา/เดโม/เทส — สร้างโลกสังเคราะห์ที่มี fair value ซ่อนอยู่ ทำให้ pipeline ได้เทรดจริง | ไม่ต้อง |
| `live` | ข้อมูลจริงจาก Polymarket + pump.fun, fill แบบ paper | ต้อง |
| `replay` | backtest จาก CSV | ไม่ต้อง |

CSV สำหรับ `replay`:

```csv
ts,key,price,bid,ask,question,liquidity_usd,volume_24h_usd
2026-11-16T01:00:00Z,polymarket:1234567890,0.42,0.41,0.43,Will X happen?,50000,120000
```

`key` = `polymarket:<clob_token_id>` หรือ `pumpfun:<mint>`

---

## 3.5 Web monitor (dashboard)

```bash
python -m bottrade serve --source mock --open     # รันบอท + เปิด dashboard เลย
python -m bottrade serve --source live            # ข้อมูลจริง
python -m bottrade serve --no-run --db state/bottrade.sqlite3   # ดู run เก่าแบบ read-only
```

เปิดที่ `http://127.0.0.1:8787/` — push ข้อมูลผ่าน **SSE** ทุก 2 วินาที (ถ้าหลุดจะ fallback เป็น polling เอง)

สิ่งที่เห็นบนจอ:

| panel | ข้อมูล |
|---|---|
| hero | equity / net P&L / return เทียบ starting cash |
| stat cards | cash, exposure, realized, unrealized, win rate, max drawdown |
| equity curve | วาด SVG เองจากตาราง `equity` (ไม่ใช้ library, ไม่มี CDN) |
| by venue | fills / notional / PnL / fees แยก Polymarket กับ pump.fun |
| open positions | snapshot ของ tick ล่าสุด + PnL ที่ยังไม่ realize |
| recent fills | 40 รายการล่าสุด พร้อม flash เมื่อมี fill ใหม่ |
| latest signals | edge, fair price vs ราคาจริง, confidence bar, เหตุผล |
| rejections | **ทำไมออเดอร์ถูกปฏิเสธ** — รวม reason ที่คล้ายกันเป็น bucket เดียว |

ทางเทคนิค: `http.server` ของ stdlib ล้วน, bind `127.0.0.1` เท่านั้น, อ่าน sqlite ด้วย read-only
connection แยกต่อ request (เปิด WAL ไว้) จึงอ่านได้ขณะบอทเขียน โดยไม่บล็อกกัน

API ที่เรียกได้ตรง ๆ:

```bash
curl http://127.0.0.1:8787/api/snapshot   # JSON ก้อนเดียวจบ
curl http://127.0.0.1:8787/api/health
curl -N http://127.0.0.1:8787/api/stream  # SSE
```

---

## 4. ขยาย agent เพิ่ม (สิ่งที่ขอ)

เพิ่มไฟล์เดียว ไม่ต้องแตะ pipeline:

```python
# my_agents/funding.py
from bottrade.agents import Agent, Context, register

@register("funding")            # ตั้งชื่อ agent
class FundingAgent(Agent):
    name = "funding"            # ใช้เป็น prefix ของ config: [funding] ...

    async def run(self, ctx: Context) -> None:
        for market in ctx.markets:
            # อ่านข้อมูลที่ agent ก่อนหน้าวางไว้
            flow = ctx.flow.get(market.key)
            # ...คำนวณอะไรก็ได้ แล้วเขียนกลับเข้า Context
            if flow and flow.bias > 0.5:
                ctx.note(f"strong bid on {market.question}")
```

แล้วเปิดใช้ใน `config.toml`:

```toml
[agents]
enabled = ["scanner", "sentiment", "whale", "funding", "signal", "risk", "executor", "portfolio"]
custom_modules = ["my_agents.funding"]
```

**ตำแหน่งใน pipeline = อำนาจของ agent** — ถ้าอยากให้มีผลต่อการตัดซื้อขาย ต้องอยู่ก่อน `risk`
ถ้าอยากแค่บันทึกข้อมูล ใสหลัง `portfolio` ก็ได้

`Context` ที่ agent เข้าถึงได้:

| field | ความหมาย |
|---|---|
| `ctx.markets` | ตลาดที่ scanner คัดแล้ว |
| `ctx.quotes` | bid/ask ต่อ key |
| `ctx.sentiment` / `ctx.flow` | ผลจาก sentiment / whale |
| `ctx.signals` | สัญญาณที่ signal สร้าง |
| `ctx.orders` / `ctx.fills` | ออเดอร์ที่ risk อนุมัติ / fill ที่เกิดขึ้น |
| `ctx.broker` | portfolio + positions + cash |
| `ctx.state` | dict ข้าม tick (price history, cooldown, counters) |
| `ctx.reject`, `ctx.note` | บันทึกเหตุผลลง report |

### เพิ่ม venue ใหม่

implement 3 method ใน `bottrade/venues/base.py::VenueAdapter` (`list_markets`, `get_quote`,
`recent_trades`) แล้วเพิ่มใน `feeds.py::build_feed` — agent ทั้งหมดไม่ต้องแก้

---

## 5. ข้อมูลมาจากไหน

| venue | แหล่งข้อมูล | สถานะ |
|---|---|---|
| Polymarket | Gamma API (ตลาด/ราคา), CLOB `/book` (order book), Data API `/trades` (tape) | สาธารณะ ไม่ต้อง key |
| pump.fun | **GeckoTerminal** `networks/solana/dexes/pump-fun/pools` (ราคา, FDV, reserve, taker counts, %change) | สาธารณะ ไม่ต้อง key |
| pump.fun (สำรอง) | `frontend-api-v3.pump.fun` → Dexscreener | endpoint หน้าเว็บถูก Cloudflare บล็อกจากหลาย IP |

จุดที่ต้องรู้ตรง ๆ:

- **pump.fun ไม่มี trading API สาธารณะ** — ถ้าจะเทรดจริงต้องสร้าง transaction เอง (bonding curve
  program) หรือใช้ Jito bundle; ไฟล์ `venues/pumpfun.py` เป็น read-only โดยเจตนา
- bonding curve ไม่มี order book จริง — bid/ask จึงเป็น **แบบจำลอง** (`scanner.pumpfun_spread_bps`)
- Polymarket `outcomePrices` บางครั้งล้าเกินจริง โค้ดจึงยึด mid ของ `bestBid/bestAsk` เมื่อมี book
- endpoint ทั้งสองเจ้าเปลี่ยนได้ตลอด — โค้ดทุกจุด degrade เป็น list ว่าง ไม่ throw

---

## 6. ค่าที่ควรจูน (ใน `config.toml`)

| กลุ่ม | key | ค่าเริ่มต้น | หมายเหตุ |
|---|---|---|---|
| edge | `signal.min_edge` | `0.05` | ต้องมี edge 5% ขึ้นไปถึงจะยิง |
| edge | `risk.min_edge_after_costs` | `0.03` | ต่ำกว่านี้ตัดทิ้ง (กัน fee/slippage กิน) |
| size | `risk.kelly_fraction` | `0.25` | quarter Kelly — สูงกว่านี้คือเดิมพันแรง |
| size | `risk.max_position_pct` | `0.08` | ไม่เกิน 8% ของ equity ต่อโพซิชัน |
| risk | `risk.daily_loss_limit_pct` | `0.08` | ขาดทุน 8%/วัน = หยุดเปิดออเดอร์ใหม่ |
| risk | `risk.max_positions_per_venue` | `4` | กันกระจุกตัวในเจ้าเดียว |
| exit | `risk.stop_loss_pct` | `0.35` | binary market |
| exit | `risk.stop_loss_pct_token` | `0.25` | memecoin (ผันผวนกว่า) |
| exit | `risk.max_hold_hours_token` | `2.0` | memecoin ไม่ควรถือ 24 ชม. |
| fill | `slippage.pumpfun_bps` | `250` | curve เลื่อนสวนทาง |
| fill | `fees.pumpfun_fee_bps` | `100` | protocol fee ~1% |

override ด้วย env ได้ทุกค่า (ไม่ต้องแก้ไฟล์):

```bash
BOTTRADE__RISK__KELLY_FRACTION=0.15 BOTTRADE__SIGNAL__MIN_EDGE=0.08 python -m bottrade run --source live
```

---

## 7. Report

ทุก run เขียนลง sqlite (`state/bottrade.sqlite3`) — ticks, markets, signals, orders, fills, equity
แล้ว `--report` (หรือ `python -m bottrade report`) จะสร้าง:

- `reports/report_<ts>.md` — headline metrics, equity curve, per-venue, trades, rejection reasons
- `reports/fills_<ts>.csv` — fill ทุกตัวเอาไปต่อ Excel ได้

---

## 8. เทส

```bash
python -m unittest discover -s bottrade/tests -t . -v
```

39 tests ครอบ: broker fills/fees/slippage/partial, Kelly sizing, ทุก hard gate, exits,
sentiment lexicon+negation, sqlite logging, report generation, backtest, monitor snapshot API

---

## 9. ถ้าจะไป live จริง

สิ่งที่ต้องทำ (และยังไม่ได้ทำ โดยเจตนา):

1. **Polymarket** — ใช้ `py-clob-client`, เซ็น EIP-712 ด้วย private key, ต้องมี USDC บน Polygon
   และ approve allowance; เริ่มจาก `post_only` order ราคาต่ำกว่าตลาดก่อน
2. **pump.fun** — ต้องสร้าง Solana transaction เอง (bonding curve buy/sell instruction)
   หรือใช้ aggregator; ต้องจัดการ priority fee + slippage guard
3. แทน `PaperBroker` ด้วย adapter จริง **โดยคง interface เดิม** (`execute(order, quote) -> Fill`)
   แล้วเก็บ `PaperBroker` ไว้เป็น dry-run mode
4. ใส่ kill-switch, per-trade max loss, และ reconciliation กับ exchange ทุก N นาที

ความเสี่ยงที่ paper trading **ไม่จำลอง**: MEV/sandwich, transaction failure, การที่ liquidity
หายตอนคุณจะออก, และการที่ตลาด resolution ผิดพลาด — พวกนี้กินเงินจริงเสมอ

---

## 10. โครงสร้าง

```
config.toml               ทุก knob อยู่ในนี้
bottrade/
  config.py               TOML + env override
  models.py               Market / Signal / Order / Fill / Position
  feeds.py                LiveFeed / MockFeed / ReplayFeed
  backtest.py             replay CSV + metrics
  report.py               sqlite → markdown
  cli.py                  run / scan / serve / backtest / report / agents
  venues/
    base.py               VenueAdapter interface
    polymarket.py         Gamma + CLOB + Data API
    pumpfun.py            GeckoTerminal + frontend fallback
    paper.py              PaperBroker (slippage, fee, partial fill)
    mock.py               synthetic world
  agents/
    base.py               Context + Agent + registry
    scanner.py sentiment.py whale.py signal.py risk.py executor.py portfolio.py
    orchestrator.py       pipeline driver
  data/
    http.py               async fetch + cache + retries (httpx ถ้ามี)
    store.py              sqlite schema + writers
  web/
    server.py             monitor API + SSE (stdlib http.server)
    ui.html               dashboard (single file, no CDN)
```

---

*Paper trading only. ไม่ใช่คำแนะนำการลงทุน — prediction market และ memecoin มีความเสี่ยงสูงมาก*

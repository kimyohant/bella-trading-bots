# bottrade

บอทเทรดแบบ **paper trading** สำหรับ **Polymarket** (prediction market) + **pump.fun** (Solana memecoin)
โครงสร้างแบบ multi-agent ที่แยกแต่ละหน้าที่ชัดเจน และ **ขยาย agent เพิ่มได้** โดยไม่ต้องแก้โค้ดเดิม

> ⚠️ **ไม่มีการส่งคำสั่งซื้อขายจริงในโปรเจกต์นี้** — ทุกออเดอร์ถูก simulate ผ่าน `PaperBroker`
> ด้วยราคาจริงจากตลาด (slippage + fee จริง) เพื่อวัดกลยุทธ์ก่อนเอาเงินจริงเข้า

---

## 1. Pipeline

```
feed ──► guard ──► scanner ──► sentiment ──► whale ──► crypto ──► signal ──► risk ──► executor ──► portfolio
         (health)  (universe)  (lexicon/news) (flow)   (spot)     (edge)    (size/gate) (fills)     (equity/report)
```

| agent | หน้าที่ | ไฟล์ |
|---|---|---|
| `guard` | เช็คว่า venue ยังส่งข้อมูลอยู่ไหม + dead-man's switch | `agents/guard.py` |
| `scanner` | กรองตลาดที่เทรดได้ + quota ต่อ venue | `agents/scanner.py` |
| `sentiment` | ให้คะแนนข้อความ → ปรับ probability (lexicon / RSS / LLM) | `agents/sentiment.py` |
| `whale` | วัดแรงซื้อ-ขายจาก trade tape / taker counts | `agents/whale.py` |
| `crypto` | fair probability ของตลาด 15 นาที จากราคา spot จริง | `agents/crypto.py` |
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
| data health | แถบใต้ topbar: venue ไหนสด / ค้างกี่ tick / dead-man's switch |

ทางเทคนิค: `http.server` ของ stdlib ล้วน, bind `127.0.0.1` เท่านั้น, อ่าน sqlite ด้วย read-only
connection แยกต่อ request (เปิด WAL ไว้) จึงอ่านได้ขณะบอทเขียน โดยไม่บล็อกกัน

API ที่เรียกได้ตรง ๆ:

```bash
curl http://127.0.0.1:8787/api/snapshot   # JSON ก้อนเดียวจบ
curl http://127.0.0.1:8787/api/health
curl -N http://127.0.0.1:8787/api/stream  # SSE
```

---

## 3.6 Guard — ตัวที่กล้าพูดว่า "ไม่รู้"

`LiveFeed` กลืน exception ของ venue ที่ล่ม แล้วคืน `[]` เฉย ๆ — ไปป์ไลน์ที่เหลือจึงไม่รู้เลยว่า
ราคาที่เห็นคือราคาค้าง และ `risk` เคยตัดสินใจขายด้วย `last_price` เดิม (`max-hold` กับ `thesis-decay`
ไม่ต้องใช้ราคาสดด้วยซ้ำ) นั่นคือวิธีที่บอท book fill ในราคาที่ไม่มีอยู่จริง

`guard` รันเป็นตัวแรกและแก้ตรงจุดนั้น:

| สถานะ | เงื่อนไข | ผล |
|---|---|---|
| fresh | venue ส่งข้อมูลทุก tick | ปล่อยผ่าน |
| stale | ไม่ส่งติดกัน `max_quote_age_ticks` | `risk` **ไม่ยอมขาย** ด้วยราคาค้าง (บันทึก `stale-mark`) |
| forced | ค้างเกิน `max_stale_hours` | ยอมปิดไม้ที่ราคาล่าสุด เพราะการถือไม้ตายไว้ตลอดกาลแย่กว่า |
| dead-man | ไม่มี venue ไหนส่งเลย `max_empty_ticks` | halt entries จนกว่าข้อมูลจะกลับมา |

```toml
[guard]
max_quote_age_ticks = 3     # ขาดกี่ tick ถึงนับว่าค้าง
max_empty_ticks = 5         # ไม่มีข้อมูลเลยกี่ tick ถึง trip dead-man
max_stale_hours = 6.0       # ค้างนานสุดก่อนยอมปิดไม้ที่ราคาล่าสุด
stale_action = "hold"       # hold = ไม่ยอม fill ปลอม (ค่าเริ่มต้น) | liquidate = ขายเลย
```

ผลข้างเคียงที่ตั้งใจ: ตอน venue ล่ม ระบบจะ **ไม่ขาย** อะไรเลย แล้วพอ venue กลับมา
stop-loss/trailing จะยิงทันทีด้วยราคาจริง — ช้ากว่าแต่ซื่อสัตย์กว่า

---

## 3.7 ตลาด 15 นาที: BTC / ETH / XRP

Polymarket มีซีรีส์ `btc-up-or-down-15m` / `eth-…` / `xrp-…` — เปิดใหม่ทุก 15 นาที
ถามง่าย ๆ ว่า "ราคาปลายช่วง ≥ ราคาต้นช่วง ไหม"

**กฎ resolution ไม่ใช่ spot ธรรมดา** — อ่านจาก description ของตลาด:

> resolve to "Up" if the TWAP of Bitcoin, generated by Chainlink, of the time range
> … is greater than or equal to the price at the beginning of that range

เพราะฉะนั้นการเอา spot มาเทียบตรง ๆ มี basis risk และนี่คือสิ่งที่ผมวัดก่อนเขียนโค้ด:

| วิธี | ตรงกับผลจริง |
|---|---|
| Coinbase close ที่ปลายช่วง vs open ที่ต้นช่วง (40 window) | **92%** (37/40) |
| TWAP เฉลี่ยทั้งหน้าต่าง | 80% (32/40) |

ที่พลาดทั้ง 3 ครั้งคือช่วงที่ราคาแกว่งไม่ถึง ~8bp คือ **อยู่ติดเส้น** ซึ่งเป็นจุดที่ส่วนต่าง
ระหว่าง feed ตัดสินผล — ไม่ใช่จุดที่ควรเทรด ตัวเลขนี้จึงกลายเป็น gate `basis_bps`

**โมเดล** (สมมติ log price เดินแบบเกาส์เซียน):

```
r  = ln(spot_now / spot_at_window_open)     วิ่งไปแล้วเท่าไร
tau = วินาทีที่เหลือในหน้าต่าง                เหลือเวลาให้วิ่งอีกเท่าไร
sigma = ความผันผวนต่อวินาที (จากแท่งเทียน 1 นาที 90 แท่ง)

P(up) = Φ( r / (sigma · √tau) )
```

พอ `tau → 0` ค่าจะลู่เข้า 1 หรือ 0 เอง ซึ่งถูกต้อง — tick สุดท้ายเป็นตัวตัดสิน

**ตรวจความแม่นของโมเดล** บน 40 window ที่ resolve แล้ว (วัดที่ นาทีที่ 7 ของหน้าต่าง):

```
Brier 0.165   (เดาสุ่ม = 0.250)
p ≥ 0.60 → ออก Up จริง 13/14 = 93%
p ≤ 0.40 → ออก Up จริง  4/17 = 24%
```

เทียบกับราคาตลาดจริงตอนเขียนโค้ด: BTC โมเดล 0.371 / book 0.36-0.37 — ตรงกันมาก
ส่วน ETH โมเดลต่ำกว่าตลาด ~6c ซึ่งเป็นกรณีที่โมเดล *ไม่* ควรเชื่อตัวเองเต็มที่ จึงมี gate

**Gate ก่อนเทรด** (ทุกข้อต้องผ่าน):

| gate | ค่า | เหตุผล |
|---|---|---|
| `basis_bps` | 4 | แกว่งน้อยกว่านี้ = basis noise ไม่ใช่ข้อมูล |
| `min_z` | 0.8 | สัญญาณต้องแรงพอเทียบความผันผวนที่เหลือ |
| `min_edge` | 0.08 | ต้องถูกกว่าราคา ask จริง 8% ขึ้นไป |
| `min_seconds_left` | 300 | ต้องเหลือเวลาให้เข้าออก (มากกว่า exit ข้างล่าง) |

**Risk profile แยก** — หน้าต่าง 15 นาทีจบก่อนกฎเดิมทั้งหมดจะทำงาน และมัน resolve เป็น 0/1
ซึ่ง paper broker settle ไม่ได้ สิ่งเดียวที่ยอมรับได้คือ **flat ก่อน resolve**

```toml
[risk]
stop_loss_pct_crypto = 0.30
max_hold_minutes_crypto = 10        # นาที ไม่ใช่ชั่วโมง
exit_seconds_before_window = 120    # ออกก่อนปิดหน้าต่าง 2 นาที
max_crypto_positions = 2            # แยกจาก max_positions_per_venue
max_crypto_trades_per_day = 24      # กัน crypto กิน budget ทั้งวัน
max_position_usd_crypto = 40        # coin flip ไม่ควรได้ Kelly เต็มก้อน
crypto_max_spread_pct = 0.08        # ต้อง >= scanner.crypto_max_spread_pct
```

**กับดักสองข้อที่เคยทำให้ crypto ไม่เทรดเลย** — ทั้งคู่แก้แล้วใน `risk.py`:

1. **spread สองชั้นไม่ตรงกัน.** scanner รับที่ 8% แต่ risk ใช้ `max_spread_pct = 0.06`
   ร่วมกับตลาด binary ทำให้ leg ราคา 14.5¢ ที่มี spread 1¢ (= 6.9%) ถูก reject ทุกตัว
   ทั้งที่ leg พวกนี้คือที่ที่ edge อยู่ (วัดจาก run จริง: ETH +34.3%, BTC +20.8%)
   ตอนนี้ crypto ใช้ `crypto_max_spread_pct` ของตัวเอง แยกจาก binary
   → **กฎง่าย ๆ: leg ที่ราคาต่ำกว่า ~16.7¢ จะเทรดไม่ได้ถ้า cap ยังเป็น 6%**

2. **crypto กินโควต้า venue ของ Polymarket.** position ทุกตัวนับรวมเป็น
   `venue_positions[POLYMARKET]` แม้จะแยก cap เป็น `max_crypto_positions` แล้ว
   เท่ากับ 2 ไม้ 15 นาทีจะลดความจุของ binary book จาก 4 เหลือ 2
   ตอนนี้ crypto ไม่ถูกนับใน venue cap/exposure ของ binary และได้ budget
   `max_crypto_positions × max_position_usd_crypto` เป็นของตัวเอง

**ข้อควรรู้ก่อนเชื่อตัวเลข** — โมเดลนี้อ้างอิง Coinbase ไม่ใช่ Chainlink:
basis ระหว่างสองเจ้าคือความเสี่ยงจริงของ strategy นี้, ตลาดพวกนี้บาง (liquidity $1k–$12k)
และ spread 1c บนสัญญา 50c = 2% ต่อขา ซึ่งกิน edge ไปแล้วครึ่งหนึ่งถ้าออกเร็ว
ผมจึงตั้ง `max_position_usd_crypto` ไว้เตี้ย ๆ และให้มันเทรดเฉพาะตอนที่สัญญาณแรงจริง

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
| spot (15m) | **Coinbase** `/ticker` + `/candles` (สำรอง Binance) | ใช้แทน Chainlink TWAP ที่ subscribe ไม่ได้ |

จุดที่ต้องรู้ตรง ๆ:

- **pump.fun ไม่มี trading API สาธารณะ** — ถ้าจะเทรดจริงต้องสร้าง transaction เอง (bonding curve
  program) หรือใช้ Jito bundle; ไฟล์ `venues/pumpfun.py` เป็น read-only โดยเจตนา
- bonding curve ไม่มี order book จริง — bid/ask จึงเป็น **แบบจำลอง** (`scanner.pumpfun_spread_bps`)
- Polymarket `outcomePrices` บางครั้งล้าเกินจริง โค้ดจึงยึด mid ของ `bestBid/bestAsk` เมื่อมี book
- **แต่ในตลาด 15 นาที `bestBid/bestAsk` ของ Gamma ก็ล้าได้เหมือนกัน** — วัดได้ 0.56/0.57
  ขณะที่ CLOB จริงคือ 0.49/0.50 ซึ่งบนสัญญา 15 นาทีคือคนละโลก ตลาด crypto จึงอ่าน `/book` ตรง ๆ
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
| risk | `risk.max_positions_per_venue` | `4` | กันกระจุกตัวในเจ้าเดียว (ไม่นับ crypto) |
| risk | `risk.crypto_max_spread_pct` | `0.08` | leg 15 นาทีราคาต่ำ spread 1c = ~7% |
| exit | `risk.stop_loss_pct` | `0.35` | binary market |
| exit | `risk.stop_loss_pct_token` | `0.25` | memecoin (ผันผวนกว่า) |
| exit | `risk.max_hold_hours_token` | `2.0` | memecoin ไม่ควรถือ 24 ชม. |
| profit | `risk.take_profit_partial_pct` | `0.60` | ถึง +60% ขายครึ่งแรก |
| profit | `risk.take_profit_partial_fraction` | `0.5` | สัดส่วนที่ขายตอนถึงเป้าแรก |
| profit | `risk.trail_arm_pct` | `0.35` | เริ่ม trailing หลังกำไร +35% |
| profit | `risk.trail_giveback_pct` | `0.30` | ถอยจากจุดสูงสุด 30% ของกำไร → ออก |
| fill | `slippage.pumpfun_bps` | `250` | curve เลื่อนสวนทาง |
| fill | `fees.pumpfun_fee_bps` | `100` | protocol fee ~1% |
| health | `guard.max_quote_age_ticks` | `3` | ขาดกี่ tick ถึงนับว่า venue ค้าง |
| health | `guard.max_empty_ticks` | `5` | ไม่มีข้อมูลเลยกี่ tick ถึง halt |
| health | `guard.stale_action` | `hold` | ไม่ยอม fill ด้วยราคาค้าง |
| crypto | `crypto.basis_bps` | `4` | ต่ำกว่านี้คือ basis noise ไม่ใช่ edge |
| crypto | `crypto.min_z` | `0.8` | สัญญาณต้องแรงพอ |
| crypto | `crypto.min_seconds_left` | `300` | ต้องเหลือเวลาให้เข้าออก |
| crypto | `risk.exit_seconds_before_window` | `120` | ออกก่อน resolve |
| crypto | `risk.max_position_usd_crypto` | `40` | coin flip ไม่ควรเดิมพันใหญ่ |

override ด้วย env ได้ทุกค่า (ไม่ต้องแก้ไฟล์):

```bash
BOTTRADE__RISK__KELLY_FRACTION=0.15 BOTTRADE__SIGNAL__MIN_EDGE=0.08 python -m bottrade run --source live
```

### ทำไมต้องมี 3 กฎขายทำกำไร ไม่ใช่ target เดียว

ถ้าตั้งแต่ target คงที่ (เช่น +100%) ตัวที่วิ่งไป +90% แล้วกลับตัวจะไม่ถูกขายเลย
แล้วไหลลงไปชน stop loss — กลายเป็นขาดทุนทั้งที่เคยกำไรเกือบเท่าตัว

ลำดับที่ `risk` ตรวจ (บนลงล่าง ตัวแรกที่จริงชนะ):

1. `stop-loss` — ขาดทุนถึงเกณฑ์ ตัดทิ้ง
2. `take-profit` — ถึง target สูงสุด ขายหมด
3. `take-profit-partial` — ถึงเป้าแรก ขายครึ่ง เก็บที่เหลือไว้วิ่ง
4. `trailing-stop` — เคยขึ้นถึง `trail_arm_pct` แล้วถอยเกิน `trail_giveback_pct` ของกำไรสูงสุด
5. `max-hold` — ถือนานเกิน
6. `thesis-decay` — sentiment/flow เปลี่ยนทางจากตอนเข้า

ทดสอบกับ mock 400 ticks (seed เดียวกัน) จะเห็นผลชัด:

| | target เดียว | + partial + trailing |
|---|---|---|
| realized PnL | +$112.51 | **+$144.69** |
| win rate | 44% | **92%** |
| stop-loss ที่ขาดทุน | 5 ครั้ง (−$61.05) | 1 ครั้ง (−$13.09) |
| position ที่เคยกำไร >40% แล้วจบขาดทุน | 4 ตัว | **0 ตัว** |

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

74 tests ครอบ: broker fills/fees/slippage/partial, Kelly sizing, ทุก hard gate, exits,
sentiment lexicon+negation, guard staleness + dead-man's switch + no-fabricated-fills,
crypto fair value + basis gate + risk profile + spread/venue isolation จาก binary book,
sqlite logging, report generation, backtest, monitor snapshot API

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
    guard.py scanner.py sentiment.py whale.py crypto.py signal.py risk.py
    executor.py portfolio.py
    orchestrator.py       pipeline driver
  data/
    http.py               async fetch + cache + retries (httpx ถ้ามี)
    spot.py               Coinbase/Binance spot + realized vol (15m markets)
    store.py              sqlite schema + writers
  web/
    server.py             monitor API + SSE (stdlib http.server)
    ui.html               dashboard (single file, no CDN)
```

---

*Paper trading only. ไม่ใช่คำแนะนำการลงทุน — prediction market และ memecoin มีความเสี่ยงสูงมาก*

# Binance Strict Crypto Scanner → Telegram

Bu bot Binance Spot public API üzerinden en yüksek hacimli USDT paritelerini tarar, TradingView'de kurduğumuz STRICT BUY mantığını Python'da hesaplar ve Telegram'a mesaj gönderir.

## Mantık

STRICT BUY için ana koşullar:

- Bull score >= 7/10
- HTF = BULL
- ER / Chop >= 0.18
- ATR Regime = OK
- Price > KAMA
- KAMA = GREEN
- State = FLAT
- Cooldown tamamlanmış
- Sadece kapanmış 5m mumlar kullanılır

## Kurulum

```bash
cd crypto_scanner_bot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`.env` dosyasını açıp şunları doldur:

```env
TELEGRAM_BOT_TOKEN=BotFather_token
TELEGRAM_CHAT_ID=senin_chat_id
```

## Telegram Chat ID bulma

1. BotFather'dan bot oluştur.
2. Botuna Telegram'da `/start` yaz.
3. Şu URL'yi tarayıcıda aç:

```text
https://api.telegram.org/botBOT_TOKEN/getUpdates
```

`chat":{"id":...}` içindeki sayı Chat ID'dir.

Grup/kanal için botu gruba ekle, bir mesaj yaz, aynı `getUpdates` çıktısında negatif veya `-100...` başlayan chat id'yi kullan.

## Çalıştırma

```bash
python main.py
```

Test için önce:

```env
DRY_RUN=true
```

kullan. Telegram'a göndermeden terminale mesaj basar.

## VPS'te sürekli çalıştırma

Basit yöntem:

```bash
nohup python main.py > scanner.log 2>&1 &
```

Daha sağlam yöntem: systemd servis dosyası oluştur.

## Notlar

- Binance API key gerekmez. Sadece public market data kullanılır.
- Bot trade açmaz, sadece sinyal gönderir.
- Liste statik değil; her taramada 24h quoteVolume'a göre top USDT pariteleri seçilir.
- `signals_state.json` aynı coin için tekrar tekrar BUY spam atmayı engeller.

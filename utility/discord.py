# Discord Bot - Push Message to Channel

import requests
import json
import datetime as dt
import pytz

class Discord:
    def __init__(self, _debug):
        # Parse Discord Config JSON
        with open('config/discord_config.json', 'r') as file:
            _config = json.load(file)

        # Set Bot Token API
        self._botToken = _config['bot_token']
        self._headers = {
            "Authorization": f"Bot {self._botToken}",
            "User-Agent": "FlowBot v.0.1.alpha",
        }

        # Set Channel to Publish Config
        _ch = _config['debug_ch'] if _debug else _config['prod_ch']
        self._baseURL = f"https://discord.com/api/channels/{_ch}/messages"

    def publish_embed(self, embed):
        response = requests.post(self._baseURL, headers=self._headers, json=embed)
        if response.status_code == 200:
            print("Message sent successfully!")
        else:
            print(f"Failed to send message: {response.status_code} - {response.text}")

    def publish_file_embed(self, file_path, embed):
        with open(file_path, 'rb') as file:
            # Correctly attach file and embed JSON payload
            response = requests.post(
                self._baseURL,
                headers=self._headers,  # Ensure this contains the 'Content-Type: multipart/form-data' if needed
                files={
                    "file": ("chart.png", file, "image/png"),  # Attach image as "chart.png"
                },
                data={"payload_json": json.dumps(embed)},  # Embed JSON must go here
            )
            if response.status_code == 200:
                print("Message sent successfully!")
            else:
                print(f"Failed to send message: {response.status_code} - {response.text}")

    def publish_trade_embed(self, trade):
        """
        trade: dict with keys like entry, stop, target, tier, trigger_reason
        Returns a Discord embed JSON payload.
        """
        # --- Calculate Notional Size (BETA) ---
        capital = 40 # USD
        risk = capital * 0.025
        rpu = trade.get("entry_price") - trade.get("stop")
        notional_size = risk/rpu * trade.get("entry_price")

        # --- Convert UTC timestamp to GMT+8 ---
        entry_ts = trade.get("entry_ts")

        if isinstance(entry_ts, str):
            dt_utc = dt.datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
        elif isinstance(entry_ts, dt.datetime):
            dt_utc = entry_ts if entry_ts.tzinfo else entry_ts.replace(tzinfo=pytz.UTC)
        else:
            dt_utc = None
            
        if dt_utc is not None:
            tz_gmt8 = pytz.timezone("Asia/Singapore")  # GMT+8
            dt_gmt8 = dt_utc.astimezone(tz_gmt8)
            ts_str = dt_gmt8.strftime("%Y-%m-%d %H:%M")
        else:
            ts_str = "N/A"

        def fmt_num(val):
            return f"{val:.2f}" if isinstance(val, (int, float)) else "N/A"

        # --- Build Discord embed ---
        embed = {
            "content": None,
            "embeds": [
                {
                    "title": f"🚨 New {trade['tier']}-Tier Trade Alert",
                    "color": 0xFF0000 if trade['direction'] == "short" else 0x00FF00,
                    "fields": [
                        {"name": "Entry", "value": fmt_num(trade.get("entry_price")), "inline": True},
                        {"name": "Stop", "value": fmt_num(trade.get("stop")), "inline": True},
                        {"name": "Target", "value": fmt_num(trade.get("target")), "inline": True},
                        {"name": "Trigger", "value": str(trade.get("trigger")), "inline": True},
                        {"name": "Market Context", "value": str(trade.get("market_context")), "inline": False}
                    ],
                    "footer": {
                        "text": f"Triggered at {ts_str} GMT+8"
                    }
                }
            ]
        }

        # Publish to Channel
        self.publish_embed(embed)
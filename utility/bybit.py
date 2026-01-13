import json
from pybit.unified_trading import HTTP

# ---------------------------
# BybitTrader: small wrapper
# ---------------------------
class BybitTrader:
    def __init__(self, key_file="keys.json", testnet=False, symbol="BTCUSDT"):
        with open(key_file, "r") as f:
            keys = json.load(f)
        self.session = HTTP(testnet=testnet,
                            demo = False,
                            api_key=keys["api_key"],
                            api_secret=keys["api_secret"])
        self.symbol = symbol

    def get_ticker_price(self):
        resp = self.session.get_tickers(category="linear", symbol=self.symbol)
        item = resp["result"]["list"][0]
        return float(item["lastPrice"])

    def _safe_float(self, x, default=None):
        try:
            if x is None or x == "":
                return default
            return float(x)
        except Exception:
            return default

    # ----- Core functions -----
    def get_capital(self, prefer_equity=True):
        """Parse Bybit get_wallet_balance() response and return capital in USDT."""
        resp = self.session.get_wallet_balance(accountType="UNIFIED", coin="USDT")

        result = resp.get("result", {}) or {}
        lst = result.get("list") or []
        if not lst:
            raise ValueError("No 'result.list' in response")

        acct = lst[0]

        if not prefer_equity:
            for key in ("totalAvailableBalance", "totalWalletBalance",
                        "totalMarginBalance", "totalEquity"):
                v = acct.get(key)
                f = self._safe_float(v)
                if f is not None:
                    return f
        else:
            for key in ("totalEquity", "totalWalletBalance", "totalAvailableBalance"):
                v = acct.get(key)
                f = self._safe_float(v)
                if f is not None:
                    return f

        coins = acct.get("coin") or []
        for c in coins:
            if c.get("coin") in ("USDT", "usdt", None):
                for ck in ("equity", "walletBalance", "usdValue", "availableToWithdraw"):
                    fv = self._safe_float(c.get(ck))
                    if fv is not None:
                        return fv

        raise ValueError("Could not locate USDT capital in the response")

    def place_market_order(self, qty_contracts: float, side: str, tp=None, sl=None, reduceOnly=False):
        params = dict(
            category="linear",
            symbol=self.symbol,
            side=side,
            orderType="Market",
            qty=str(qty_contracts),
            timeInForce="GoodTillCancel",
        )

        # Hedge mode support
        if side == "Buy":
            params["positionIdx"] = 1
        else:
            params["positionIdx"] = 2

        if tp is not None:
            params["takeProfit"] = str(tp)
            params["tpTriggerBy"] = "LastPrice"
        if sl is not None:
            params["stopLoss"] = str(sl)
            params["slTriggerBy"] = "LastPrice"
        if reduceOnly:
            params["reduceOnly"] = True

        return self.session.place_order(**params)

    def set_trading_stop(self, stopLoss=None, takeProfit=None):
        params = dict(category="linear", symbol=self.symbol)
        if stopLoss is not None:
            params["stopLoss"] = str(stopLoss)
            params["slTriggerBy"] = "LastPrice"
        if takeProfit is not None:
            params["takeProfit"] = str(takeProfit)
            params["tpTriggerBy"] = "LastPrice"
        return self.session.set_trading_stop(**params)

    def get_positions(self):
        return self.session.get_positions(category="linear", symbol=self.symbol)

    def get_position_info(self):
        """Return normalized info for the current position or None if flat."""
        resp = self.get_positions()
        items = resp.get("result", {}).get("list", [])
        if not items:
            return None
        pos = items[0]
        size = self._safe_float(pos.get("size"), 0)
        if size == 0:
            return None
        return {
            "side": pos.get("side"),
            "size": size,
            "avg_price": self._safe_float(pos.get("avgPrice")),
            "liq_price": self._safe_float(pos.get("liqPrice")),
            "stop_loss": self._safe_float(pos.get("stopLoss")),
            "take_profit": self._safe_float(pos.get("takeProfit")),
        }

    def get_last_closed_pnl(self, limit=1):
        """
        Get the most recent closed trade record.
        Returns dict like:
        {"side": "Buy", "closed_size": 0.01, "avg_entry": 27000, "avg_exit": 27100,
         "closed_pnl": 10.5, "reason": "TP"}  # reason may be TP/SL/manual/liq
        """
        resp = self.session.get_closed_pnl(category="linear",
                                           symbol=self.symbol,
                                           limit=limit)
        records = resp.get("result", {}).get("list", [])
        if not records:
            return None
        r = records[0]
        reason_map = {
            "TakeOver": "manual",
            "TP": "tp",
            "SL": "sl",
            "Liq": "liquidation",
        }
        reason = reason_map.get(r.get("execType"), r.get("execType"))
        return {
            "side": r.get("side"),
            "closed_size": self._safe_float(r.get("qty")),
            "avg_entry": self._safe_float(r.get("avgEntryPrice")),
            "avg_exit": self._safe_float(r.get("avgExitPrice")),
            "closed_pnl": self._safe_float(r.get("closedPnl")),
            "reason": reason,
            "time": r.get("updatedTime"),
        }

    def cancel_all_orders(self):
        return self.session.cancel_all_orders(category="linear", symbol=self.symbol)

    def set_leverage_to_max(self):
        info = self.session.get_instruments_info(category="linear", symbol=self.symbol)
        maxLev = info["result"]["list"][0]["leverageFilter"]["maxLeverage"]
        return self.session.set_leverage(category="linear", symbol=self.symbol,
                                         buyLeverage=str(maxLev), sellLeverage=str(maxLev))

    def get_instrument_filters(self):
        info = self.session.get_instruments_info(category="linear", symbol=self.symbol)
        return info["result"]["list"][0]  # contains lot/qty/leverage filters

import os
import time
from datetime import datetime, timedelta

import requests
import pandas as pd
import yfinance as yf

from .config import FINMIND_URL


def fetch_finmind(
    dataset: str,
    stock_id: str,
    start_date: str,
    timeout: int = 30,
    max_retries: int = 2,
) -> pd.DataFrame:
    """FinMind GET with retry + timeout 30s.
    對 timeout / connection error 自動 retry，間隔 exponential backoff (1s, 2s)。
    """
    params = {
        "dataset": dataset,
        "data_id": stock_id,
        "start_date": start_date,
        "token": os.environ["FINMIND_TOKEN"],
    }
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(FINMIND_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return pd.DataFrame(r.json().get("data", []))
        except (
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ) as e:
            last_err = e
            if attempt < max_retries:
                wait = 1.0 * (2 ** attempt)
                print(
                    f"[finmind] {dataset}/{stock_id} {type(e).__name__}, "
                    f"retry {attempt + 1}/{max_retries} after {wait:.1f}s"
                )
                time.sleep(wait)
                continue
            raise
    if last_err:
        raise last_err
    return pd.DataFrame()


def get_price_history(stock_id: str, years: int = 3) -> pd.DataFrame:
    
    stock_id = stock_id.strip()

    # 💡 判斷：如果是英文（美股），改用 yfinance 抓取
    if not stock_id.isdigit():
        print(f"[yfinance] 正在抓取美股歷史股價: {stock_id}")
        try:
            ticker = yf.Ticker(stock_id)
            # 依據傳入的 years 決定抓取區間
            period_str = f"{years}y" if years <= 3 else "5y"
            history = ticker.history(period=period_str)
            
            if history.empty:
                return pd.DataFrame()
                
            # 把 yfinance 的資料結構化妝成跟台股一模一樣
            df = pd.DataFrame()
            df["date"] = pd.to_datetime(history.index.date)
            df["stock_id"] = stock_id
            df["open"] = history["Open"].values
            df["high"] = history["High"].values
            df["low"] = history["Low"].values
            df["close"] = history["Close"].values
            df["volume"] = history["Volume"].values
            
            df = df.sort_values("date").reset_index(drop=True)
            
            # 確保型態皆為數字
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
        except Exception as e:
            print(f"[yfinance] 抓取美股 {stock_id} 失敗: {e}")
            return pd.DataFrame()
    
    start = (datetime.now() - timedelta(days=365 * years + 60)).strftime("%Y-%m-%d")
    df = fetch_finmind("TaiwanStockPrice", stock_id, start)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = df.rename(columns={"max": "high", "min": "low", "Trading_Volume": "volume"})
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_fundamental(stock_id: str) -> dict:
    """近 3 完整年度 EPS、ROE"""
    stock_id = stock_id.strip()
    cy = datetime.now().year

    # 💡 判斷：如果是英文（美股），改用 yfinance 撈取美股財報
    if not stock_id.isdigit():
        print(f"[yfinance] 正在抓取美股基本面: {stock_id}")
        try:
            ticker = yf.Ticker(stock_id)
            # 撈取年度財報與資產負債表
            financials = ticker.financials      # 包含 Net Income, EPS 等
            balancesheet = ticker.balancesheet  # 包含 Total Stockholders Equity
            
            eps_dict = {}
            roe_dict = {}
            
            # yfinance 財報欄位通常是以 Timestamp 為 columns (例如 2024-12-31)
            for col_date in financials.columns:
                year = col_date.year
                # 篩選近 3 年
                if cy - 3 <= year < cy:
                    # 1. 抓取 EPS (Basic EPS 或 Diluted EPS)
                    if "Basic EPS" in financials.index:
                        eps_val = financials.loc["Basic EPS", col_date]
                    elif "Diluted EPS" in financials.index:
                        eps_val = financials.loc["Diluted EPS", col_date]
                    else:
                        eps_val = None
                        
                    # 2. 計算 ROE = Net Income / Total Stockholders Equity
                    if "Net Income" in financials.index and "Stockholders Equity" in balancesheet.index:
                        net_income = financials.loc["Net Income", col_date]
                        equity = balancesheet.loc["Stockholders Equity", col_date]
                        # 轉成百分比 (%)
                        roe_val = (net_income / equity) * 100 if equity and not pd.isna(equity) else None
                    else:
                        roe_val = None
                        
                    if pd.notna(eps_val):
                        eps_dict[year] = round(float(eps_val), 2)
                    if pd.notna(roe_val):
                        roe_dict[year] = round(float(roe_val), 2)
                        
            return {"eps": eps_dict, "roe": roe_dict}
        except Exception as e:
            print(f"[yfinance] 抓取美股基本面 {stock_id} 失敗: {e}")
            return {"eps": {}, "roe": {}}

    start = f"{datetime.now().year - 4}-01-01"
    df = fetch_finmind("TaiwanStockFinancialStatements", stock_id, start)
    if df.empty:
        return {"eps": {}, "roe": {}}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    eps = df[df["type"] == "EPS"].groupby("year")["value"].sum().to_dict()
    roe = df[df["type"] == "ROE"].groupby("year")["value"].sum().to_dict()

    cy = datetime.now().year
    return {
        "eps": {y: round(v, 2) for y, v in eps.items() if cy - 3 <= y < cy},
        "roe": {y: round(v, 2) for y, v in roe.items() if cy - 3 <= y < cy},
    }

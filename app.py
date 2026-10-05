# -*- coding: utf-8 -*-
"""
实验中心 - 量化研究工作台 (云端修复版)
"""
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import akshare as ak
import yfinance as yf

st.set_page_config(page_title="实验中心", page_icon="📊", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@st.cache_data(ttl=1800, show_spinner=False)
def load_a_stock(code: str, start: str, end: str) -> pd.DataFrame:
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date","open","close","high","low","volume"]].set_index("date")
    except Exception:
        ycode = code + (".SS" if code.startswith(("6","9")) else ".SZ")
        df = yf.download(ycode, start=start, end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        return df[["open","close","high","low","volume"]].dropna()

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for n in (5, 10, 20, 60):
        out[f"MA{n}"] = out["close"].rolling(n).mean()
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    out["DEA"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["MACD"] = 2 * (out["DIF"] - out["DEA"])
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - 100 / (1 + rs)
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["BOLL_UP"] = mid + 2*std
    out["BOLL_MID"] = mid
    out["BOLL_LOW"] = mid - 2*std
    out["ret"] = out["close"].pct_change()
    return out

st.sidebar.title("📊 实验中心")
code = st.sidebar.text_input("股票代码", value="600519")
start = st.sidebar.date_input("开始日期", value=pd.Timestamp("2025-01-01"))
end = st.sidebar.date_input("结束日期", value=pd.Timestamp.today())
if st.sidebar.button("🔄 刷新数据"):
    st.cache_data.clear()

tab_kline, tab_ind, tab_stats, tab_export = st.tabs(["🕯 K线", "📈 指标", "📊 统计", "📥 导出"])

try:
    raw = load_a_stock(code, str(start), str(end))
    if raw is None or raw.empty:
        st.error(f"未获取到 {code} 的数据，请检查代码或日期范围")
        st.stop()
    df = calc_indicators(raw)
    st.sidebar.success(f"✅ {len(df)} 条日线 ({df.index[0].date()} ~ {df.index[-1].date()})")

    with tab_kline:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
        ax = axes[0]
        up = df["close"] >= df["open"]
        ax.vlines(df.index, df["low"], df["high"], color="#999", linewidth=0.5)
        ax.bar(df.index[up], (df["close"]-df["open"])[up], 0.6, bottom=df["open"][up], color="#ef5350")
        ax.bar(df.index[~up], (df["close"]-df["open"])[~up], 0.6, bottom=df["open"][~up], color="#26a69a")
        for n, c in [(20,"#f39c12"), (60,"#3498db")]:
            ax.plot(df.index, df[f"MA{n}"], label=f"MA{n}", color=c, linewidth=1)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{code} K-line", fontsize=13)
        ax.grid(alpha=0.3)
        axes[1].bar(df.index[up], df["volume"][up], 0.6, color="#ef5350", alpha=0.7)
        axes[1].bar(df.index[~up], df["volume"][~up], 0.6, color="#26a69a", alpha=0.7)
        axes[1].set_title("Volume", fontsize=10)
        axes[1].grid(alpha=0.3)
        axes[2].plot(df.index, df["DIF"], label="DIF", color="#f39c12", linewidth=1)
        axes[2].plot(df.index, df["DEA"], label="DEA", color="#3498db", linewidth=1)
        macd_up = df["MACD"] >= 0
        axes[2].bar(df.index[macd_up], df["MACD"][macd_up], 0.6, color="#ef5350", alpha=0.7)
        axes[2].bar(df.index[~macd_up], df["MACD"][~macd_up], 0.6, color="#26a69a", alpha=0.7)
        axes[2].axhline(0, color="#666", linewidth=0.5)
        axes[2].legend(loc="upper left", fontsize=8)
        axes[2].set_title("MACD", fontsize=10)
        axes[2].grid(alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ 下载K线图", buf, file_name=f"{code}_kline.png", mime="image/png")
        plt.close(fig)

    with tab_ind:
        last = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盘", f"{last['close']:.2f}", f"{last['ret']*100:+.2f}%" if not pd.isna(last['ret']) else "")
        c2.metric("RSI(14)", f"{last['RSI']:.1f}" if not pd.isna(last['RSI']) else "N/A")
        ret = df["ret"].dropna()
        c3.metric("年化波动", f"{ret.std()*np.sqrt(252)*100:.1f}%" if len(ret)>1 else "N/A")
        if not pd.isna(last['BOLL_UP']) and last['BOLL_UP'] != last['BOLL_LOW']:
            c4.metric("布林位置", f"{(last['close']-last['BOLL_LOW'])/(last['BOLL_UP']-last['BOLL_LOW'])*100:.0f}%")
        else:
            c4.metric("布林位置", "N/A")
        st.subheader("RSI(14)")
        st.line_chart(df["RSI"].tail(120).dropna(), height=200)
        st.subheader("MACD")
        st.area_chart(df[["DIF","DEA","MACD"]].tail(120).dropna(), height=250)
        st.subheader("布林带")
        st.line_chart(df[["close","BOLL_UP","BOLL_LOW"]].tail(120).dropna(), height=250)
        signals = []
        if not pd.isna(last['RSI']):
            if last['RSI'] > 70: signals.append("⚠️ RSI超买(>70)")
            elif last['RSI'] < 30: signals.append("✅ RSI超卖(<30)")
        if len(df) >= 2:
            if last['DIF'] > last['DEA'] and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
                signals.append("🔔 MACD金叉")
            elif last['DIF'] < last['DEA'] and df['DIF'].iloc[-2] >= df['DEA'].iloc[-2]:
                signals.append("🔔 MACD死叉")
        st.info(" | ".join(signals) if signals else "当前无强信号")

    with tab_stats:
        ret = df["ret"].dropna()
        if len(ret) > 2:
            ann_ret = (1+ret).prod() ** (252/len(ret)) - 1
            ann_vol = ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            downside = ret[ret < 0].std() * np.sqrt(252)
            sortino = ann_ret / downside if downside and downside > 0 else 0
            cummax = df["close"].cummax()
            dd = df["close"]/cummax - 1
            mdd = dd.min()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("年化收益", f"{ann_ret*100:.1f}%")
            m2.metric("年化波动", f"{ann_vol*100:.1f}%")
            m3.metric("Sharpe", f"{sharpe:.2f}")
            m4.metric("Sortino", f"{sortino:.2f}")
            m5, m6 = st.columns(2)
            m5.metric("最大回撤", f"{mdd*100:.1f}%")
            pos, neg = ret[ret>0].mean(), abs(ret[ret<0].mean())
            m6.metric("盈亏比", f"{pos/neg:.2f}" if neg and neg > 0 else "N/A")
            st.subheader("回撤曲线")
            st.area_chart(dd, height=200, color="#ef5350")
            st.subheader("累计收益")
            st.line_chart(df["close"]/df["close"].iloc[0] - 1, height=250)
        else:
            st.warning("数据不足2条，无法计算统计指标")

    with tab_export:
        st.subheader("📥 导出Excel报告")
        buf_x = io.BytesIO()
        with pd.ExcelWriter(buf_x, engine="openpyxl") as w:
            df.to_excel(w, sheet_name="日线数据")
            summary = pd.DataFrame({
                "指标": ["数据条数","最新收盘","RSI(14)"],
                "数值": [len(df), f"{df['close'].iloc[-1]:.2f}", f"{df['RSI'].iloc[-1]:.1f}" if not pd.isna(df['RSI'].iloc[-1]) else "N/A"]
            })
            summary.to_excel(w, sheet_name="摘要", index=False)
        buf_x.seek(0)
        st.download_button("⬇️ 下载Excel", buf_x, file_name=f"{code}_report.xlsx",
                          mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption(f"生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")

except Exception as e:
    st.error(f"数据加载失败: {type(e).__name__}: {e}")
    with st.expander("详细错误"):
        import traceback
        st.code(traceback.format_exc(), language="python")

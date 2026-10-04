# -*- coding: utf-8 -*-
"""
实验中心 - 量化研究工作台
启动: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 全局配置 ----------
st.set_page_config(page_title="实验中心", page_icon="📊", layout="wide")

# 中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据源
import akshare as ak
import yfinance as yf

@st.cache_data(ttl=3600)
def load_a_stock(code: str, start: str, end: str) -> pd.DataFrame:
    """加载A股日线数据（akshare主源，yfinance备源）"""
    try:
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close","最高":"high","最低":"low","成交量":"volume"})
        df["date"] = pd.to_datetime(df["date"])
        return df[["date","open","close","high","low","volume"]].set_index("date")
    except Exception:
        ycode = code + (".SS" if code.startswith(("6","9")) else ".SZ")
        df = yf.download(ycode, start=start, end=end, progress=False, auto_adjust=True)
        df.columns = [c.lower() if isinstance(c,str) else c for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["open","close","high","low","volume"]].dropna()

def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算核心技术指标"""
    out = df.copy()
    for n in (5, 10, 20, 60):
        out[f"MA{n}"] = out["close"].rolling(n).mean()
    # MACD
    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["DIF"] = ema12 - ema26
    out["DEA"] = out["DIF"].ewm(span=9, adjust=False).mean()
    out["MACD"] = 2 * (out["DIF"] - out["DEA"])
    # RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - 100 / (1 + rs)
    # 布林带
    mid = out["close"].rolling(20).mean()
    std = out["close"].rolling(20).std()
    out["BOLL_UP"] = mid + 2*std
    out["BOLL_MID"] = mid
    out["BOLL_LOW"] = mid - 2*std
    # 波动率/年化
    out["ret"] = out["close"].pct_change()
    out["vol_annual"] = out["ret"].rolling(20).std() * np.sqrt(252)
    return out

# ---------- 侧边栏 ----------
st.sidebar.title("📊 实验中心")
code = st.sidebar.text_input("股票代码", value="600519", help="A股代码，如 600519 茅台 / 510300 ETF")
start = st.sidebar.date_input("开始日期", value=pd.Timestamp("2025-01-01"))
end = st.sidebar.date_input("结束日期", value=pd.Timestamp.today())
st.sidebar.caption("数据源: akshare(主) / yfinance(备)")

tab_kline, tab_ind, tab_stats, tab_export = st.tabs(["🕯 K线", "📈 指标", "📊 统计", "📥 导出"])

try:
    raw = load_a_stock(code, str(start), str(end))
    if raw.empty:
        st.error("未获取到数据，请检查代码或日期")
        st.stop()
    df = calc_indicators(raw)
    st.sidebar.success(f"✅ {len(df)} 条日线数据 ({df.index[0].date()} ~ {df.index[-1].date()})")

    # ================= K线 =================
    with tab_kline:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                                 gridspec_kw={'height_ratios': [3, 1, 1]})
        ax = axes[0]
        # 蜡烛
        up = df["close"] >= df["open"]
        ax.vlines(df.index, df["low"], df["high"], color="#999", linewidth=0.5)
        ax.bar(df.index[up], (df["close"]-df["open"])[up], 0.6, bottom=df["open"][up], color="#ef5350")
        ax.bar(df.index[~up], (df["close"]-df["open"])[~up], 0.6, bottom=df["open"][~up], color="#26a69a")
        for n, c in [(20,"#f39c12"), (60,"#3498db")]:
            ax.plot(df.index, df[f"MA{n}"], label=f"MA{n}", color=c, linewidth=1)
        ax.legend(loc="upper left", fontsize=8)
        ax.set_title(f"{code} 日K线", fontsize=13)
        ax.grid(alpha=0.3)

        # 成交量
        axes[1].bar(df.index[up], df["volume"][up], 0.6, color="#ef5350", alpha=0.7)
        axes[1].bar(df.index[~up], df["volume"][~up], 0.6, color="#26a69a", alpha=0.7)
        axes[1].set_title("成交量", fontsize=10)
        axes[1].grid(alpha=0.3)

        # MACD
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
        st.pyplot(fig, use_container_width=True)
        # 下载K线图
        buf = pd.io.formats.style
        fig.savefig("kline_charts/kline.png", bbox_inches="tight")
        with open("kline_charts/kline.png", "rb") as f:
            st.download_button("⬇️ 下载K线图", f, file_name="kline.png", mime="image/png")

    # ================= 指标 =================
    with tab_ind:
        last = df.iloc[-1]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("最新收盘", f"{last['close']:.2f}", f"{last['ret']*100:+.2f}%")
        c2.metric("RSI(14)", f"{last['RSI']:.1f}", "超买>70 超卖<30" if np.isnan(last['RSI'])==False else "")
        c3.metric("年化波动率", f"{last['vol_annual']*100:.1f}%")
        c4.metric("布林带位置", f"{(last['close']-last['BOLL_LOW'])/(last['BOLL_UP']-last['BOLL_LOW'])*100:.0f}%")

        st.subheader("📉 RSI(14)")
        st.line_chart(df["RSI"].tail(120), height=200)
        st.subheader("📉 MACD")
        st.area_chart(df[["DIF","DEA","MACD"]].tail(120), height=250)
        st.subheader("📉 布林带")
        st.line_chart(df[["close","BOLL_UP","BOLL_LOW"]].tail(120), height=250)

        # 信号提示
        signals = []
        if not np.isnan(last['RSI']):
            if last['RSI'] > 70: signals.append("⚠️ RSI超买 (>70)")
            elif last['RSI'] < 30: signals.append("✅ RSI超卖 (<30)")
        if last['DIF'] > last['DEA'] and df['DIF'].iloc[-2] <= df['DEA'].iloc[-2]:
            signals.append("🔔 MACD金叉")
        elif last['DIF'] < last['DEA'] and df['DIF'].iloc[-2] >= df['DEA'].iloc[-2]:
            signals.append("🔔 MACD死叉")
        if signals:
            st.info(" | ".join(signals))
        else:
            st.caption("当前无强信号")

    # ================= 统计 =================
    with tab_stats:
        ret = df["ret"].dropna()
        ann_ret = (1+ret).prod() ** (252/len(ret)) - 1
        ann_vol = ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        downside = ret[ret < 0].std() * np.sqrt(252)
        sortino = ann_ret / downside if downside > 0 else 0
        # 最大回撤
        cummax = df["close"].cummax()
        dd = df["close"]/cummax - 1
        mdd = dd.min()

        st.subheader("📊 风险收益指标")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("年化收益", f"{ann_ret*100:.1f}%")
        m2.metric("年化波动", f"{ann_vol*100:.1f}%")
        m3.metric("Sharpe", f"{sharpe:.2f}", "无风险利率按0计")
        m4.metric("Sortino", f"{sortino:.2f}")
        m5, m6 = st.columns(2)
        m5.metric("最大回撤", f"{mdd*100:.1f}%", delta_color="inverse")
        m6.metric("盈亏比", f"{ret[ret>0].mean()/abs(ret[ret<0].mean()):.2f}" if (ret<0).any() else "N/A")

        st.subheader("📉 回撤曲线")
        st.area_chart(dd, height=200, color="#ef5350")
        st.subheader("📉 累计收益")
        st.line_chart((df["close"]/df["close"].iloc[0] - 1), height=250)

    # ================= 导出 =================
    with tab_export:
        st.subheader("📥 导出Excel报告")
        with pd.ExcelWriter("xlsx_reports/report.xlsx", engine="openpyxl") as w:
            df.to_excel(w, sheet_name="日线数据")
            summary = pd.DataFrame({
                "指标": ["最新收盘","年化收益","年化波动","Sharpe","Sortino","最大回撤","RSI(14)"],
                "数值": [f"{last['close']:.2f}", f"{ann_ret*100:.2f}%", f"{ann_vol*100:.2f}%",
                        f"{sharpe:.2f}", f"{sortino:.2f}", f"{mdd*100:.2f}%", f"{last['RSI']:.1f}"]
            })
            summary.to_excel(w, sheet_name="风险指标", index=False)
        with open("xlsx_reports/report.xlsx", "rb") as f:
            st.download_button("⬇️ 下载Excel", f, file_name=f"{code}_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.caption(f"输出目录: xlsx_reports/  |  生成时间: {pd.Timestamp.now():%Y-%m-%d %H:%M}")

except Exception as e:
    st.error(f"数据加载失败: {e}")
    st.code(str(e), language="python")

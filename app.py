# -*- coding: utf-8 -*-
"""
资产配置分析（云端部署版）
====================================
面向 Streamlit Community Cloud 的独立版本，与本地真实数据版本相互独立。

模型：双资产三组合对比
- 资产：沪深300指数 + 中债-7-10年国债指数（财富指数口径，CBA06501.CB）
- 组合A：基准，固定 40% 沪深300 / 60% 债券
- 组合B：自定义权重（滑块控制股票占比 0%~100%，步长1%）
- 组合C：最小方差优化权重（约束：权重之和 = 1，不允许做空）

数据来源：
1. 默认：程序内置生成的模拟演示数据（随机种子固定，界面明确标注，不代表任何真实行情）；
2. 可选：上传 iFinD 导出的 Excel/CSV（须含「时间」与「收盘价(元)」两列，其余列自动忽略）。
   本版本不读取任何本地数据文件、不联网抓取行情、不包含任何凭据。

⚠️ 债券指数口径（重要）：
中债-7-10年国债指数 财富指数代码 CBA06501，净价指数 CBA06502，全价指数 CBA06503。
财富指数以全价计算并计入票息再投资的现金流入，可作为含息总回报口径；
净价指数不含应计利息，不能与财富指数混用。若上传的是到期收益率(YTM)，
它只是利率水平而非投资回报，界面会拦截并要求显式确认近似口径。

计算假设：
1. 日度简单收益率 r_t = P_t / P_{t-1} - 1（P 为相邻交易日指数点位）。
2. 组合每日按目标权重再平衡（每日固定权重）：组合日收益 = Σ w_i × r_i。
3. 年化按 252 个交易日；年化波动率 = 日收益样本标准差(ddof=1) × √252。
4. 几何平均日收益 = (1 + 区间累计收益)^(1/N) - 1，N 为日收益样本数。
5. 年化收益率 = (1 + 几何平均日收益)^252 - 1。
6. 最小方差权重：由日收益协方差矩阵（×252 年化）解析求解二资产全局最小方差组合，
   并将权重截断至 [0, 1] 以满足不做空约束。
"""

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from asset_data_loader import PRICE_NAMES, YTM_NAMES, align_and_calculate, load_ifind_prices

TRADING_DAYS = 252
STOCK_NAME = "沪深300"
BOND_NAME = "中债7-10年国债指数"
BOND_CODE = "CBA06501.CB"

# ----------------------- 页面配置 -----------------------
st.set_page_config(page_title="资产配置分析", page_icon="📊", layout="wide")

st.title("资产配置分析")
st.caption(
    "双资产组合：A 基准40/60 ｜ B 自定义权重 ｜ C 最小方差优化。仅供研究参考，不构成投资建议。"
)

# ----------------------- 侧边栏 -----------------------
with st.sidebar:
    st.header("1️⃣ 数据来源")
    source = st.radio(
        "选择使用哪套数据",
        ["模拟演示数据（默认，非真实行情）", "上传的真实数据（请使用获准分享的文件）"],
        index=0,
        help=(
            "云端版默认使用内置的模拟演示数据；如需真实分析，"
            "请上传你获准使用的 iFinD 导出行情文件。"
        ),
    )
    use_mock = source.startswith("模拟")

    st.header("2️⃣ 上传真实数据（可选）")
    st.markdown(
        "文件须包含 **「时间」** 与 **「收盘价(元)」** 两列；"
        "代码、简称、涨跌幅等其他列会自动忽略。两个资产**都要上传**才能计算。\n\n"
        "⚠️ 请只上传你获准公开使用的行情数据。"
    )
    up_stock = st.file_uploader(
        "沪深300指数（000300.SH）", type=["xlsx", "xls", "csv"], key="stock"
    )
    up_bond = st.file_uploader(
        f"{BOND_NAME}（{BOND_CODE}）", type=["xlsx", "xls", "csv"], key="bond"
    )

    st.header("3️⃣ 债券数据口径")
    bond_kind = st.radio(
        "债券数据口径确认",
        ["中债-7-10年国债指数（财富指数，含票息再投资）", "到期收益率 YTM（%）— 仅演示，不推荐"],
        index=0,
    )
    bond_is_ytm = bond_kind.startswith("到期")
    allow_ytm_approx = False
    if bond_is_ytm:
        st.error(
            "⚠️ 到期收益率是利率水平，不是投资回报率，不能直接当收益计算！\n\n"
            "正式分析请上传财富（总值）指数 CBA06501。"
        )
        allow_ytm_approx = st.checkbox(
            "使用粗略近似：日收益 ≈ 当日YTM/252（忽略久期与价格效应，仅供演示）",
            value=False,
        )

    st.header("4️⃣ 历史区间")
    period = st.selectbox(
        "统计区间", ["近10年（默认）", "近5年", "近3年", "全部数据"], index=0
    )

    st.header("5️⃣ 组合B：自定义权重")
    wB_user = st.slider("组合B — 沪深300占比", 0, 100, 60, format="%d%%")
    st.markdown(f"债券权重自动为 **{100 - wB_user}%**（合计 100%）")


# ----------------------- 数据读取 -----------------------
def make_demo_data() -> tuple[pd.Series, pd.Series]:
    """生成明确标注的模拟演示数据（随机种子固定，可复现；非真实行情）。"""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2016-01-04", periods=2600)  # 约10年
    n = len(dates)
    stock = 3000 * np.cumprod(1 + rng.normal(0.08 / 252, 0.18 / np.sqrt(252), n))
    bond = 100 * np.cumprod(1 + rng.normal(0.035 / 252, 0.03 / np.sqrt(252), n))
    return (
        pd.Series(stock, index=dates, name="沪深300"),
        pd.Series(bond, index=dates, name="债券"),
    )


meta_stock = meta_bond = None
stock_source_label = bond_source_label = None
stock_src = bond_src = None

if use_mock:
    stock_raw, bond_raw = make_demo_data()
    source_desc = "模拟演示数据（程序随机生成、种子固定可复现，仅用于演示与教学，不代表任何真实行情）"
    if (up_stock is not None) or (up_bond is not None):
        st.warning(
            "已上传文件，但当前选择的是**模拟演示数据**，上传的文件不会被读取。"
            "如需使用你上传的数据，请把左侧「数据来源」切换为「上传的真实数据」。"
        )
else:
    stock_src, stock_source_label = (
        up_stock,
        f"上传文件（{up_stock.name}）" if up_stock is not None else None,
    )
    bond_src, bond_source_label = (
        up_bond,
        f"上传文件（{up_bond.name}）" if up_bond is not None else None,
    )

    missing = []
    if stock_src is None:
        missing.append(f"**{STOCK_NAME}**：未上传行情文件")
    if bond_src is None:
        missing.append(f"**{BOND_NAME}**：未上传行情文件")
    if missing:
        st.error(
            "**「上传的真实数据」缺少文件，未做任何替代（不会自动切换到模拟数据）。**\n\n"
            + "\n\n".join(f"- {m}" for m in missing)
            + "\n\n请在左侧把两个资产的行情文件都上传后再计算；"
            "若只想先看看界面效果，请把「数据来源」切换为模拟演示数据。"
        )
        st.stop()

    try:
        stock_raw, meta_stock = load_ifind_prices(stock_src, return_meta=True)
    except ValueError as exc:
        st.error(
            f"**{STOCK_NAME} 数据读取失败（来源：{stock_source_label}），未做任何替代。**\n\n{exc}"
        )
        st.stop()

    try:
        bond_raw, meta_bond = load_ifind_prices(
            bond_src,
            price_names=YTM_NAMES if bond_is_ytm else PRICE_NAMES,
            return_meta=True,
        )
    except ValueError as exc:
        st.error(
            f"**{BOND_NAME} 数据读取失败（来源：{bond_source_label}），未做任何替代。**\n\n{exc}"
        )
        st.stop()

    source_desc = (
        f"用户上传的真实数据（{STOCK_NAME}：{stock_source_label}；"
        f"{BOND_NAME}：{bond_source_label}）"
    )

# 债券 YTM：必须显式确认近似口径，才能继续
if bond_is_ytm and not use_mock and not allow_ytm_approx:
    st.error(
        "**已检测到债券数据类型为「到期收益率 YTM」。**\n\n"
        "到期收益率衡量的是持有到期的年化回报水平（利率），其日度变化不是资产价格回报，"
        "**不能直接用相邻交易日变动率计算收益率**。\n\n"
        "请改用中债-7-10年国债指数**财富指数 CBA06501**（含票息再投资）；"
        "或勾选左侧「粗略近似」（结果仅供演示，会严重低估久期带来的价格波动）。"
    )
    st.stop()

if use_mock:
    st.warning(
        "**当前数据来源：模拟演示数据**（程序随机生成、种子固定，仅用于演示界面与计算逻辑，"
        f"不代表任何真实行情，请勿据此得出投资结论）。演示区间："
        f"{stock_raw.index[0]:%Y-%m-%d} ~ {stock_raw.index[-1]:%Y-%m-%d}。"
    )
else:
    st.success(
        f"**当前数据来源：上传的真实数据** — {STOCK_NAME} + {BOND_NAME}（{BOND_CODE}）。"
    )
    src_table = pd.DataFrame(
        [
            {
                "资产": STOCK_NAME,
                "来源": stock_source_label,
                "识别列": f"{meta_stock['date_col']} + {meta_stock['price_col']}",
                "有效交易日": meta_stock["rows"],
                "日期区间": f"{stock_raw.index[0]:%Y-%m-%d} ~ {stock_raw.index[-1]:%Y-%m-%d}",
                "已忽略列": "、".join(meta_stock["skipped"]) or "—",
            },
            {
                "资产": BOND_NAME,
                "来源": bond_source_label,
                "识别列": f"{meta_bond['date_col']} + {meta_bond['price_col']}",
                "有效交易日": meta_bond["rows"],
                "日期区间": f"{bond_raw.index[0]:%Y-%m-%d} ~ {bond_raw.index[-1]:%Y-%m-%d}",
                "已忽略列": "、".join(meta_bond["skipped"]) or "—",
            },
        ]
    )
    st.dataframe(src_table, hide_index=True, width="stretch")

if bond_is_ytm and allow_ytm_approx and not use_mock:
    bond_raw = (1 + bond_raw / 100 / TRADING_DAYS).cumprod()
    bond_label = "国债YTM粗略近似（日收益≈YTM/252，非真实回报）"
    bond_desc = "到期收益率 YTM 粗略近似（仅演示，非含息总回报）"
else:
    bond_label = BOND_NAME
    bond_desc = f"{BOND_NAME}（财富指数，{BOND_CODE}，含票息再投资）"

if not bond_is_ytm and not use_mock:
    st.info(
        f"**债券口径**：中债-7-10年国债指数按中债估值中心编制说明分为 财富指数 {BOND_CODE}、"
        "净价指数 CBA06502、全价指数 CBA06503。财富指数以全价计算并计入票息再投资，"
        "可作为**含息总回报**口径；净价指数不含应计利息，两者不可混用。"
    )

# ----------------------- 对齐与区间 -----------------------
try:
    prices_all, _ = align_and_calculate(stock_raw, bond_raw)
except ValueError as exc:
    st.error(f"**两资产无法对齐，未做任何替代。**\n\n{exc}")
    st.stop()

df = prices_all.copy()
end = df.index.max()
years = {"近10年（默认）": 10, "近5年": 5, "近3年": 3}.get(period)
if years:
    df = df[df.index >= end - pd.DateOffset(years=years)]

if len(df) < 30:
    st.error(
        f"选中区间内共同交易日仅 **{len(df)}** 条，样本过少无法统计。"
        "请把「历史区间」改为「全部数据」或上传更长区间的数据。"
    )
    st.stop()

r_stock = df["stock"].pct_change().dropna()
r_bond = df["bond"].pct_change().dropna()
assert r_stock.index.equals(r_bond.index)

st.caption(
    f"对齐结果：两资产按**共同交易日**取交集（缺失日期不填充），全区间共同交易日 "
    f"**{len(prices_all)}** 个（{prices_all.index[0]:%Y-%m-%d} ~ {prices_all.index[-1]:%Y-%m-%d}）；"
    f"当前统计区间（{period}）内为 **{len(df)}** 个交易日、收益率样本 **{len(r_stock)}** 个"
    f"（{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}）。"
)


# ----------------------- 组合 -----------------------
def min_variance_weights(rs: pd.Series, rb: pd.Series) -> tuple[float, float]:
    """二资产全局最小方差组合（解析解），截断至 [0,1] 满足不做空约束。
    w_s = (σ_b² - σ_sb) / (σ_s² + σ_b² - 2σ_sb)，年化协方差。
    """
    cov = np.cov(np.vstack([rs.values, rb.values])) * TRADING_DAYS
    var_s, var_b, cov_sb = cov[0, 0], cov[1, 1], cov[0, 1]
    denom = var_s + var_b - 2 * cov_sb
    w = 0.5 if denom <= 1e-12 else (var_b - cov_sb) / denom
    w = float(min(max(w, 0.0), 1.0))
    return w, 1.0 - w


wA_s, wA_b = 0.40, 0.60
wB_s, wB_b = wB_user / 100, 1 - wB_user / 100
wC_s, wC_b = min_variance_weights(r_stock, r_bond)

rA = wA_s * r_stock + wA_b * r_bond
rB = wB_s * r_stock + wB_b * r_bond
rC = wC_s * r_stock + wC_b * r_bond


# ----------------------- 统计指标 -----------------------
def calc_stats(returns: pd.Series) -> dict:
    n = len(returns)
    cum = float((1 + returns).prod() - 1)
    arith = float(returns.mean())
    geo = (1 + cum) ** (1 / n) - 1
    ann_ret = (1 + geo) ** TRADING_DAYS - 1
    ann_vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    return {
        "算术平均日收益率": arith,
        "几何平均日收益率": geo,
        "年化收益率": ann_ret,
        "年化波动率": ann_vol,
    }


rows = [
    {"名称": f"组合A（基准 {wA_s:.0%}/{wA_b:.0%}）", "权重(股/债)": f"{wA_s:.0%} / {wA_b:.0%}", **calc_stats(rA)},
    {"名称": "组合B（自定义）", "权重(股/债)": f"{wB_s:.0%} / {wB_b:.0%}", **calc_stats(rB)},
    {"名称": "组合C（最小方差）", "权重(股/债)": f"{wC_s:.1%} / {wC_b:.1%}", **calc_stats(rC)},
    {"名称": STOCK_NAME, "权重(股/债)": "—", **calc_stats(r_stock)},
    {"名称": bond_label, "权重(股/债)": "—", **calc_stats(r_bond)},
]
stats_df = pd.DataFrame(rows).set_index("名称")

show_df = stats_df.copy()
for c in ["算术平均日收益率", "几何平均日收益率", "年化收益率", "年化波动率"]:
    show_df[c] = show_df[c].map(lambda x: f"{x*100:.2f}%")

st.header("📐 计算公式与假设")
st.markdown(
    """
**1. 简单收益率**：R = (今天价格 − 昨天价格) / 昨天价格

> R 是当天的收益率；"今天价格""昨天价格"分别是相邻两个交易日的指数点位。

**2. 组合收益率**：Rp = ws × Rs + wb × Rb

> Rs、Rb 分别是沪深300和债券指数当天的简单收益率；ws、wb 是两者的权重，ws + wb = 100%。
> 本应用假设**每日收盘后按目标权重再平衡**（每日固定权重），因此组合日收益始终按此公式加权。

**3. 组合波动率**：σp = √( ws²·σs² + wb²·σb² + 2·ws·wb·ρ·σs·σb )

> σs、σb 是股票和债券收益率的波动率；ρ 是两者收益率的相关系数。
"""
)
with st.expander("🔍 详细计算方法（年化、最小方差优化等）"):
    st.markdown(
        f"""
- **算术平均收益率**：日收益率序列的简单平均，即 R̄ = (1/N)·ΣR_t，N 为日收益样本数（当前 {len(rA)} 个）。
- **几何平均日收益率**：G = (1 + 区间累计收益)^(1/N) − 1，其中区间累计收益 = ∏(1+R_t) − 1。
- **年化收益率** = (1 + 几何平均日收益)^252 − 1（一年按 252 个交易日）。
- **年化波动率** = 日收益率样本标准差（ddof=1）× √252。
- **组合C（最小方差）解析解**：w_股 = (σ²_债 − σ_股债) / (σ²_股 + σ²_债 − 2σ_股债)，
  其中各方差、协方差由日收益样本计算并 ×252 年化；结果截断至 [0, 1] 满足**不做空**约束，权重和恒为 1。
  当前最优权重：**股票 {wC_s:.1%} / 债券 {wC_b:.1%}**。
- **再平衡方式**：每日收盘后按目标权重再平衡（每日固定权重）。注意：与"买入持有不调仓"的结果不同。
- **债券数据类型**：{bond_desc}。
- **数据来源**：{source_desc}。
- **统计区间**：{period}（{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}）。
"""
    )

# ----------------------- 对比表 -----------------------
st.header("📋 组合结果对比")
st.dataframe(show_df, width="stretch")

c1, c2, c3 = st.columns(3)
c1.metric(
    "组合A（基准） 年化收益",
    f"{stats_df.iloc[0]['年化收益率']*100:.2f}%",
    delta=f"波动 {stats_df.iloc[0]['年化波动率']*100:.2f}%",
    delta_color="off",
)
c2.metric(
    "组合B（自定义） 年化收益",
    f"{stats_df.iloc[1]['年化收益率']*100:.2f}%",
    delta=f"波动 {stats_df.iloc[1]['年化波动率']*100:.2f}%",
    delta_color="off",
)
c3.metric(
    "组合C（最小方差） 年化收益",
    f"{stats_df.iloc[2]['年化收益率']*100:.2f}%",
    delta=f"波动 {stats_df.iloc[2]['年化波动率']*100:.2f}%",
    delta_color="off",
)
st.caption(
    f"组合B权重（滑块控制）：沪深300 **{wB_s:.0%}** / 债券 **{wB_b:.0%}**；"
    f"组合C最小方差权重：沪深300 **{wC_s:.1%}** / 债券 **{wC_b:.1%}**"
    f"（约束：权重之和=1，不做空）。所有组合均假设**每日固定权重再平衡**。"
)

# ----------------------- 累计净值 -----------------------
st.header("📉 累计净值曲线（期初 = 1）")
nav = pd.DataFrame(
    {
        f"组合A 基准 {wA_s:.0%}/{wA_b:.0%}": (1 + rA).cumprod(),
        f"组合B 自定义 {wB_s:.0%}/{wB_b:.0%}": (1 + rB).cumprod(),
        "组合C 最小方差": (1 + rC).cumprod(),
        STOCK_NAME: (1 + r_stock).cumprod(),
        bond_label: (1 + r_bond).cumprod(),
    }
)
st.line_chart(nav, height=420)

# ----------------------- 导出 -----------------------
st.header("📤 导出计算结果")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    stats_df.to_excel(writer, sheet_name="组合对比统计")
    nav.to_excel(writer, sheet_name="累计净值")
    pd.DataFrame(
        {
            "沪深300_日简单收益": r_stock,
            f"{bond_label}_日简单收益": r_bond,
            "组合A_加权日收益": rA,
            "组合B_加权日收益": rB,
            "组合C_加权日收益": rC,
        },
        index=r_stock.index,
    ).to_excel(writer, sheet_name="日度收益率")
    pd.DataFrame(
        {
            "项目": [
                "数据来源", "沪深300文件与来源", "沪深300有效交易日",
                "债券文件与来源", "债券有效交易日", "对齐后共同交易日",
                "组合A权重(股/债)", "组合B权重(股/债)", "组合C权重(股/债)",
                "再平衡方式", "收益率口径", "年化交易日", "组合C优化约束",
                "债券数据类型", "统计区间", "导出时间",
            ],
            "内容": [
                "模拟演示数据（非真实行情）" if use_mock else "上传的真实数据（用户自行提供）",
                "—（模拟演示数据）" if use_mock else f"{stock_src.name}｜{stock_source_label}",
                "—（模拟演示数据）" if use_mock else f"{meta_stock['rows']} 个",
                "—（模拟演示数据）" if use_mock else f"{bond_src.name}｜{bond_source_label}",
                "—（模拟演示数据）" if use_mock else f"{meta_bond['rows']} 个",
                f"{len(prices_all)} 个（{prices_all.index[0]:%Y-%m-%d} ~ {prices_all.index[-1]:%Y-%m-%d}）",
                f"{wA_s:.0%} / {wA_b:.0%}",
                f"{wB_s:.0%} / {wB_b:.0%}",
                f"{wC_s:.1%} / {wC_b:.1%}",
                "每日收盘后按目标权重再平衡（每日固定权重）",
                "日度简单收益率 R = (今天价格 - 昨天价格) / 昨天价格",
                "252 个交易日",
                "权重之和=1，不允许做空（权重截断至[0,1]）",
                bond_desc,
                f"{period}：{df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}"
                f"（{len(df)} 个交易日）",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ],
        }
    ).to_excel(writer, sheet_name="计算假设与权重", index=False)

st.download_button(
    "⬇️ 下载 Excel（对比统计 + 净值 + 日收益 + 假设与权重）",
    data=buffer.getvalue(),
    file_name=(
        f"资产配置分析_{'演示' if use_mock else '真实'}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    ),
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

st.markdown("---")
st.caption(
    "免责声明：本应用为公开演示版，默认使用程序生成的模拟演示数据（非真实行情）；"
    "不联网抓取行情、不存储任何用户数据，上传文件仅在当次会话内存中处理。"
    "所有结果仅供研究与教学参考，不构成投资建议。"
)

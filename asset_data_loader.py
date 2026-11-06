"""iFinD 导出文件加载器（集成本地版）

用途：把 iFinD 导出的「时间 / 收盘价(元)」两列行情读成 pandas Series，
并自动忽略代码、简称、涨跌幅等其他列。

设计要点：
1. 只按列名匹配，不依赖列位置；列名做去空格、全角括号兼容处理。
2. iFinD 的 Excel 常在首行放标题（如“数据来源：同花顺iFinD”），
   因此会尝试前几行作为表头，取第一个能识别出日期列与收盘价列的表头。
3. 无法识别时抛出 ValueError，并在消息中列出缺失的列和文件现有列，
   由调用方原样展示给用户——绝不在失败时悄悄切换到模拟数据。
4. 不做任何网络请求，不读取任何凭据。
"""
import io
import re

import pandas as pd

DATE_NAMES = ("时间", "日期", "交易日期", "date", "datetime", "trade_date")
PRICE_NAMES = ("收盘价(元)", "收盘价", "收盘点位", "收盘", "指数点位", "close", "close_price")
YTM_NAMES = ("收盘价(%)", "到期收益率(%)", "到期收益率", "ytm", "收盘价(%)".replace("(", "（"))


def _norm(text: str) -> str:
    """列名归一化：去空白、全角括号转半角、转小写。"""
    s = str(text).strip().replace(" ", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    return s.lower()


def _match(columns, names):
    lowered = {_norm(c): c for c in columns}
    for n in names:
        key = _norm(n)
        if key in lowered:
            return lowered[key]
    # 宽松匹配：以“收盘价”开头（如 收盘价(元)、收盘价(净价)），
    # 但必须排除收益率类列（含 % 或“收益率”），避免把 YTM 当点位。
    allow_yield = any("%" in _norm(n) or "收益率" in _norm(n) for n in names)
    for c in columns:
        key = _norm(c)
        if not key.startswith("收盘价"):
            continue
        if not allow_yield and ("%" in key or "收益率" in key):
            continue
        return c
    return None


def _read_raw(uploaded_file) -> pd.DataFrame:
    raw = (
        uploaded_file.getvalue()
        if hasattr(uploaded_file, "getvalue")
        else uploaded_file.read()
    )
    filename = str(getattr(uploaded_file, "name", "")).lower()
    if filename.endswith(".csv"):
        for enc in ("utf-8-sig", "gb18030", "utf-8"):
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("CSV 编码无法识别，请另存为 UTF-8 或 xlsx 后重试。")
    return pd.read_excel(io.BytesIO(raw), sheet_name=0)


def load_ifind_prices(uploaded_file, price_names=PRICE_NAMES, return_meta: bool = False):
    """读取 iFinD 行情文件，返回按日期升序、去重后的收盘价 Series。

    return_meta=True 时返回 (series, meta)，meta 含识别到的列名与跳过列，
    便于页面展示“实际用了哪两列”。
    """
    if not isinstance(price_names, tuple):
        price_names = tuple(price_names)

    # iFinD 导出可能带标题行，尝试前 4 行作为表头（CSV 只按首行解析）
    is_csv = str(getattr(uploaded_file, "name", "")).lower().endswith(".csv")
    first_cols = None
    chosen = None
    for header_row in range(1 if is_csv else 4):
        try:
            if header_row == 0:
                candidate = _read_raw(uploaded_file)
            else:
                candidate = pd.read_excel(
                    io.BytesIO(uploaded_file.getvalue()), sheet_name=0, header=header_row
                )
        except Exception as exc:  # 读取失败直接抛出，交调用方展示
            raise ValueError(f"文件无法打开：{exc}") from exc
        candidate.columns = [str(c).strip() for c in candidate.columns]
        if first_cols is None:
            first_cols = list(candidate.columns)

        d, p = _match(candidate.columns, DATE_NAMES), _match(candidate.columns, price_names)
        if d is None or p is None or d == p:
            continue

        dates = pd.to_datetime(candidate[d], errors="coerce")
        prices = pd.to_numeric(
            candidate[p]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False),
            errors="coerce",
        )
        # 关键校验：表头漂移时会把数据行当列名，导致日期被解析成 1970 年等伪值。
        # 只有“日期合理 + 数值有效”的记录足够多，才认定该表头正确。
        ok = (
            dates.notna()
            & prices.notna()
            & (prices > 0)
            & dates.between(pd.Timestamp("1990-01-01"), pd.Timestamp("2100-12-31"))
        )
        if int(ok.sum()) >= 3:
            chosen = (dates, prices, d, p, list(candidate.columns))
            break

    if chosen is None:
        cols = "、".join(map(str, (first_cols or [])[:12]))
        raise ValueError(
            "未能识别出「日期列 + 收盘价列」的有效组合。文件首行列为：" + cols
            + f"。请确认文件包含「时间」和「{price_names[0]}」两列，且为日频行情（日期可被识别、价格为数字）。"
        )

    dates, prices, date_col, price_col, chosen_cols = chosen
    series = pd.Series(prices.to_numpy(dtype=float), index=dates, name="close")
    series = series.loc[series.index.notna() & series.notna()]
    series = series.loc[series > 0].sort_index()
    series = series[~series.index.duplicated(keep="last")]
    if len(series) < 3:
        raise ValueError(f"有效行情不足 3 个交易日（当前 {len(series)} 条），无法计算收益率。")

    if not return_meta:
        return series
    used = {date_col, price_col}
    skipped = [
        c for c in chosen_cols
        if c not in used and not re.fullmatch(r"unnamed.*", _norm(c))
    ]
    return series, {
        "date_col": date_col,
        "price_col": price_col,
        "rows": int(len(series)),
        "skipped": skipped,
    }


def align_and_calculate(stock_prices: pd.Series, bond_prices: pd.Series):
    """按共同交易日对齐两资产，返回 (价格表, 简单日收益率表)。"""
    prices = pd.concat(
        [stock_prices.rename("stock"), bond_prices.rename("bond")], axis=1, join="inner"
    ).dropna()
    if len(prices) < 3:
        raise ValueError("两种资产共同交易日不足 3 天，请检查两份文件的日期范围是否重叠。")
    returns = prices.pct_change().dropna()
    return prices, returns

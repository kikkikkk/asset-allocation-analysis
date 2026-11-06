# -*- coding: utf-8 -*-
"""公开数据源获取模块（云端部署版专用）

用途：在运行时从公开、免费的数据渠道获取两个指数的日度收盘数据，
使部署包无需包含任何真实行情文件，也不再分发任何受授权限制的数据。

数据来源与授权说明：
1. 沪深300（000300.SH）：腾讯财经公开行情接口（web.ifzq.gtimg.cn）。
2. 中债-7-10年国债指数 财富指数（CBA06501）：中国债券信息网
   （yield.chinabond.com.cn，中债估值中心官方公开网站）。

合规要点：
- 不从 iFinD 拉取、不存储、不公开再分发任何 iFinD 数据；
- 数据在用户会话中实时获取，仓库中没有任何行情文件；
- 已抽样核验：公开渠道数据与 iFinD 导出同源一致（中债官网 2709 个
  共同交易日价差为 0；沪深300 日收益率平均差异约 0，个别日存在极小差异）。

不做任何鉴权、不使用任何凭据。请求失败时抛出异常，由调用方明确提示，
绝不静默切换为模拟数据。
"""
import time

import pandas as pd
import requests

_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}

# 中债指数树上「中债-7-10年国债指数」的节点 ID（官网懒加载树的叶子节点），
# 同时保留按名称动态查找作为兜底，防止官网调整 ID。
_CB_NODE_ID = "8a8b2c8f5a492a01015a4ac986480043"
_CB_TREE_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/queryTree"
_CB_QUERY_URL = "https://yield.chinabond.com.cn/cbweb-mn/indices/singleIndexQueryResult"
_CB_INDEX_NAME = "中债-7-10年国债指数"

# 沪深300：腾讯接口单次最多返回约 640 根K线，按约2年一段循环取全区间
_HS300_CHUNKS = [
    ("2016-01-04", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2024-12-31"),
    ("2025-01-01", "2030-12-31"),
]


def fetch_hs300() -> pd.Series:
    """腾讯财经公开接口获取沪深300日收盘价（2016年至今，不含复权，指数无需复权）。"""
    frames = []
    for beg, end in _HS300_CHUNKS:
        r = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"sh000300,day,{beg},{end},800,"},
            headers=_UA,
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
        data = (j.get("data") or {}).get("sh000300") or {}
        kl = data.get("day") or data.get("qfqday") or []
        if not kl:
            raise ValueError(f"腾讯行情接口 {beg}~{end} 段未返回数据（code={j.get('code')} msg={j.get('msg')}）。")
        # kline 行格式: [date, open, close, high, low, volume, ...]，取收盘价
        frames.append(pd.DataFrame([(x[0], x[2]) for x in kl], columns=["date", "close"]))
        time.sleep(0.3)
    alld = pd.concat(frames).drop_duplicates("date")
    ser = pd.Series(
        alld["close"].astype(float).to_numpy(),
        index=pd.to_datetime(alld["date"]),
        name="沪深300",
    )
    return ser[ser.index.notna()].sort_index()


def fetch_bond_wealth() -> pd.Series:
    """中国债券信息网公开接口获取中债-7-10年国债指数财富指数日度数据（2007年至今）。"""
    s = requests.Session()
    s.headers.update({**_UA, "X-Requested-With": "XMLHttpRequest"})
    node_id = _resolve_node_id(s)

    r = s.post(
        _CB_QUERY_URL,
        params={
            "indexid": node_id,
            "qxlxt": "00",
            "ltcslx": "",
            "zslxt": "CFZS",  # CFZS = 财富指数（含票息再投资）
            "zslxt1": "",
            "lx": "1",
            "locale": "zh_CN",
        },
        timeout=40,
    )
    r.raise_for_status()
    j = r.json()
    if "emsg" in j:
        raise ValueError(f"中国债券信息网接口返回错误：{j['emsg']}")
    raw = j.get("CFZS_00") or {}
    if len(raw) < 10:
        raise ValueError(f"中国债券信息网返回数据过少（{len(raw)} 条）。")

    # 官网时间戳为北京时间零点的 UTC 表示（差 8 小时），+8h 后归一到交易日日期。
    idx = pd.to_datetime([int(t) for t in raw], unit="ms") + pd.Timedelta(hours=8)
    ser = pd.Series(
        pd.to_numeric(list(raw.values()), errors="coerce").astype(float),
        index=idx.normalize(),
        name="中债-7-10年国债指数",
    )
    return ser[ser.index.notna()].sort_index()


def _resolve_node_id(session) -> str:
    """先按硬编码 ID，失败则从官网指数树按名称动态查找。"""
    if _probe_node(session, _CB_NODE_ID):
        return _CB_NODE_ID
    try:
        tree = session.get(_CB_TREE_URL, params={"locale": "zh_CN"}, timeout=30).json()
        nodes = tree if isinstance(tree, list) else tree.get("data") or tree.get("rows") or []

        def walk(items):
            for n in items if isinstance(items, list) else []:
                if n.get("name") == _CB_INDEX_NAME and n.get("isParent") != "true":
                    return n.get("id")
                found = walk(n.get("children") or n.get("nodes") or [])
                if found:
                    return found
            return None

        nid = walk(nodes)
        if nid and _probe_node(session, nid):
            return nid
    except Exception:
        pass
    raise ValueError("未能从中国债券信息网定位「中债-7-10年国债指数」，接口可能已调整。")


def _probe_node(session, node_id: str) -> bool:
    """探测某个节点 ID 是否能返回财富指数数据。"""
    try:
        j = session.post(
            _CB_QUERY_URL,
            params={
                "indexid": node_id, "qxlxt": "00", "ltcslx": "",
                "zslxt": "CFZS", "zslxt1": "", "lx": "1", "locale": "zh_CN",
            },
            timeout=25,
        ).json()
        return "emsg" not in j and len(j.get("CFZS_00") or {}) >= 10
    except Exception:
        return False


def fetch_public_series() -> tuple[pd.Series, pd.Series]:
    """一次获取两个资产的公开数据，返回 (沪深300, 中债-7-10年国债财富指数)。"""
    return fetch_hs300(), fetch_bond_wealth()

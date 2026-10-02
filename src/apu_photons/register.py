"""Stage 3 後半：三角形不變量配對 + RANSAC，求 frame → 參考 的相似變換（平移、旋轉、等比縮放）。

相似變換本身就涵蓋中天翻轉（旋轉 180°）。錯配比配不上更危險，所以殘差或配對數不夠就判定失敗。
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .i18n import Msg

ALGO_VERSION = "tri-ransac-1"
TRI_STARS = 25
INV_TOL = 0.005
MATCH_RADIUS = 2.0
MIN_INLIERS = 8
MAX_RESIDUAL = 1.0
RANSAC_ITERS = 400
SCALE_TOL = 0.1  # 縮放倍率與預期差超過 10% 就是錯配（同一套器材應該是 1）


class RegistrationError(RuntimeError):
    pass


def _triangles(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """回傳（不變量 (T,2), 頂點索引 (T,3)）。頂點依「對邊由短到長」排序，讓兩邊的頂點能一一對應。"""
    n = min(len(xy), TRI_STARS)
    inv, verts = [], []
    for i, j, k in combinations(range(n), 3):
        p = xy[[i, j, k]]
        # 頂點 v 的對邊長度
        opp = np.array([np.linalg.norm(p[1] - p[2]), np.linalg.norm(p[0] - p[2]), np.linalg.norm(p[0] - p[1])])
        order = np.argsort(opp)
        a, b, c = opp[order]
        if c < 10 or a / c < 0.1:  # 太小或太扁的三角形不穩定
            continue
        inv.append((a / c, b / c))
        verts.append(np.array((i, j, k))[order])
    return np.asarray(inv, float).reshape(-1, 2), np.asarray(verts, int).reshape(-1, 3)


def similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """最小平方相似變換（Umeyama，不允許鏡像），回傳 3×3。"""
    ms, md = src.mean(0), dst.mean(0)
    s, d = src - ms, dst - md
    cov = d.T @ s / len(src)
    u, sv, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt)) or 1.0
    D = np.diag([1.0, sign])
    r = u @ D @ vt
    scale = (sv * np.diag(D)).sum() / (s ** 2).sum() * len(src)
    t = md - scale * r @ ms
    m = np.eye(3)
    m[:2, :2], m[:2, 2] = scale * r, t
    return m


def apply(m: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return xy @ m[:2, :2].T + m[:2, 2]


def scale_of(m: np.ndarray) -> float:
    return float(np.sqrt(abs(np.linalg.det(m[:2, :2]))))


def _plausible(m: np.ndarray, expected: float | None) -> bool:
    """縮放倍率合理。縮到接近 0 時所有星點會擠到同一顆星附近，殘差反而很小，必須另外擋掉。"""
    if not np.all(np.isfinite(m)):
        return False
    s = scale_of(m)
    if expected is None:
        return 0.2 < s < 5.0
    return abs(s / expected - 1.0) <= SCALE_TOL


def register(src_xy: np.ndarray, ref_xy: np.ndarray, rng: np.random.Generator | None = None,
             expected_scale: float | None = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """src_xy、ref_xy 依亮度由亮到暗排序。回傳（3×3 變換, 殘差 RMS 像素, 配對的 src 索引, 對應的 ref 索引）。

    expected_scale：預期的縮放倍率（同一套器材是 1；合併不同器材時是像素尺度的比例；None＝不知道，只擋明顯不合理的）。"""
    if len(src_xy) < 4 or len(ref_xy) < 4:
        raise RegistrationError(Msg("msg.reg.few_stars"))
    rng = rng or np.random.default_rng(0)
    s_inv, s_v = _triangles(src_xy)
    r_inv, r_v = _triangles(ref_xy)
    if not len(s_inv) or not len(r_inv):
        raise RegistrationError(Msg("msg.reg.no_triangles"))
    tree = cKDTree(r_inv)
    votes = np.zeros((min(len(src_xy), TRI_STARS), min(len(ref_xy), TRI_STARS)), int)
    for si, hits in enumerate(tree.query_ball_point(s_inv, INV_TOL)):
        for ri in hits:
            np.add.at(votes, (s_v[si], r_v[ri]), 1)
    pairs = [(i, int(votes[i].argmax())) for i in range(votes.shape[0]) if votes[i].max() >= 2]
    if len(pairs) < 3:
        raise RegistrationError(Msg("msg.reg.no_match"))
    pairs = np.asarray(pairs)
    a, b = src_xy[pairs[:, 0]], ref_xy[pairs[:, 1]]

    # RANSAC：兩點決定一個相似變換
    best, best_n = None, 0
    for _ in range(RANSAC_ITERS):
        idx = rng.choice(len(pairs), 2, replace=False)
        if np.linalg.norm(a[idx[0]] - a[idx[1]]) < 5:
            continue
        m = similarity(a[idx], b[idx])
        if not _plausible(m, expected_scale):
            continue
        n = int((np.linalg.norm(apply(m, a) - b, axis=1) < MATCH_RADIUS * 2).sum())
        if n > best_n:
            best, best_n = m, n
    if best is None or best_n < 3:
        raise RegistrationError(Msg("msg.reg.ransac"))

    # 用全部星點精修：最近鄰配對 → 重新擬合，重複幾次
    ref_tree = cKDTree(ref_xy)
    m = best
    for _ in range(3):
        dist, j = ref_tree.query(apply(m, src_xy))
        ok = dist < MATCH_RADIUS
        if ok.sum() < MIN_INLIERS:
            raise RegistrationError(Msg("msg.reg.few_matches", n=int(ok.sum())))
        m = similarity(src_xy[ok], ref_xy[j[ok]])
    if not _plausible(m, expected_scale):
        raise RegistrationError(Msg("msg.reg.scale", scale=scale_of(m) if np.all(np.isfinite(m)) else float("nan")))
    dist, j = ref_tree.query(apply(m, src_xy))
    ok = dist < MATCH_RADIUS
    rms = float(np.sqrt(np.mean(dist[ok] ** 2)))
    if rms > MAX_RESIDUAL:
        raise RegistrationError(Msg("msg.reg.residual", rms=rms))
    return m, rms, np.flatnonzero(ok), j[ok]

"""Stage 3 後半：三角形不變量配對 + RANSAC，求 frame → 參考 的相似變換（平移、旋轉、等比縮放）。

相似變換本身就涵蓋中天翻轉（旋轉 180°）。錯配比配不上更危險，所以殘差或配對數不夠就判定失敗。
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
from scipy.spatial import cKDTree

from .i18n import Msg
from .localnorm import _basis, eval_offset

ALGO_VERSION = "tri-ransac-1"
TRI_STARS = 25
INV_TOL = 0.005
MATCH_RADIUS = 2.0
MIN_INLIERS = 8
MAX_RESIDUAL = 1.0
RANSAC_ITERS = 400
SCALE_TOL = 0.1  # 縮放倍率與預期差超過 10% 就是錯配（同一套器材應該是 1）
# 畸變修正（選用）：相似變換之外再加一個低階多項式
DISTORT_MIN_STARS = 20   # 配對的星少於這個數不擬合（2 階要 6 項）
DISTORT_CUBIC_STARS = 40  # 夠多才用 3 階（10 項；鏡頭的徑向畸變是 3 階）
DISTORT_ITERS = 4
DISTORT_MAX = 0.05       # 修正量超過影像對角線的這個比例：擬合出問題了，不用
DISTORT_GRID = 24


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
        order = np.argsort(opp, kind="stable")  # 等腰時頂點順序固定
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


def overlap_fraction(m: np.ndarray, src_shape: tuple[int, int], ref_shape: tuple[int, int], n: int = 24) -> float:
    """frame 與參考影像重疊的面積，除以兩者中較小的面積（合併不同視野的器材時，小視野整張落在大視野內算 100%）。

    用 frame 上 n×n 個格點估計落在參考範圍內的比例。"""
    h, w = src_shape
    ys, xs = np.meshgrid(np.linspace(0, h - 1, n), np.linspace(0, w - 1, n), indexing="ij")
    p = apply(m, np.column_stack([xs.ravel(), ys.ravel()]))
    rh, rw = ref_shape
    inside = float(((p[:, 0] >= 0) & (p[:, 0] <= rw - 1) & (p[:, 1] >= 0) & (p[:, 1] <= rh - 1)).mean())
    src_area = h * w * scale_of(m) ** 2  # frame 在參考座標裡的面積
    return inside * src_area / max(min(src_area, rh * rw), 1e-9)


def register(src_xy: np.ndarray, ref_xy: np.ndarray, rng: np.random.Generator | None = None,
             expected_scale: float | None = 1.0,
             max_residual: float = MAX_RESIDUAL) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
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
    if rms > max_residual:
        raise RegistrationError(Msg("msg.reg.residual", rms=rms))
    return m, rms, np.flatnonzero(ok), j[ok]


def _poly(coeffs, xy: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """多項式修正量 (P, 2)。coeffs：[x 的係數, y 的係數]，座標依 shape 換成 −0.5～0.5（與 localnorm 同一套）。"""
    return np.column_stack([eval_offset(np, c, xy[:, 0], xy[:, 1], shape) for c in coeffs])


def _fit_poly(xy: np.ndarray, resid: np.ndarray, shape: tuple[int, int], degree: int) -> list[list[float]]:
    """resid (P, 2) 對 xy 的多項式擬合，剔除一次離群的配對（配錯的星）。"""
    a = np.column_stack(_basis(xy[:, 0], xy[:, 1], shape, degree))
    c, *_ = np.linalg.lstsq(a, resid, rcond=None)
    d = np.hypot(*(resid - a @ c).T)
    keep = d < max(3.0 * 1.4826 * float(np.median(d)), 0.3)
    if keep.sum() >= 2 * a.shape[1] and not keep.all():
        c, *_ = np.linalg.lstsq(a[keep], resid[keep], rcond=None)
    return [[float(v) for v in c[:, 0]], [float(v) for v in c[:, 1]]]


def fit_distortion(src_xy: np.ndarray, ref_xy: np.ndarray, m: np.ndarray, src_shape: tuple[int, int],
                   ref_shape: tuple[int, int]):
    """相似變換 m 之外的畸變修正：frame → 參考 = m·p + 多項式(p)。

    廣角鏡頭的畸變在每張自己的座標上是固定的；兩張之間位移很大（沒導星的漂移、拼接、不同器材合併）時，
    相似變換對得到中間、對不到角落。用配對到的星擬合 2～3 階多項式，每擬合一次就能多配到一些角落的星，重複幾次。

    回傳（畸變 dict, 殘差 RMS, 配對的 src 索引, 對應的 ref 索引）；星太少、沒有比較好、修正量不合理時回傳 None。
    畸變 dict：fwd（frame 座標 → 修正量，drizzle 用）、inv（參考座標 → frame 座標的修正量，整合的反向取樣用）、
    max（最大修正量，像素）、src_shape、ref_shape。"""
    tree = cKDTree(ref_xy)
    base = apply(m, src_xy)
    dist0, _j = tree.query(base)
    ok0 = dist0 < MATCH_RADIUS
    if ok0.sum() < DISTORT_MIN_STARS:
        return None
    rms0 = float(np.sqrt(np.mean(dist0[ok0] ** 2)))
    fwd = None
    for it in range(DISTORT_ITERS):
        pred = base if fwd is None else base + _poly(fwd, src_xy, src_shape)
        dist, j = tree.query(pred)
        ok = dist < MATCH_RADIUS
        if ok.sum() < DISTORT_MIN_STARS:
            return None
        degree = 3 if it > 0 and ok.sum() >= DISTORT_CUBIC_STARS else 2  # 第一次先用 2 階：星只在中間時 3 階外插不穩
        fwd = _fit_poly(src_xy[ok], ref_xy[j[ok]] - base[ok], src_shape, degree)
    dist, j = tree.query(base + _poly(fwd, src_xy, src_shape))
    ok = dist < MATCH_RADIUS
    rms = float(np.sqrt(np.mean(dist[ok] ** 2))) if ok.any() else float("inf")
    if ok.sum() < ok0.sum() or rms >= rms0:
        return None
    # 反向（參考 → frame）：在 frame 上取一片格點，用正向模型送到參考座標，擬合回來的修正量
    h, w = src_shape
    gy, gx = np.meshgrid(np.linspace(0, h - 1, DISTORT_GRID), np.linspace(0, w - 1, DISTORT_GRID), indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    corr = _poly(fwd, grid, src_shape)
    biggest = float(np.hypot(*corr.T).max())
    if not np.isfinite(biggest) or biggest > DISTORT_MAX * float(np.hypot(h, w)):
        return None
    on_ref = apply(m, grid) + corr
    inv = _fit_poly(on_ref, grid - apply(np.linalg.inv(m), on_ref), ref_shape, 3)
    back = apply(np.linalg.inv(m), on_ref) + _poly(inv, on_ref, ref_shape)
    if float(np.hypot(*(back - grid).T).max()) > 0.25:  # 反向對不回來：畸變太劇烈，3 階表示不了
        return None
    info = {"fwd": fwd, "inv": inv, "max": biggest, "src_shape": tuple(src_shape), "ref_shape": tuple(ref_shape)}
    return info, rms, np.flatnonzero(ok), j[ok]


def _register_task(task):
    """task：(src_xy, ref_xy, 預期倍率, 畸變修正用的 (frame 尺寸, 參考尺寸) 或 None)。
    回傳（(變換, 殘差, src 索引, ref 索引, 畸變 dict 或 None) 或 None, 失敗原因或 None）。"""
    src_xy, ref_xy, expected, shapes = task
    try:
        if shapes is None:
            return (*register(src_xy, ref_xy, expected_scale=expected), None), None
        # 畸變大時相似變換的殘差本來就大：先放寬，修正後再用原本的門檻把關
        m, rms, si, ri = register(src_xy, ref_xy, expected_scale=expected, max_residual=MATCH_RADIUS)
        fitted = fit_distortion(src_xy, ref_xy, m, *shapes)
        info = None
        if fitted is not None:
            info, rms, si, ri = fitted
        if rms > MAX_RESIDUAL:
            raise RegistrationError(Msg("msg.reg.residual", rms=rms))
        return (m, rms, si, ri, info), None
    except RegistrationError as exc:
        return None, exc.args[0] if exc.args else str(exc)


def register_many(tasks: list[tuple], workers: int, progress=None) -> list[tuple]:
    """多張一起對齊：tasks 見 `_register_task`，回傳 [(結果或 None, 失敗原因或 None)]。

    每張都用相同的隨機數種子，多行程與逐張的結果完全相同。張數多、每張又小（例如 Seestar 幾百張）時，
    對齊是單執行緒的 Python 迴圈，占整次時間的四分之一；分給多個行程。progress(done, total)。"""
    if workers <= 1 or len(tasks) < 2 * workers:
        results = map(_register_task, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=min(workers, len(tasks)))
        results = pool.map(_register_task, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
    out = []
    try:
        for r in results:
            out.append(r)
            if progress:
                progress(len(out), len(tasks))
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return out

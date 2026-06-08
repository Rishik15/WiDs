from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
pd.set_option("display.max_columns", 100)

HORIZONS = [12, 24, 48, 72]

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = ROOT_DIR / "SETTINGS.json"


def load_settings(settings_path: str | Path | None = None) -> dict:
    """
    Load SETTINGS.json from the project root or from a user-provided path.
    """
    if settings_path is None:
        settings_path = DEFAULT_SETTINGS_PATH

    settings_path = Path(settings_path)

    if not settings_path.is_absolute():
        settings_path = ROOT_DIR / settings_path

    if not settings_path.exists():
        raise FileNotFoundError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(settings: dict, key: str) -> Path:
    """
    Resolve a path from SETTINGS.json relative to the project root.
    """
    if key not in settings:
        raise KeyError(f"Missing required settings key: {key}")

    path = Path(settings[key])

    if path.is_absolute():
        return path

    return ROOT_DIR / path


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    eps = 1e-6

    dist = d["dist_min_ci_0_5h"].astype(float).clip(lower=1.0)
    dist_km = dist / 1000.0

    perims = d["num_perimeters_0_5h"].astype(float).clip(lower=0.0)
    dt_span = d["dt_first_last_0_5h"].astype(float).clip(lower=0.0)

    closing = d["closing_speed_m_per_h"].astype(float)
    closing_pos = closing.clip(lower=0.0)

    radial_rate = d["radial_growth_rate_m_per_h"].astype(float).clip(lower=0.0)
    along_track = (
        d["along_track_speed"].astype(float)
        if "along_track_speed" in d.columns
        else 0.0
    )

    align_abs = d["alignment_abs"].astype(float).clip(lower=0.0, upper=1.0)
    area = d["area_first_ha"].astype(float).clip(lower=0.0)
    growth_rate_ha_h = d["area_growth_rate_ha_per_h"].astype(float).clip(lower=0.0)

    d["dist_km"] = dist_km
    d["log_dist"] = np.log1p(dist)
    d["sqrt_dist"] = np.sqrt(dist)

    d["inv_dist"] = 1.0 / (dist + 1.0)
    d["inv_dist_sq"] = d["inv_dist"] ** 2
    d["inv_dist_km"] = 1.0 / (dist_km + 0.05)

    d["zone_lt3km"] = (dist < 3000).astype(int)
    d["zone_3to5km"] = ((dist >= 3000) & (dist < 5000)).astype(int)
    d["zone_5to10km"] = ((dist >= 5000) & (dist < 10000)).astype(int)
    d["zone_ge10km"] = (dist >= 10000).astype(int)

    if "dist_fit_r2_0_5h" in d.columns:
        r2 = d["dist_fit_r2_0_5h"].astype(float).clip(lower=0.0, upper=1.0)
        d["dist_trend_reliable"] = (r2 > 0.6).astype(int)
        d["dist_r2"] = r2
    else:
        d["dist_trend_reliable"] = 0
        d["dist_r2"] = 0.0

    for col in [
        "dist_std_ci_0_5h",
        "dist_change_ci_0_5h",
        "dist_slope_ci_0_5h",
        "dist_accel_m_per_h2",
    ]:
        if col in d.columns:
            d[col] = d[col].astype(float)

    if "dist_change_ci_0_5h" in d.columns:
        d["dist_change_km"] = d["dist_change_ci_0_5h"] / 1000.0
        d["dist_change_norm"] = d["dist_change_ci_0_5h"] / (dist + 1.0)

    if "dist_slope_ci_0_5h" in d.columns:
        d["dist_slope_norm"] = d["dist_slope_ci_0_5h"] / (dist + 1.0)

    if "dist_std_ci_0_5h" in d.columns:
        d["dist_std_norm"] = d["dist_std_ci_0_5h"] / (dist + 1.0)

    effective_closing = closing_pos + radial_rate
    d["effective_closing_speed"] = effective_closing

    d["eta_closing_h"] = np.where(
        closing_pos > 0.1,
        dist / (closing_pos + eps),
        9999.0,
    )

    d["eta_effective_h"] = np.where(
        effective_closing > 0.1,
        dist / (effective_closing + eps),
        9999.0,
    )

    d["log_eta_effective"] = np.log1p(np.clip(d["eta_effective_h"], 0, 9999))
    d["log_eta_closing"] = np.log1p(np.clip(d["eta_closing_h"], 0, 9999))

    for h in HORIZONS:
        d[f"eta_within_{h}h"] = (d["eta_effective_h"] <= float(h)).astype(int)
        d[f"eta_margin_{h}h"] = d["eta_effective_h"] - float(h)

    if isinstance(along_track, pd.Series):
        along_pos = along_track.astype(float).clip(lower=0.0)

        d["eta_alongtrack_h"] = np.where(
            along_pos > 0.1,
            dist / (along_pos + eps),
            9999.0,
        )

        d["log_eta_alongtrack"] = np.log1p(np.clip(d["eta_alongtrack_h"], 0, 9999))
    else:
        d["eta_alongtrack_h"] = 9999.0
        d["log_eta_alongtrack"] = np.log1p(9999.0)

    fire_radius_m = np.sqrt((area * 10000.0) / np.pi)
    d["fire_radius_m"] = fire_radius_m
    d["radius_to_dist"] = fire_radius_m / (dist + 1.0)

    d["dist_minus_radius"] = np.clip(dist - fire_radius_m, 1.0, None)
    d["log_dist_minus_radius"] = np.log1p(d["dist_minus_radius"])

    d["area_to_dist_km"] = area / (dist_km + 0.1)
    d["growth_to_dist_km"] = growth_rate_ha_h / (dist_km + 0.1)

    if "radial_growth_m" in d.columns:
        d["radial_growth_m"] = d["radial_growth_m"].astype(float).clip(lower=0.0)
        d["radial_to_dist"] = d["radial_growth_m"] / (dist + 1.0)

    d["threat_pressure"] = align_abs * effective_closing / (np.log1p(dist) + eps)
    d["alignment_x_closing"] = align_abs * closing_pos
    d["alignment_x_effective"] = align_abs * effective_closing

    if "cross_track_component" in d.columns:
        ctc = d["cross_track_component"].astype(float)
        d["cross_track_abs"] = np.abs(ctc)
        d["cross_track_norm"] = np.abs(ctc) / (dist + 1.0)

    if "spread_bearing_sin" in d.columns and "spread_bearing_cos" in d.columns:
        sb_sin = d["spread_bearing_sin"].astype(float)
        sb_cos = d["spread_bearing_cos"].astype(float)
        d["bearing_strength"] = np.sqrt(sb_sin**2 + sb_cos**2)

    d["has_movement"] = (perims > 1).astype(int)
    d["perim_density"] = perims / (dt_span + 0.25)
    d["short_window"] = (dt_span < 0.5).astype(int)

    if "low_temporal_resolution_0_5h" in d.columns:
        d["low_temporal_resolution_0_5h"] = d["low_temporal_resolution_0_5h"].astype(
            int
        )

    hour = (
        d["event_start_hour"].astype(float) if "event_start_hour" in d.columns else 0.0
    )
    d["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    d["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    dow = (
        d["event_start_dayofweek"].astype(float)
        if "event_start_dayofweek" in d.columns
        else 0.0
    )
    d["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    d["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    month = (
        d["event_start_month"].astype(float)
        if "event_start_month" in d.columns
        else 1.0
    )
    d["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    d["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    d = d.replace([np.inf, -np.inf], np.nan).fillna(0)

    return d


def km_censor_survival(times: np.ndarray, censor_event: np.ndarray):
    order = np.argsort(times)
    times = times[order]
    censor_event = censor_event[order]

    uniq = np.unique(times)
    n = len(times)
    at_risk = n
    G = 1.0
    G_map = {}

    for tt in uniq:
        m = np.sum(times == tt)
        d = np.sum((times == tt) & (censor_event == 1))

        if at_risk > 0:
            G *= 1.0 - d / at_risk

        G_map[tt] = max(G, 1e-6)
        at_risk -= m

    uniq_times = np.array(sorted(G_map.keys()))
    G_vals = np.array([G_map[u] for u in uniq_times])

    return uniq_times, G_vals


def G_of_t(t_query, uniq_times, G_vals):
    t_query = np.asarray(t_query)
    idx = np.searchsorted(uniq_times, t_query, side="right") - 1

    out = np.ones_like(t_query, dtype=float)
    ok = idx >= 0
    out[ok] = G_vals[idx[ok]]

    return np.clip(out, 1e-6, 1.0)


def enforce_monotone(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0.0, 1.0)
    P[:, 1] = np.maximum(P[:, 1], P[:, 0])
    P[:, 2] = np.maximum(P[:, 2], P[:, 1])
    P[:, 3] = np.maximum(P[:, 3], P[:, 2])
    return np.clip(P, 0.0, 1.0)


def make_ipcw_data(X, t, e, H, uniq_times, G_vals, weight_cap=30.0):
    y = ((e == 1) & (t <= H)).astype(int)
    mask = ((e == 1) & (t <= H)) | (t > H)

    t_clip = np.minimum(t, H)
    G_t = G_of_t(t_clip, uniq_times, G_vals)

    w = 1.0 / G_t
    w = np.clip(w, 1.0, weight_cap)

    return X[mask], y[mask], w[mask], mask


def make_tail_data(X, t, e, H0, H1, uniq_times, G_vals, weight_cap=50.0):
    y = ((e == 1) & (t > H0) & (t <= H1)).astype(int)
    mask = ((e == 1) & (t > H0) & (t <= H1)) | (t > H1)

    t_clip = np.minimum(t, H1)
    G_t = G_of_t(t_clip, uniq_times, G_vals)

    w = 1.0 / G_t
    w = np.clip(w, 1.0, weight_cap)

    return X[mask], y[mask], w[mask], mask


def bagged_lgb_predict(X_tr, y_tr, w_tr, X_va, n_seeds=5):
    preds = np.zeros(len(X_va))

    for seed in range(42, 42 + n_seeds):
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=3,
            min_child_samples=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=8.0,
            objective="binary",
            random_state=seed,
            verbose=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds += model.predict_proba(X_va)[:, 1]

    return preds / n_seeds


def bagged_xgb_predict(X_tr, y_tr, w_tr, X_va, n_seeds=5):
    preds = np.zeros(len(X_va))

    for seed in range(42, 42 + n_seeds):
        model = xgb.XGBClassifier(
            n_estimators=2500,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=1,
            subsample=0.9,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=1.0,
            gamma=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds += model.predict_proba(X_va)[:, 1]

    return preds / n_seeds


def bagged_cb_predict(X_tr, y_tr, w_tr, X_va, n_seeds=5):
    preds = np.zeros(len(X_va))

    for seed in range(42, 42 + n_seeds):
        model = CatBoostClassifier(
            iterations=4000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=8.0,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=seed,
            verbose=False,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds += model.predict_proba(X_va)[:, 1]

    return preds / n_seeds


def run_pipeline_lgb_submission(X_tr, t_tr, e_tr, X_test):
    c_tr = (e_tr == 0).astype(int)
    uniq_times, G_vals = km_censor_survival(t_tr, c_tr)

    P = np.zeros((len(X_test), 4))

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 12, uniq_times, G_vals)
    p12 = bagged_lgb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=10)

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 24, uniq_times, G_vals)
    p24 = bagged_lgb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=10)

    X48_tr, y48_tr, w48_tr, _ = make_tail_data(
        X_tr, t_tr, e_tr, 24, 48, uniq_times, G_vals
    )

    tail_feats = [
        "dist_km",
        "eta_effective_h",
        "effective_closing_speed",
        "alignment_abs",
        "has_movement",
    ]

    lr_48 = LogisticRegression(C=0.2, max_iter=2000)
    lr_48.fit(X48_tr[tail_feats], y48_tr, sample_weight=w48_tr)

    p48_tail = lr_48.predict_proba(X_test[tail_feats])[:, 1]
    p48 = p24 + (1 - p24) * p48_tail

    late_events = np.sum((e_tr == 1) & (t_tr > 48))
    survive_48 = np.sum(t_tr > 48)
    alpha = (late_events + 1) / (survive_48 + 2)
    p72 = p48 + (1 - p48) * alpha

    P[:, 0] = p12
    P[:, 1] = p24
    P[:, 2] = p48
    P[:, 3] = p72

    return enforce_monotone(P)


def run_pipeline_xgb_submission(X_tr, t_tr, e_tr, X_test):
    c_tr = (e_tr == 0).astype(int)
    uniq_times, G_vals = km_censor_survival(t_tr, c_tr)

    P = np.zeros((len(X_test), 4))

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 12, uniq_times, G_vals)
    p12 = bagged_xgb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=8)

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 24, uniq_times, G_vals)
    p24 = bagged_xgb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=8)

    X48_tr, y48_tr, w48_tr, _ = make_tail_data(
        X_tr, t_tr, e_tr, 24, 48, uniq_times, G_vals
    )

    tail_feats = [
        "dist_km",
        "eta_effective_h",
        "effective_closing_speed",
        "alignment_abs",
        "has_movement",
    ]

    p48_tail = bagged_xgb_predict(
        X48_tr[tail_feats],
        y48_tr,
        w48_tr,
        X_test[tail_feats],
        n_seeds=10,
    )

    p48 = p24 + (1 - p24) * p48_tail

    late_events = np.sum((e_tr == 1) & (t_tr > 48))
    survive_48 = np.sum(t_tr > 48)
    alpha = (late_events + 1) / (survive_48 + 2)
    p72 = p48 + (1 - p48) * alpha

    P[:, 0] = p12
    P[:, 1] = p24
    P[:, 2] = p48
    P[:, 3] = p72

    return enforce_monotone(P)


def run_pipeline_cat_submission(X_tr, t_tr, e_tr, X_test):
    c_tr = (e_tr == 0).astype(int)
    uniq_times, G_vals = km_censor_survival(t_tr, c_tr)

    P = np.zeros((len(X_test), 4))

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 12, uniq_times, G_vals)
    p12 = bagged_cb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=5)

    Xh_tr, yh_tr, wh_tr, _ = make_ipcw_data(X_tr, t_tr, e_tr, 24, uniq_times, G_vals)
    p24 = bagged_cb_predict(Xh_tr, yh_tr, wh_tr, X_test, n_seeds=5)

    X48_tr, y48_tr, w48_tr, _ = make_tail_data(
        X_tr, t_tr, e_tr, 24, 48, uniq_times, G_vals
    )

    tail_feats = [
        "dist_km",
        "eta_effective_h",
        "effective_closing_speed",
        "alignment_abs",
        "has_movement",
    ]

    p48_tail = bagged_cb_predict(
        X48_tr[tail_feats],
        y48_tr,
        w48_tr,
        X_test[tail_feats],
        n_seeds=5,
    )

    p48 = p24 + (1 - p24) * p48_tail

    late_events = np.sum((e_tr == 1) & (t_tr > 48))
    survive_48 = np.sum(t_tr > 48)
    alpha = (late_events + 1) / (survive_48 + 2)
    p72 = p48 + (1 - p48) * alpha

    P[:, 0] = p12
    P[:, 1] = p24
    P[:, 2] = p48
    P[:, 3] = p72

    return enforce_monotone(P)


def fit_xgb_aft_submission(X_tr, t_tr, e_tr, seed=42):
    lower = np.log1p(t_tr.astype(float))
    upper = np.log1p(t_tr.astype(float))
    upper[e_tr == 0] = np.inf

    dtrain = xgb.DMatrix(X_tr)
    dtrain.set_float_info("label_lower_bound", lower)
    dtrain.set_float_info("label_upper_bound", upper)

    params = {
        "objective": "survival:aft",
        "aft_loss_distribution": "extreme",
        "aft_loss_distribution_scale": 1.5,
        "eta": 0.02,
        "max_depth": 8,
        "min_child_weight": 5,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "lambda": 3.0,
        "alpha": 0.5,
        "tree_method": "hist",
        "seed": seed,
    }

    return xgb.train(params, dtrain, num_boost_round=3000, verbose_eval=False)


def aft_probs_submission(model, X_tr, t_tr, e_tr, X_test):
    c_tr = (e_tr == 0).astype(int)
    uniq_times, G_vals = km_censor_survival(t_tr, c_tr)

    risk_tr = -model.predict(xgb.DMatrix(X_tr))
    risk_test = -model.predict(xgb.DMatrix(X_test))

    P = np.zeros((len(X_test), 4))

    for j, H in enumerate(HORIZONS):
        Xh_tr, yh_tr, wh_tr, mask_tr = make_ipcw_data(
            X_tr, t_tr, e_tr, H, uniq_times, G_vals
        )

        r_tr = risk_tr[mask_tr]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(r_tr, yh_tr, sample_weight=wh_tr)

        P[:, j] = iso.predict(risk_test)

    return enforce_monotone(P)


def fit_xgb_cox_submission(X_tr, t_tr, e_tr, seed=42):
    y = t_tr.astype(float).copy()
    y[e_tr == 0] *= -1.0

    dtrain = xgb.DMatrix(X_tr, label=y)

    params = {
        "objective": "survival:cox",
        "eta": 0.02,
        "max_depth": 8,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "min_child_weight": 2,
        "lambda": 1.0,
        "alpha": 0.0,
        "seed": seed,
        "tree_method": "hist",
    }

    return xgb.train(params, dtrain, num_boost_round=3000, verbose_eval=False)


def cox_probs_submission(model, X_tr, t_tr, e_tr, X_test):
    c_tr = (e_tr == 0).astype(int)
    uniq_times, G_vals = km_censor_survival(t_tr, c_tr)

    risk_tr = model.predict(xgb.DMatrix(X_tr))
    risk_test = model.predict(xgb.DMatrix(X_test))

    P = np.zeros((len(X_test), 4))

    for j, H in enumerate(HORIZONS):
        Xh_tr, yh_tr, wh_tr, mask_tr = make_ipcw_data(
            X_tr, t_tr, e_tr, H, uniq_times, G_vals
        )

        r_tr = risk_tr[mask_tr]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(r_tr, yh_tr, sample_weight=wh_tr)

        P[:, j] = iso.predict(risk_test)

    return enforce_monotone(P)


def blend_per_horizon(models, W):
    n = models[0].shape[0]
    P = np.zeros((n, 4))

    for j in range(4):
        for m_idx, Pi in enumerate(models):
            P[:, j] += W[j, m_idx] * Pi[:, j]

    return enforce_monotone(P)


def sharpen_p12(P, best_p):
    P = P.copy()
    P[:, 0] = np.power(P[:, 0], best_p)
    return enforce_monotone(P)


def generate_submission(
    settings_path: str | Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    settings = load_settings(settings_path)

    if output_path is None:
        output_path = resolve_path(settings, "SUBMISSION_PATH")

    print("Loading data...")
    train = pd.read_csv(resolve_path(settings, "TRAIN_PATH"))
    val = pd.read_csv(resolve_path(settings, "VALIDATION_PATH"))
    test = pd.read_csv(resolve_path(settings, "TEST_PATH"))

    print("Loading blend settings...")
    bundle = joblib.load(resolve_path(settings, "STACK_BUNDLE_PATH"))
    W = bundle["W"]
    w_risk = bundle["w_risk"]
    best_p = bundle["best_p"]

    print("Building features...")
    all_train = pd.concat([train, val], ignore_index=True)

    train_fe = build_features(all_train)
    test_fe = build_features(test)

    drop_cols = ["event_id", "event", "time_to_hit_hours"]

    X_tr = train_fe.drop(columns=drop_cols)
    t_tr = train_fe["time_to_hit_hours"].values.astype(float)
    e_tr = train_fe["event"].values.astype(int)

    X_test = test_fe.drop(columns=["event_id"])
    test_ids = test_fe["event_id"].values

    print(f"Training feature matrix: {X_tr.shape}")
    print(f"Test feature matrix: {X_test.shape}")

    print("Training LightGBM component...")
    P_lgb_test = run_pipeline_lgb_submission(X_tr, t_tr, e_tr, X_test)

    print("Training XGBoost component...")
    P_xgb_test = run_pipeline_xgb_submission(X_tr, t_tr, e_tr, X_test)

    print("Training CatBoost component...")
    P_cat_test = run_pipeline_cat_submission(X_tr, t_tr, e_tr, X_test)

    print("Training XGBoost AFT survival component...")
    aft_model = fit_xgb_aft_submission(X_tr, t_tr, e_tr)
    P_aft_test = aft_probs_submission(aft_model, X_tr, t_tr, e_tr, X_test)

    print("Training XGBoost Cox survival component...")
    cox_model = fit_xgb_cox_submission(X_tr, t_tr, e_tr)
    P_cox_test = cox_probs_submission(cox_model, X_tr, t_tr, e_tr, X_test)

    print("Blending predictions...")
    P_risk_mix = w_risk * P_cox_test + (1 - w_risk) * P_aft_test
    P_risk_mix = enforce_monotone(P_risk_mix)

    models_test = [
        P_lgb_test,
        P_xgb_test,
        P_cat_test,
        P_risk_mix,
    ]

    P_blend = blend_per_horizon(models_test, W)
    P_blend = sharpen_p12(P_blend, best_p)
    P_final = enforce_monotone(P_blend)

    sub = pd.DataFrame(
        {
            "event_id": test_ids.astype(int),
            "prob_12h": P_final[:, 0],
            "prob_24h": P_final[:, 1],
            "prob_48h": P_final[:, 2],
            "prob_72h": P_final[:, 3],
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(output_path, index=False)

    print(f"Submission saved to: {output_path}")
    print(sub.head())

    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the final WiDS submission file."
    )
    parser.add_argument(
        "--settings",
        default="SETTINGS.json",
        help="Path to SETTINGS.json. Defaults to SETTINGS.json in the project root.",
    )
    args = parser.parse_args()

    start_time = perf_counter()

    generate_submission(settings_path=args.settings)

    elapsed = perf_counter() - start_time
    minutes = int(elapsed // 60)
    seconds = elapsed % 60

    print("\nRuntime")
    print("-" * 40)
    print(f"Total script runtime: {elapsed:.2f} seconds")
    print(f"Total script runtime: {minutes} min {seconds:.2f} sec")

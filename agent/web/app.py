"""Flask web 界面 — M4 自動微調 Agent 的分析 / 訓練 / 監控儀表板.

啟動:
    python -m agent.web.app                       # http://127.0.0.1:5000
    python -m agent.web.app --host 0.0.0.0 --port 8080

功能:
  - 分析 (dry-run): DatasetProfile + encoder 建議 + Recipe + 訓練指令
  - 單一 trial 背景訓練 (/run) + 即時進度 (/progress)
  - 完整流程 (/run_full): 多 encoder × 改良迴圈 × 多 fold (LoopController);
    可填「引導方向 (guidance)」納入 LLM 決策
  - 整體進度儀表板 (/run_status): 結果 histogram、每個 trial 的完整 Recipe 參數、
    LLM 決策理由、逐 epoch 曲線 (train/val loss + score)、confusion matrix、report.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import threading
import time
import traceback
from typing import get_args

from flask import Flask, jsonify, redirect, render_template, request

from . import optdocs as app_optdocs
from . import settings as app_settings
from .. import (auto_finetune, conversation as convo, dataset_analyzer,
                dataset_ingest, dataset_registry as dsreg,
                encoder_registry as reg, log_curves, presets as presets_mod)
from ..advisor import HeuristicAdvisor
from ..config import AgentConfig
from ..ledger import Ledger
from ..privacy import egress as egress_mod
from ..privacy import load_user_facts, save_user_facts
from ..privacy.facts import Anatomy, Modality, UserAnswer
from ..run import single_trial
from ..schemas import EvalConfig
from ..trainer import gpu_allocatable

_MODALITIES = list(get_args(Modality))
_ANATOMIES = list(get_args(Anatomy))

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RUNS = os.path.join(_ROOT, "runs")
_DATA = os.path.join(_ROOT, "data")
_DEFAULT_DATA = os.path.join(_ROOT, "data", "5_fold_PAPILA", "PAPILA_seed42_fold0")

app = Flask(__name__)
# 上傳資料集: 單次請求上限 (壓縮檔)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 ** 3
# 反向代理相容: 關掉 Flask 對「結尾斜線不符」自動發的 301 (其 Location 為絕對路徑,
# 在代理下會壞掉且被瀏覽器永久快取)。
app.url_map.strict_slashes = False


@app.after_request
def _no_cache(resp):
    """動態儀表板 / API 一律不快取, 避免瀏覽器快取舊回應或 301 redirect。"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


# 啟動時把設定頁存的 API key 灌進 os.environ, 讓 advisor=llm/skill 直接可用
app_settings.apply_env()

# 背景訓練工作 (in-memory)
JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def _list_runs() -> list[dict]:
    out = []
    if not os.path.isdir(_RUNS):
        return out
    for name in sorted(os.listdir(_RUNS), reverse=True):
        d = os.path.join(_RUNS, name)
        led = os.path.join(d, "ledger.jsonl")
        if os.path.isfile(led):
            try:
                trials = Ledger(d).history()
            except Exception:
                trials = []
            out.append({"name": name, "n_trials": len(trials),
                        "trials": [t.model_dump() for t in trials]})
    return out


def _run_names() -> list[str]:
    """所有 run 目錄名 (有 logs/ 或 ledger.jsonl)，最新在前。"""
    if not os.path.isdir(_RUNS):
        return []
    names = []
    for name in os.listdir(_RUNS):
        d = os.path.join(_RUNS, name)
        if not os.path.isdir(d) or name.startswith("_"):
            continue
        if os.path.isdir(os.path.join(d, "logs")) or \
                os.path.isfile(os.path.join(d, "ledger.jsonl")) or \
                os.path.isfile(os.path.join(d, "conversation.jsonl")):
            names.append(name)
    return sorted(names, reverse=True)


def _dataset_ctx() -> dict:
    """資料集選擇器要用的 context: 分組清單 + 預設選取路徑。"""
    groups = dsreg.groups(_DATA)
    flat = [d for g in groups for d in g["datasets"]]
    default = os.path.abspath(_DEFAULT_DATA)
    if not any(d["path"] == default for d in flat):
        default = flat[0]["path"] if flat else _DEFAULT_DATA
    return {"dataset_groups": groups, "default_data": default}


def _latest_log(run_dir: str) -> str | None:
    logs = glob.glob(os.path.join(run_dir, "logs", "*.txt"))
    return max(logs, key=os.path.getmtime) if logs else None


# val 指標區塊 / 最終評估切割 — 與 agent/log_curves.py 共用同一份實作,
# 避免兩邊 regex 各自演化 (曾因此讓 web 曲線與 advisor 看到的資料不一致)。
_METRIC_KEYS = log_curves.METRIC_KEYS
_METRIC_RE = log_curves.METRIC_RE
_split_final_eval = log_curves.split_final_eval


def _parse_progress(log_path: str) -> dict:
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    # body = 只有逐 epoch 訓練的部分 (指標/loss 一律從這裡取, 否則跑完後
    # 「最新 val 指標」會變成最終 test 的數字)
    body, finals = _split_final_eval(text)

    total = int(m.group(1)) if (m := re.search(r"epochs=(\d+),", text)) else None
    epochs_seen = [int(x) for x in re.findall(r"Epoch:\s*\[(\d+)\]", body)]
    cur_epoch = max(epochs_seen) if epochs_seen else None

    val = None
    if mm := list(_METRIC_RE.finditer(body)):
        vals = [float(x) for x in mm[-1].groups()]
        val = dict(zip(_METRIC_KEYS, vals))
    val_loss = float(m.group(1)) if (m := re.search(
        r"val loss:\s*([\d.]+)(?!.*val loss:)", body, re.S)) else None
    train_loss = None
    if tl := re.findall(r"Averaged stats:.*?loss:\s*[\d.]+\s*\(([\d.]+)\)", body):
        train_loss = float(tl[-1])
    # 最終 test (用 val 最佳 checkpoint 重跑) — 與逐 epoch val 分開回報
    final_test = next((f for f in reversed(finals) if f["mode"] == "test"), None)

    best_epoch = best_score = None
    if bm := re.findall(r"Best epoch =\s*(\d+),\s*Best score =\s*([\d.]+)", text):
        best_epoch, best_score = int(bm[-1][0]), float(bm[-1][1])

    finished = "Training time" in text
    tt = re.search(r"Training time\s*([\d:]+)", text)
    crashed = (not finished) and bool(
        re.search(r"Traceback \(most recent call last\)|Error:", text))
    tail = "\n".join(text.splitlines()[-24:])

    if finished:
        percent = 100.0
    elif total and cur_epoch is not None:
        percent = round(min(cur_epoch + 1, total) / total * 100, 1)
    else:
        percent = 0.0

    if finished:
        status = "finished"
    elif crashed:
        status = "error"
    else:
        status = "running"

    return {
        "total_epochs": total, "cur_epoch": cur_epoch,
        "train_loss": train_loss, "val_loss": val_loss, "val": val,
        "final_test": final_test,
        "best_epoch": best_epoch, "best_score": best_score,
        "finished": finished, "crashed": crashed, "status": status,
        "train_time": tt.group(1) if tt else None,
        "percent": percent, "tail": tail,
        "mtime": os.path.getmtime(log_path),
    }


def _parse_curves(log_path: str) -> dict:
    """從單一 trial 的 stdout log 抽出逐 epoch 曲線: train_loss / val_loss / val score / val 各指標。"""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return {}
    # 只取逐 epoch 的訓練部分; 最終 test 另外回報 (見 _split_final_eval)
    text, finals = _split_final_eval(text)
    train_loss = [float(x) for x in re.findall(
        r"Averaged stats:.*?loss:\s*[\d.]+\s*\(([\d.]+)\)", text)]
    val_loss = [float(x) for x in re.findall(r"val loss:\s*([\d.]+)", text)]
    val_score, val_acc, val_auc, val_f1 = [], [], [], []
    for mm in _METRIC_RE.finditer(text):
        g = [float(x) for x in mm.groups()]
        d = dict(zip(_METRIC_KEYS, g))
        val_score.append(d["score"]); val_acc.append(d["accuracy"])
        val_auc.append(d["roc_auc"]); val_f1.append(d["f1"])
    n = max(len(train_loss), len(val_score), len(val_loss))
    return {
        "epochs": list(range(n)),
        "train_loss": train_loss, "val_loss": val_loss,
        "val_score": val_score, "val_accuracy": val_acc,
        "val_roc_auc": val_auc, "val_f1": val_f1,
        "final_test": next((f for f in reversed(finals) if f["mode"] == "test"), None),
    }


@app.route("/progress")
def progress():
    """解析 run 的訓練 log → 即時進度 + 最新 val 指標。GET ?run=<name>。
    省略 run 則取最新一個。"""
    run = request.args.get("run") or (_run_names()[0] if _run_names() else None)
    if not run:
        return jsonify({"ok": False, "error": "尚無任何 run"}), 404
    run_dir = os.path.join(_RUNS, run)
    if not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": f"找不到 run: {run}"}), 404
    log_path = _latest_log(run_dir)
    if not log_path:
        return jsonify({"ok": True, "run": run, "log_file": None,
                        "status": "no-log", "percent": 0.0,
                        "note": "此 run 尚無訓練 log (可能剛排入或未開始)"})
    data = _parse_progress(log_path)
    data.update(ok=True, run=run, log_file=os.path.basename(log_path))
    return jsonify(data)


@app.route("/")
def index():
    """工作台: 左=決策討論, 右=notebook (即時實驗檢視)。"""
    return render_template(
        "workspace.html",
        nav="workspace", route_seg="",
        presets=presets_mod.names(),
        advisors=["llm", "heuristic", "skill"],   # llm 為預設 (下拉第一個)
        run_names=_run_names(),
        modalities=_MODALITIES, anatomies=_ANATOMIES,
        settings=app_settings.load(),   # 表單預設值取自整體設定
        primary_metrics=app_settings.PRIMARY_METRICS,
        option_docs=app_optdocs.load(),  # 表單各選項的 markdown 說明 (右側面板)
        **_dataset_ctx(),
    )


@app.route("/experiments")
def experiments():
    """新實驗頁: 分析 (dry-run) + 單一 trial 訓練 + 即時進度。"""
    return render_template(
        "experiments.html",
        nav="experiments", route_seg="experiments",
        encoders=[c.model_dump() for c in reg.all_cards(include_unavailable=True)],
        available_keys=[c.model_key for c in reg.available_cards()],
        presets=presets_mod.names(),
        run_names=_run_names(),
        primary_metrics=app_settings.PRIMARY_METRICS,
        **_dataset_ctx(),
    )


@app.route("/datasets-page")
def datasets_page():
    """資料集頁: 上傳新資料集 + 瀏覽 data/ 底下所有可用資料集。"""
    return render_template(
        "datasets.html", nav="datasets", route_seg="datasets-page",
        data_root=_DATA, **_dataset_ctx(),
    )


@app.route("/datasets")
def datasets_api():
    """data/ 底下所有 ImageFolder 資料集 (JSON)。?refresh=1 強制重掃。"""
    refresh = bool(request.args.get("refresh"))
    return jsonify({"ok": True, "data_root": _DATA,
                    "groups": dsreg.groups(_DATA, refresh=refresh)})


def _under_data(path: str) -> str | None:
    """把使用者給的路徑正規化到 data/ 底下; 越界回 None。"""
    if not path:
        return None
    p = os.path.realpath(path if os.path.isabs(path) else os.path.join(_DATA, path))
    root = os.path.realpath(_DATA)
    return p if p.startswith(root + os.sep) else None


@app.route("/datasets/validate", methods=["POST"])
def datasets_validate():
    """檢查任一路徑的資料集格式 MedClaw 是否吃得下 (不限 data/ 底下)。"""
    path = (request.form.get("path") or "").strip()
    if not path:
        return jsonify({"ok": False, "error": "缺少 path"}), 400
    if not os.path.isabs(path):
        path = os.path.join(_DATA, path)
    try:
        return jsonify(dsreg.validate(path))
    except Exception as e:                       # noqa: BLE001
        return jsonify({"ok": False, "errors": [f"檢查失敗: {e}"],
                        "warnings": [], "summary": None}), 400


@app.route("/datasets/upload", methods=["POST"])
def datasets_upload():
    """上傳壓縮檔 → 解壓到 data/<name>/ → 檢查格式。回傳檢查報告 (JSON)。"""
    import tempfile

    f = request.files.get("archive")
    if f is None or not f.filename:
        return jsonify({"ok": False, "errors": ["沒有收到檔案"], "warnings": []}), 400
    if dataset_ingest.archive_ext(f.filename) is None:
        return jsonify({"ok": False, "warnings": [], "errors": [
            f"不支援的檔案格式：{f.filename}（請上傳 .zip / .tar / .tar.gz）"]}), 400

    name = (request.form.get("name") or "").strip() or \
        dataset_ingest.default_name(f.filename)
    overwrite = bool(request.form.get("overwrite"))

    # 暫存在 data/ 底下 (與最終目的地同一個檔案系統); 底線開頭 → 不會被掃描器列出
    tmp_dir = tempfile.mkdtemp(prefix="_upload_", dir=_DATA)
    tmp_path = os.path.join(tmp_dir, "archive" + dataset_ingest.archive_ext(f.filename))
    try:
        f.save(tmp_path)
        report = dataset_ingest.ingest(tmp_path, _DATA, name, overwrite=overwrite)
    except dataset_ingest.IngestError as e:
        return jsonify({"ok": False, "errors": [str(e)], "warnings": [],
                        "summary": None}), 400
    except Exception as e:                       # noqa: BLE001
        return jsonify({"ok": False, "errors": [f"上傳處理失敗: {e}"], "warnings": [],
                        "summary": None, "trace": traceback.format_exc()}), 500
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return jsonify(report)


@app.route("/datasets/delete", methods=["POST"])
def datasets_delete():
    """刪除 data/ 底下的一個資料集目錄 (前端會先要求確認)。"""
    import shutil
    target = _under_data((request.form.get("path") or "").strip())
    if not target or not os.path.isdir(target):
        return jsonify({"ok": False, "error": "無效的資料集路徑（僅能刪除 data/ 底下的目錄）"}), 400
    try:
        shutil.rmtree(target)
    except Exception as e:                       # noqa: BLE001
        return jsonify({"ok": False, "error": str(e)}), 500
    dsreg.invalidate(_DATA)
    return jsonify({"ok": True})


@app.route("/jobs-page")
def jobs_page():
    """背景工作頁。"""
    return render_template("jobs.html", nav="jobs", route_seg="jobs-page", jobs=JOBS)


@app.route("/history")
def history():
    """歷史 runs (ledger) 頁。"""
    return render_template("history.html", nav="history", route_seg="history", runs=_list_runs())


# HyperParams 欄位 (順序 = 編輯表單顯示順序)
_HP_FIELDS = [
    ("batch_size", "batch size", "int"), ("epochs", "epochs", "int"),
    ("blr", "base lr (blr)", "float"), ("layer_decay", "layer decay", "float"),
    ("drop_path", "drop path", "float"), ("weight_decay", "weight decay", "float"),
    ("warmup_epochs", "warmup epochs", "int"), ("input_size", "input size", "int"),
    ("accum_iter", "accum iter", "int"),
]


@app.route("/settings")
def settings_page():
    """整體設定頁: 左 nav 分類 ｜ 中 item ｜ 右 description (與實驗設定同格式)。"""
    values = app_settings.load()
    helps = app_optdocs.load_settings()
    groups = []
    for cid, label, keys in app_settings.SECTIONS:
        items = []
        for k in keys:
            fd = app_settings.field(k)
            if not fd:
                continue
            _k, flabel, ftype, _default, choices = fd
            items.append({"key": k, "label": flabel, "type": ftype,
                          "choices": choices, "value": values.get(k),
                          "help": helps.get(k, "")})
        groups.append({"id": cid, "label": label, "rows": items})
    return render_template(
        "settings.html", nav="settings", route_seg="settings",
        groups=groups, values=values,
    )


@app.route("/settings/save", methods=["POST"])
def settings_save():
    """寫入整體設定並即時套用 (API key 灌進 env)。"""
    app_settings.save(request.form.to_dict())
    ref = request.referrer or "."
    return redirect(f"{ref}{'&' if '?' in ref else '?'}saved=1")


@app.route("/presets")
def presets_page():
    """超參起點編輯頁。"""
    return render_template(
        "presets.html", nav="presets", route_seg="presets",
        presets=presets_mod.as_dict(),
        builtins=list(presets_mod.BUILTIN),
        fields=_HP_FIELDS,
    )


@app.route("/presets/save", methods=["POST"])
def presets_save():
    """儲存/更新一組 preset (新增或編輯)。"""
    name = request.form.get("name", "").strip()
    if not name:
        return redirect(request.referrer or "presets")
    fields = {k: request.form.get(k, "") for k, _, _ in _HP_FIELDS}
    try:
        presets_mod.save_one(name, fields)
    except Exception:
        pass
    return redirect(request.referrer or "presets")


@app.route("/presets/delete", methods=["POST"])
def presets_delete():
    """刪除自訂 preset (內建則還原為預設值)。"""
    name = request.form.get("name", "").strip()
    if name:
        presets_mod.delete(name)
    return redirect(request.referrer or "presets")


@app.route("/presets/reset", methods=["POST"])
def presets_reset():
    """全部還原為內建預設。"""
    presets_mod.reset_all()
    return redirect(request.referrer or "presets")


@app.route("/analyze", methods=["POST"])
def analyze():
    """分析 + dry-run: 安全, 不需 GPU。回傳 JSON。"""
    data_path = request.form["data_path"].strip()
    preset = request.form.get("preset", "default")
    encoder_key = request.form.get("encoder") or None
    try:
        out = single_trial(
            data_path, run_dir=os.path.join(_RUNS, "_dryrun_tmp"),
            preset=preset, encoder_key=encoder_key, dry_run=True,
        )
        return jsonify({
            "ok": True,
            "profile": out["profile"].model_dump(),
            "recipe": out["recipe"].model_dump(),
            "command": out["command"],
            "unsupported": out["unsupported"],
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"{e}",
                        "trace": traceback.format_exc()}), 400


def _job_worker(job_id: str, kwargs: dict):
    with _LOCK:
        JOBS[job_id]["status"] = "running"
    try:
        out = single_trial(**kwargs)
        t = out["trial"]
        with _LOCK:
            JOBS[job_id].update(status=t.status, primary_score=t.primary_score,
                                metrics=t.metrics, trial_id=t.trial_id,
                                message=t.message, run_dir=kwargs["run_dir"])
    except Exception as e:
        with _LOCK:
            JOBS[job_id].update(status="failed", message=str(e))


@app.route("/run", methods=["POST"])
def run():
    """背景執行單一 trial (實際訓練, 需 GPU)。"""
    data_path = request.form["data_path"].strip()
    preset = request.form.get("preset", "default")
    encoder_key = request.form.get("encoder") or None
    metric = request.form.get("primary_metric", "score")
    run_dir = os.path.join(_RUNS, time.strftime("run_%Y%m%d_%H%M%S"))

    job_id = f"job_{len(JOBS) + 1}"
    JOBS[job_id] = {"status": "queued", "data_path": data_path,
                    "preset": preset, "encoder": encoder_key, "run_dir": run_dir}
    kwargs = dict(data_path=data_path, run_dir=run_dir, preset=preset,
                  encoder_key=encoder_key, eval_cfg=EvalConfig(primary_metric=metric),
                  dry_run=False)
    threading.Thread(target=_job_worker, args=(job_id, kwargs), daemon=True).start()
    return redirect(request.referrer or ".")


@app.route("/run_full", methods=["POST"])
def run_full():
    """背景執行完整 pipeline (多 encoder × 改良迴圈 × 多 fold; LoopController)。"""
    f = request.form
    # 整體設定頁的值當預設; 表單有填的欄位覆寫。model / improve_temperature 無表單
    # 欄位, 完全由設定頁決定。
    st = app_settings.load()
    cfg = AgentConfig(data_path=f["data_path"].strip())
    cfg.advisor.type = f.get("advisor") or st["advisor_type"]
    cfg.advisor.model = st["model"]
    cfg.advisor.preset = f.get("preset", "default")
    cfg.advisor.guidance = f.get("guidance", "").strip()
    cfg.advisor.allow_code_edit = bool(f.get("allow_code_edit"))
    # 資料圍欄 (docs/data_firewall_design.md): mode 是巨集, strict 會強制關掉
    # allow_code_edit — 這裡設定順序無所謂, AgentConfig 的 validator 會再套一次。
    cfg.privacy.mode = f.get("privacy_mode") or st["privacy_mode"]
    cfg = AgentConfig.model_validate(cfg.model_dump())
    cfg.loop.num_drafts = int(f.get("num_drafts") or st["num_drafts"])
    cfg.loop.max_trials = int(f.get("max_trials") or st["max_trials"])
    cfg.loop.min_trials = int(f.get("min_trials") or st["min_trials"])
    cfg.loop.patience = int(f.get("patience") or st["patience"])
    cfg.loop.improve_temperature = st["improve_temperature"]
    cfg.loop.debug_prob = st["debug_prob"]
    cfg.loop.max_debug_depth = st["max_debug_depth"]
    cfg.loop.resume_epochs = st["resume_epochs"]
    cfg.loop.max_resumes = st["max_resumes"]
    cfg.eval.primary_metric = f.get("primary_metric") or st["primary_metric"]
    cfg.eval.aggregation = f.get("aggregation", "single")
    # 集成 (ensemble): 由整體設定套用 (docs/ensemble_design.md)
    _b = lambda v: str(v).strip().lower() in ("true", "1", "on", "yes")
    cfg.ensemble.enabled = _b(st["ensemble_enabled"])
    cfg.ensemble.method = st["ensemble_method"]
    cfg.ensemble.llm_select = _b(st["ensemble_llm_select"])
    cfg.ensemble.min_members = int(st["ensemble_min_members"])
    cfg.ensemble.max_members = int(st["ensemble_max_members"])
    cfg.ensemble.member_delta = float(st["ensemble_member_delta"])
    cfg.ensemble.in_search = _b(st["ensemble_in_search"])
    cfg.ensemble.search_patience = int(st["ensemble_search_patience"])
    cfg.ensemble.max_search_ensembles = int(st["ensemble_max_search_ensembles"])
    # 逐 fold 平行的 GPU 共用 (docs/per_fold_parallel_design.md)
    cfg.gpu.pack_low_util = _b(st["gpu_pack_low_util"])
    cfg.gpu.max_folds_per_gpu = int(st["gpu_max_folds_per_gpu"])
    cfg.gpu.pack_util_below = float(st["gpu_pack_util_below"])
    cfg.gpu.pack_mem_below = float(st["gpu_pack_mem_below"])
    cfg.stream_logs = False  # web: 不 tee 到 console; log 檔仍寫, 供輪詢

    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(_RUNS, run_name)
    # 立刻建立 run 目錄 + 初始訊息, 讓前端馬上看得到 (advisor 環境檢查等前置需要幾秒)
    os.makedirs(run_dir, exist_ok=True)
    try:
        cfg.dump_yaml(os.path.join(run_dir, "config.yaml"))
    except Exception:
        pass
    # 使用者親自提供的資料特性 (模態/部位/類別序數) — LLM 不從路徑猜這些
    _seed_user_facts(run_dir, f)
    convo.append(run_dir, "system",
                 f"實驗已建立（advisor={cfg.advisor.type}）。正在初始化決策層與挑選 encoder，請稍候…",
                 kind="status")

    job_id = f"full_{len(JOBS) + 1}"
    JOBS[job_id] = {"status": "queued", "kind": "full", "run": run_name,
                    "run_dir": run_dir, "advisor": cfg.advisor.type,
                    "encoder": f"{cfg.loop.num_drafts} drafts",
                    "preset": cfg.advisor.preset, "data_path": cfg.data_path}

    def _worker():
        with _LOCK:
            JOBS[job_id]["status"] = "running"
        try:
            if cfg.eval.aggregation == "per_fold":
                from ..fold_fleet import run_fleet     # 逐 fold 平行 (各自子執行/GPU)
                out = run_fleet(cfg, run_dir=run_dir)
            else:
                out = auto_finetune.run(cfg, run_dir=run_dir)
            with _LOCK:
                JOBS[job_id].update(
                    status="finished", n_trials=out["n_trials"],
                    primary_score=(out["best"].primary_score if out["best"] else None),
                    report=out["report_path"])
        except Exception as e:
            with _LOCK:
                JOBS[job_id].update(status="failed", message=str(e),
                                    trace=traceback.format_exc())
            # 把錯誤寫進對話, 讓工作台聊天框直接看到 (例: advisor=llm 環境不對)
            try:
                convo.append(run_dir, "system", f"實驗失敗：{e}", kind="final")
            except Exception:
                pass

    threading.Thread(target=_worker, args=(), daemon=True).start()
    if f.get("ajax"):
        return jsonify({"ok": True, "run": run_name})
    return redirect(request.referrer or ".")


@app.route("/resume_run", methods=["POST"])
def resume_run():
    """繼續之前停止/失敗的實驗: 從 run 目錄的 config.yaml + ledger 重建狀態接續。"""
    run = request.form.get("run", "").strip()
    run_dir = os.path.realpath(os.path.join(_RUNS, run))
    if not run or not run_dir.startswith(os.path.realpath(_RUNS) + os.sep) \
            or not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": "無效的 run"}), 400
    cfg_path = os.path.join(run_dir, "config.yaml")
    if not os.path.isfile(cfg_path):
        return jsonify({"ok": False, "error": "此 run 沒有 config.yaml（可能是舊版單一 trial），無法繼續"}), 400
    with _LOCK:
        busy = any(v.get("run") == run and v.get("status") in ("queued", "running")
                   for v in JOBS.values())
    if busy:
        return jsonify({"ok": False, "error": "此 run 正在執行中，不需要繼續"}), 400
    try:
        cfg = AgentConfig.load(cfg_path)
    except Exception as e:
        return jsonify({"ok": False, "error": f"config.yaml 載入失敗：{e}"}), 400
    cfg.stream_logs = False
    # 可選: 放寬停止條件 / 補充引導方向
    if (mt := request.form.get("max_trials", "").strip()):
        try:
            cfg.loop.max_trials = int(mt)
        except ValueError:
            pass
    if (g := request.form.get("guidance", "").strip()):
        cfg.advisor.guidance = g

    convo.clear_stop(run_dir)   # 立即清旗標, 讓前端「已中斷」橫幅先消失
    convo.append(run_dir, "system",
                 "▶ 已排入繼續實驗（重建解答樹中，請稍候…）", kind="status")

    job_id = f"resume_{len(JOBS) + 1}"
    JOBS[job_id] = {"status": "queued", "kind": "resume", "run": run,
                    "run_dir": run_dir, "advisor": cfg.advisor.type,
                    "encoder": "resume", "preset": cfg.advisor.preset,
                    "data_path": cfg.data_path}

    def _worker():
        with _LOCK:
            JOBS[job_id]["status"] = "running"
        try:
            out = auto_finetune.run(cfg, run_dir=run_dir, resume=True)
            with _LOCK:
                JOBS[job_id].update(
                    status="finished", n_trials=out["n_trials"],
                    primary_score=(out["best"].primary_score if out["best"] else None),
                    report=out["report_path"])
        except Exception as e:
            with _LOCK:
                JOBS[job_id].update(status="failed", message=str(e),
                                    trace=traceback.format_exc())
            try:
                convo.append(run_dir, "system", f"繼續實驗失敗：{e}", kind="final")
            except Exception:
                pass

    threading.Thread(target=_worker, args=(), daemon=True).start()
    return jsonify({"ok": True, "run": run})


@app.route("/run_status")
def run_status():
    """整體進度: ledger 全 trial (含完整 Recipe 參數) + 當前 trial epoch + report.md。"""
    run = request.args.get("run") or (_run_names()[0] if _run_names() else None)
    if not run:
        return jsonify({"ok": False, "error": "尚無任何 run"}), 404
    # 允許巢狀路徑 run=<parent>/foldN (逐 fold 平行的子執行); 但限制在 _RUNS 內
    run_dir = os.path.realpath(os.path.join(_RUNS, run))
    if not (run_dir == os.path.realpath(_RUNS)
            or run_dir.startswith(os.path.realpath(_RUNS) + os.sep)):
        return jsonify({"ok": False, "error": "無效的 run 路徑"}), 400
    if not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": f"找不到 run: {run}"}), 404

    trials = []
    try:
        trials = [t.model_dump() for t in Ledger(run_dir).history()]
    except Exception:
        pass

    current = None
    lp = _latest_log(run_dir)
    if lp:
        current = _parse_progress(lp)
        current["log_file"] = os.path.basename(lp)

    report = None
    rp = os.path.join(run_dir, "report.md")
    if os.path.isfile(rp):
        with open(rp, encoding="utf8") as fh:
            report = fh.read()

    # 每個 trial: confusion matrix 圖 + 逐 epoch 曲線 (從各自的 log 抽)
    for t in trials:
        tid = t["trial_id"]
        cm = os.path.join(run_dir, "trials", tid, "confusion_matrix_test.jpg")
        t["cm_img"] = (f"artifact/?run={run}&path=trials/{tid}/"
                       f"confusion_matrix_test.jpg") if os.path.isfile(cm) else None
        tlog = os.path.join(run_dir, "logs", f"log_{tid}.txt")
        t["curves"] = _parse_curves(tlog) if os.path.isfile(tlog) else {}

    # 本次 run 的完整設定 (config.yaml 快照)
    config_text = None
    cp = os.path.join(run_dir, "config.yaml")
    if os.path.isfile(cp):
        with open(cp, encoding="utf8") as fh:
            config_text = fh.read()

    # 進行中的 trial (ledger 尚未有): 完整參數 + 即時曲線
    running_trial = None
    ctp = os.path.join(run_dir, "current_trial.json")
    if os.path.isfile(ctp):
        try:
            with open(ctp, encoding="utf8") as fh:
                ct = json.load(fh)
            done_ids = {t["trial_id"] for t in trials}
            tid = ct.get("trial_id")
            if tid and tid not in done_ids:
                tlog = os.path.join(run_dir, "logs", f"log_{tid}.txt")
                curves = _parse_curves(tlog) if os.path.isfile(tlog) else {}
                prog = _parse_progress(tlog) if os.path.isfile(tlog) else {}
                running_trial = {
                    "trial_id": tid, "recipe": ct.get("recipe", {}),
                    "status": prog.get("status", "running"),
                    # 進行中還沒跑最終 test → 這裡的分數/指標其實是 val (前端會標明)
                    "is_running": True,
                    "primary_score": prog.get("best_score") or 0.0,
                    "metrics": prog.get("val") or {},
                    "curves": curves, "cm_img": None,
                    "cur_epoch": prog.get("cur_epoch"),
                    "total_epochs": prog.get("total_epochs"),
                    "percent": prog.get("percent", 0.0),
                }
        except Exception:
            pass

    # AIDE 式解答樹 (search_tree.json; LoopController 每個 trial 後更新)
    search_tree = None
    stp = os.path.join(run_dir, "search_tree.json")
    if os.path.isfile(stp):
        try:
            with open(stp, encoding="utf8") as fh:
                search_tree = json.load(fh)
        except Exception:
            pass

    # per_fold: folds.json —— 兩種模式
    #   per_fold_parallel: 各 fold 是獨立子執行 (parent/foldN) → 回 parallel_folds 供 UI 切換
    #   per_fold (舊·單 run 循序): 各 fold 的解答樹附在同一 run → 回 folds (含 tree)
    folds = None
    parallel_folds = None
    fjp = os.path.join(run_dir, "folds.json")
    if os.path.isfile(fjp):
        try:
            with open(fjp, encoding="utf8") as fh:
                fj = json.load(fh)
            if fj.get("mode") == "per_fold_parallel":
                parallel_folds = {"folds": fj.get("folds") or [],
                                  "devices": fj.get("devices") or [],
                                  "done": bool(fj.get("done"))}
            else:
                folds = fj.get("folds") or []
                cur = fj.get("current", -1)
                for ent in folds:
                    tfile = ("search_tree.json" if ent.get("index") == cur
                             else ent.get("tree"))
                    ent["tree"] = None
                    tp = os.path.join(run_dir, tfile) if tfile else None
                    if tp and os.path.isfile(tp):
                        try:
                            with open(tp, encoding="utf8") as th:
                                ent["tree"] = json.load(th)
                        except Exception:
                            ent["tree"] = None
        except Exception:
            folds = None

    # 若這是逐 fold 平行的子執行 (parent/foldN): 附上『父』的 parallel_folds，讓左側
    # fold 選擇器即使在檢視某個 fold 時也能刷新各 fold 的即時狀態 (running→finished)。
    if parallel_folds is None:
        parent_dir = os.path.dirname(run_dir)
        pfp = os.path.join(parent_dir, "folds.json")
        if os.path.isfile(pfp):
            try:
                with open(pfp, encoding="utf8") as fh:
                    pj = json.load(fh)
                subs = {os.path.realpath(os.path.join(parent_dir, e.get("subdir", "")))
                        for e in pj.get("folds", [])}
                if pj.get("mode") == "per_fold_parallel" and run_dir in subs:
                    parallel_folds = {"folds": pj.get("folds") or [],
                                      "devices": pj.get("devices") or [],
                                      "done": bool(pj.get("done"))}
            except Exception:
                pass

    # LLM 完整 prompt/回應記錄 (llm_calls.jsonl); 只回最近 30 筆以控大小
    llm_calls = []
    lcp = os.path.join(run_dir, "llm_calls.jsonl")
    if os.path.isfile(lcp):
        try:
            with open(lcp, encoding="utf8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        llm_calls.append(json.loads(line))
        except Exception:
            pass
        llm_calls = llm_calls[-30:]

    # QA agent 串流中的部份回答 (qa_stream.json; 完成即刪 → 正式訊息入 conversation)
    qa_stream = None
    qsp = os.path.join(run_dir, "qa_stream.json")
    if os.path.isfile(qsp):
        try:
            with open(qsp, encoding="utf8") as fh:
                qa_stream = json.load(fh)
        except Exception:
            pass

    # 資料圍欄: 使用者已回答的事實 + 出口稽核摘要 (讓使用者看得到攔截狀況)
    user_facts = load_user_facts(run_dir).model_dump()
    audit = egress_mod.read_audit(run_dir)
    privacy = {
        "n_egress": sum(1 for a in audit if a.get("kind") == "request"),
        "n_blocked": sum(1 for a in audit if a.get("verdict") == "blocked"),
        "n_warned": sum(1 for a in audit if a.get("verdict") == "warn"),
        "mode": next((a.get("mode") for a in reversed(audit) if a.get("mode")),
                     None),
        "recent": [{k: a.get(k) for k in
                    ("ts", "label", "kind", "verdict", "violations", "n_chars")}
                   for a in audit[-40:]],
    }

    return jsonify({"ok": True, "run": run, "trials": trials,
                    "current": current, "report": report, "n_done": len(trials),
                    "config": config_text, "running_trial": running_trial,
                    "conversation": convo.read(run_dir),
                    "stopped": convo.stop_requested(run_dir)[0],
                    "qa_stream": qa_stream, "user_facts": user_facts,
                    "privacy": privacy, "folds": folds,
                    "parallel_folds": parallel_folds,
                    "llm_calls": llm_calls, "search_tree": search_tree})


def _seed_user_facts(run_dir: str, form) -> None:
    """把「開始實驗」表單上的資料特性寫進 user_facts.json (管道 B 的起點)。

    modality / anatomy 刻意由使用者指定 — 舊版是從資料集路徑關鍵字猜 (見
    dataset_analyzer._modality_hint)，那既不準又等於把路徑內容送進決策。
    """
    uf = load_user_facts(run_dir)
    ts = time.time()
    picked: list[tuple[str, str]] = []
    if (m := form.get("modality", "").strip()) in _MODALITIES:
        uf.modality = m
        picked.append(("modality", m))
    if (a := form.get("anatomy", "").strip()) in _ANATOMIES:
        uf.anatomy = a
        picked.append(("anatomy", a))
    if (o := form.get("class_ordinal", "").strip()) in ("0", "1"):
        uf.class_ordinal = (o == "1")
        picked.append(("class_ordinal", "true" if o == "1" else "false"))
    for key, value in picked:
        uf.answers.append(UserAnswer(key=key, question="（建立實驗時填寫）",
                                     value=value, ts=ts))
    if picked:
        save_user_facts(run_dir, uf)


@app.route("/answer", methods=["POST"])
def answer_questions():
    """使用者回答決策層的提問 (管道 B) — 寫入 user_facts.json, 下一輪決策就會看到。

    這裡的內容是使用者**自願**提供的, 會原文進入 prompt; 前端已明確告知。
    """
    run = request.form.get("run", "").strip()
    run_dir = os.path.realpath(os.path.join(_RUNS, run))
    if not run or not run_dir.startswith(os.path.realpath(_RUNS) + os.sep) \
            or not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": "無效的 run"}), 400
    try:
        items = json.loads(request.form.get("answers", "[]"))
    except Exception:
        return jsonify({"ok": False, "error": "answers 不是合法 JSON"}), 400
    if not isinstance(items, list) or not items:
        return jsonify({"ok": False, "error": "沒有任何回答"}), 400

    uf = load_user_facts(run_dir)
    ts = time.time()
    lines = []
    for it in items:
        key = str(it.get("key", "")).strip()
        value = str(it.get("value", "")).strip()
        if not key or not value:
            continue
        uf.answers.append(UserAnswer(key=key, question=str(it.get("question", "")),
                                     value=value[:500], ts=ts))
        # 結構化欄位: 直接映射, 讓 DatasetFacts 也帶得到
        if key == "modality" and value in _MODALITIES:
            uf.modality = value
        elif key == "anatomy" and value in _ANATOMIES:
            uf.anatomy = value
        elif key == "class_ordinal":
            uf.class_ordinal = value.lower() in ("true", "1", "是", "yes")
        lines.append(f"・{it.get('question') or key}：{value}")
    if not lines:
        return jsonify({"ok": False, "error": "沒有任何有效回答"}), 400
    save_user_facts(run_dir, uf)
    convo.append(run_dir, "user", "我的回答：\n" + "\n".join(lines),
                 kind="user_msg")
    return jsonify({"ok": True, "n": len(lines)})


@app.route("/discuss", methods=["POST"])
def discuss():
    """使用者加入討論 — 先由獨立問答 agent 立即以既有實驗資料回答 (背景),
    其結論標記【QA 結論】寫回討論, 主 agent 下一輪決策時優先採納;
    訊息本身也照舊成為下一輪 LLM 決策的參考。"""
    run = request.form.get("run", "").strip()
    text = request.form.get("text", "").strip()
    run_dir = os.path.join(_RUNS, run)
    if not run or not os.path.isdir(run_dir) or not text:
        return jsonify({"ok": False, "error": "缺少 run 或 text"}), 400
    convo.append(run_dir, "user", text, kind="user_msg")
    # 問答 agent (advisor=llm 時): 背景回答, 不阻塞這個請求; 失敗不影響原流程
    from .. import qa_agent
    qa_agent.answer_async(run_dir, text)
    return jsonify({"ok": True})


@app.route("/delete_run", methods=["POST"])
def delete_run():
    """刪除一個 run (整個 runs/<run> 目錄)。"""
    import shutil
    run = request.form.get("run", "").strip()
    run_dir = os.path.realpath(os.path.join(_RUNS, run))
    # 安全: 限定在 runs/ 底下, 且必須是既有目錄
    if not run or not run_dir.startswith(os.path.realpath(_RUNS) + os.sep) \
            or not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": "無效的 run"}), 400
    try:
        shutil.rmtree(run_dir)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    # 一併移除對應的 in-memory job
    with _LOCK:
        for jid in [k for k, v in JOBS.items() if v.get("run") == run]:
            JOBS.pop(jid, None)
    return jsonify({"ok": True})


@app.route("/stop_run", methods=["POST"])
def stop_run():
    """使用者中斷實驗 — LoopController 會在下一輪前停止。"""
    run = request.form.get("run", "").strip()
    run_dir = os.path.join(_RUNS, run)
    if not run or not os.path.isdir(run_dir):
        return jsonify({"ok": False, "error": "缺少 run"}), 400
    convo.set_stop(run_dir, reason=request.form.get("reason", "使用者於 web 中斷"))
    convo.append(run_dir, "system", "已送出中斷要求，將在目前 trial 結束後停止。", kind="final")
    return jsonify({"ok": True})


@app.route("/artifact")
def artifact():
    """安全地提供 run 目錄內的產物 (圖檔), 限定在 runs/<run>/ 下。"""
    from flask import send_file, abort
    run = request.args.get("run", "")
    rel = request.args.get("path", "")
    run_dir = os.path.realpath(os.path.join(_RUNS, run))
    target = os.path.realpath(os.path.join(run_dir, rel))
    if not run_dir.startswith(os.path.realpath(_RUNS)) or \
            not target.startswith(run_dir + os.sep) or not os.path.isfile(target):
        abort(404)
    return send_file(target)


@app.route("/jobs")
def jobs():
    return jsonify(JOBS)


@app.route("/gpu")
def gpu():
    return jsonify({"allocatable": gpu_allocatable(0, timeout=15)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    os.makedirs(_RUNS, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

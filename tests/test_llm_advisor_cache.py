"""LLMAdvisor 的 prompt cache 策略: TTL 決策、breakpoint 下限、前綴穩定性。

這三件事壞掉時 API 都不會報錯 —— 只會安靜地每輪重寫幾萬 token 的 cache 卻
永遠讀不到, 所以只能靠測試釘住。
"""
from __future__ import annotations

import json

import pytest

from agent import llm_advisor as mod
from agent.llm_advisor import LLMAdvisor
from agent.privacy.context import PrivacyContext, save_user_facts
from agent.privacy.facts import AnalysisFacts, UserFacts

THRESHOLD = LLMAdvisor._CACHE_TTL_THRESHOLD_S


@pytest.fixture
def replay(monkeypatch):
    """以指定的呼叫間隔 (秒) 重放, 回傳每次請求真正選到的 ttl。"""
    def _replay(gaps: list[float]) -> list[str]:
        clock = {"t": 1_000_000.0}
        monkeypatch.setattr(mod.time, "time", lambda: clock["t"])
        adv = LLMAdvisor()
        out = [adv._next_cache_ttl()]          # 本 run 第一次呼叫
        for gap in gaps:
            clock["t"] += gap
            out.append(adv._next_cache_ttl())
        return out
    return _replay


def test_short_gaps_stay_5m(replay):
    """小資料集: 訓練 3 分鐘就跑完, 5m entry 撐得住 → 不多付 1h 的 2x 寫入。"""
    assert set(replay([20.0, 11.0, 180.0, 15.0, 180.0])) == {"5m"}


def test_one_long_gap_switches_to_1h_and_sticks(replay):
    """一次長間隔就代表 5m 會死在半路; 切到 1h 後不再切回去。"""
    ttls = replay([20.0, 11.0, 705.0, 11.0, 11.0])
    assert ttls[:3] == ["5m", "5m", "5m"]   # 第一次長間隔前無從得知, 必然 miss 一次
    assert ttls[3:] == ["1h", "1h", "1h"]   # 之後一律 1h — 舊的 avg 版會在此掉回 5m


def test_bursts_do_not_dilute_the_long_gap(replay):
    """重試/連發的近 0 秒間隔不得把判斷拉回 5m (avg 會, max 不會)。

    這正是 fold4 00:51 的情境: 平均 (19+11+705+11)/4 = 186s < 240s → 舊版選 5m,
    於是 61k 字元的資料塊用 5m 寫進去, 下一輪 12 分鐘後必然讀不到。
    """
    assert replay([19.0, 11.0, 705.0] + [0.2] * 20)[-1] == "1h"


def test_first_call_is_conservative(replay):
    """尚無任何間隔可觀察時用 5m。"""
    assert replay([])[0] == "5m"


def test_threshold_is_exclusive(replay):
    assert replay([THRESHOLD])[-1] == "5m"
    assert replay([THRESHOLD + 0.1])[-1] == "1h"


def test_gap_beyond_1h_is_not_evidence(replay):
    """比 1h 還長的間隔連 1h cache 都活不過 → 該用比較便宜的 5m 寫入。"""
    assert replay([LLMAdvisor._CACHE_GAP_CEILING_S + 60.0])[-1] == "5m"


# --------------------------------------------------------------------------
# resume: 從既有 llm_calls.jsonl 接續統計, 不必再浪費一輪重新學節奏
# --------------------------------------------------------------------------
def _write_log(path, gaps: list[float]) -> None:
    t = 1_000_000.0
    with open(path, "w", encoding="utf8") as f:
        for gap in [0.0] + gaps:
            t += gap
            f.write(json.dumps({"ts": t, "label": "review_and_decide"}) + "\n")


def test_resume_seeds_from_previous_log(tmp_path):
    """上一段跑過 12 分鐘的訓練空檔 → resume 後第一次呼叫就該選 1h。"""
    p = tmp_path / "llm_calls.jsonl"
    _write_log(p, [19.0, 11.0, 705.0, 11.0])
    adv = LLMAdvisor()
    assert adv._next_cache_ttl() == "5m"      # 未接續前 (對照組)

    adv = LLMAdvisor()
    adv.log_path = str(p)
    assert adv._max_gap == pytest.approx(705.0)
    assert adv._next_cache_ttl() == "1h"      # 不用再 miss 一輪


def test_resume_does_not_seed_from_short_gaps(tmp_path):
    """上一段都是小資料集的短間隔 → 維持 5m, 別因為 resume 就多付 2x。"""
    p = tmp_path / "llm_calls.jsonl"
    _write_log(p, [19.0, 11.0, 180.0, 15.0])
    adv = LLMAdvisor()
    adv.log_path = str(p)
    assert adv._next_cache_ttl() == "5m"


def test_resume_ignores_the_interruption_gap(tmp_path):
    """中斷隔夜留下的超長間隔不算數 (fold4 那次隔了 8 小時)。"""
    p = tmp_path / "llm_calls.jsonl"
    _write_log(p, [19.0, 11.0, 8 * 3600.0, 11.0])
    adv = LLMAdvisor()
    adv.log_path = str(p)
    assert adv._max_gap == pytest.approx(19.0)
    assert adv._next_cache_ttl() == "5m"


def test_seeding_survives_a_corrupt_log(tmp_path):
    """壞行只跳過 —— ttl 只是啟發式, 不值得為它擋住整個 run。"""
    p = tmp_path / "llm_calls.jsonl"
    p.write_text('{"ts": 1000000.0}\nnot json\n{"no_ts": 1}\n[]\n'
                 '{"ts": 1000705.0}\n', encoding="utf8")
    adv = LLMAdvisor()
    adv.log_path = str(p)                     # 不得拋錯
    assert adv._next_cache_ttl() == "1h"

    adv = LLMAdvisor()
    adv.log_path = str(tmp_path / "does_not_exist.jsonl")   # 全新 run
    assert adv._next_cache_ttl() == "5m"


def test_log_path_still_round_trips(tmp_path):
    """log_path 改成 property 後仍要能讀回來 (_log_call 靠它)。"""
    adv = LLMAdvisor()
    assert adv.log_path is None
    adv.log_path = str(tmp_path / "x.jsonl")
    assert adv.log_path == str(tmp_path / "x.jsonl")
    adv._log_call({"ts": 1.0, "label": "t"})
    assert json.loads((tmp_path / "x.jsonl").read_text(encoding="utf8"))["label"] == "t"


# --------------------------------------------------------------------------
# breakpoint 下限 — 依模型不同, 且不隨世代單調遞減
# --------------------------------------------------------------------------
def test_cache_min_chars_is_model_specific():
    assert LLMAdvisor(model="claude-opus-5")._cache_min_chars == 512 * 4
    assert LLMAdvisor(model="claude-opus-4-8")._cache_min_chars == 1024 * 4
    # 4.7 的下限比 4.8 高 —— 不是越新越小, 不能用世代推
    assert LLMAdvisor(model="claude-opus-4-7")._cache_min_chars == 2048 * 4
    # 不認得的模型取最保守值
    assert (LLMAdvisor(model="claude-future-9")._cache_min_chars
            == LLMAdvisor._CACHE_MIN_TOKENS_DEFAULT * 4)


def test_cache_min_chars_understands_proxy_model_ids():
    """經 OpenRouter 時 id 長得不一樣 (provider 前綴 / 點號版號 / 變體後綴)。

    沒正規化的話會全部落到最保守的 4096, 白白少下 breakpoint。
    """
    assert LLMAdvisor(model="anthropic/claude-opus-4.5")._cache_min_chars == 4096 * 4
    assert LLMAdvisor(model="anthropic/claude-opus-4.8")._cache_min_chars == 1024 * 4
    assert LLMAdvisor(model="~anthropic/claude-sonnet-5:beta")._cache_min_chars == 1024 * 4


def test_breakpoint_only_when_prefix_is_long_enough():
    adv = LLMAdvisor(model="claude-opus-5")
    n = adv._cache_min_chars

    short = adv._user_content(["x" * (n - 1)], "本輪指令", "5m")
    assert "cache_control" not in short[0]

    ok = adv._user_content(["x" * n], "本輪指令", "1h")
    assert ok[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert "cache_control" not in ok[1]        # volatile 段不得下 breakpoint


def test_no_stable_segments_is_a_single_block():
    assert LLMAdvisor()._user_content([], "本輪指令", "5m") == [
        {"type": "text", "text": "本輪指令"}]


def test_opus5_covers_the_no_history_calls():
    """review_search_choice 那類不帶 history 的呼叫 (~2.4k-3.2k 字元) 在
    opus-5 下要能快取; 舊的固定 4000 字元門檻會把它們全擋掉。"""
    assert LLMAdvisor(model="claude-opus-5")._cache_min_chars <= 2438


# --------------------------------------------------------------------------
# 前綴穩定性
# --------------------------------------------------------------------------
def _trial_dict(i: int) -> dict:
    # 實測一筆 trial 約 7k 字元 (含逐 epoch 曲線), 這裡照量級模擬 —— 太小的話
    # 累計長度過不了 _cache_min_chars, 就測不到 breakpoint 的落點。
    return {"trial_id": f"t{i}", "score": 0.5 + i / 100,
            "val_loss_curve": [round(1.0 - j / 1000, 4) for j in range(700)]}


def _advisor_with_trials(n: int) -> LLMAdvisor:
    """帶 n 筆 trial 的 advisor (_hist 直接給 dict, 免去組 TrialResult)。"""
    adv = LLMAdvisor(model="claude-opus-5")
    hist = [_trial_dict(i) for i in range(n)]
    adv._hist = lambda history: hist[:len(history)]
    return adv


def test_analyses_block_is_order_independent():
    """分析結果排在 trials 歷史前面 —— 順序一變, 後面整段 (50k+) 全失效。

    resume 後 LLM 重新點名分析器的順序不保證相同, 實測就因此在 char 1548 斷掉。
    """
    a = AnalysisFacts(key="image_stats", ok=True, result={"width_p5": 240})
    b = AnalysisFacts(key="split_leakage", ok=True, result={"n_hashed_train": 861})

    one, other = LLMAdvisor(), LLMAdvisor()
    one.analyses = [a, b]
    other.analyses = [b, a]
    assert one._data_block(None) == other._data_block(None)


def test_analyses_grow_append_only():
    """新增一個分析器時, 既有的段必須**逐字且逐段**不變。

    只有內容是前綴還不夠 —— 快取比對以 content block 邊界為單位, 所以每個分析器
    各自成段, 新增時前面的段連邊界都不會動。

    (邊界: 新 key 若排序在既有 key 之前, 它會插在中間 —— 那是 append-only 本身的
    限制, 不是排序造成的。實務上分析器都在 plan_information 階段就點完, 之後才進
    決策迴圈, 中途插入的機會很低。)
    """
    a = AnalysisFacts(key="aaa_first", ok=True, result={})
    b = AnalysisFacts(key="zzz_last", ok=True, result={})
    before, after = LLMAdvisor(), LLMAdvisor()
    before.analyses = [a]
    after.analyses = [a, b]
    old, new = before._data_block(None), after._data_block(None)
    assert new[:len(old)] == old        # 前面的段一字未動
    assert len(new) == len(old) + 1     # 只多出新分析器那一段


# --------------------------------------------------------------------------
# 核心: trials 每筆自成一段 —— 這是大區塊能命中的前提
# --------------------------------------------------------------------------
def test_each_trial_is_its_own_segment():
    adv = _advisor_with_trials(5)
    segs = adv._data_block(None, history=[None] * 5)
    assert segs[-5:] == [json.dumps(_trial_dict(i), ensure_ascii=False) + "\n"
                         for i in range(5)]


def test_adding_a_trial_leaves_every_earlier_segment_untouched():
    """這次改動的重點。

    早期版本把 trials 併成一個會長大的 block: 內容確實是乾淨的 append-only 前綴鏈
    (實測 66512 → 74325 → 80729 … 每次都 EXTENDS), 但上一輪的斷點落在這一輪那個
    block 的中間, 沒有邊界可對, 於是幾萬 token 每輪重寫卻一次都沒讀到。
    """
    adv = _advisor_with_trials(6)
    before = adv._data_block(None, history=[None] * 5)
    after = adv._data_block(None, history=[None] * 6)
    assert after[:len(before)] == before      # ← 逐段逐字不變, 邊界也沒動
    assert len(after) == len(before) + 1


def test_previous_breakpoint_survives_the_next_round():
    """上一輪的 breakpoint 位移, 這一輪必須仍落在某個段的邊界上 (否則讀不到)。"""
    adv = _advisor_with_trials(12)
    prev = adv._data_block(None, history=[None] * 11)
    cur = adv._data_block(None, history=[None] * 12)

    prev_offsets = adv._cache_log_fields(prev)["cache_breakpoints"]
    cur_boundaries = {sum(len(s) for s in cur[:i + 1]) for i in range(len(cur))}
    assert prev_offsets, "上一輪應該有下 breakpoint"
    assert set(prev_offsets) <= cur_boundaries


def test_lookback_stays_within_20_blocks():
    """breakpoint 只往回找 20 個 block —— 每輪只多 1 段, 遠在額度內。"""
    adv = _advisor_with_trials(30)
    prev = adv._data_block(None, history=[None] * 29)
    cur = adv._data_block(None, history=[None] * 30)
    assert len(cur) - len(prev) == 1


# --------------------------------------------------------------------------
# 使用者事實: 唯一會「改寫」的一段, 必須排在 trials 之後並自成一個 breakpoint
# --------------------------------------------------------------------------
@pytest.fixture
def adv_with_run_dir(tmp_path):
    ctx = PrivacyContext.strict_for(None)
    ctx.run_dir = str(tmp_path)
    return LLMAdvisor(model="claude-opus-5", privacy=ctx), tmp_path


def test_user_facts_are_a_separate_trailing_segment(adv_with_run_dir):
    adv, run_dir = adv_with_run_dir
    assert len(adv._data_block(None)) == 1          # 尚未填寫 → 只有 core

    save_user_facts(str(run_dir), UserFacts(anatomy="eye"))
    segs = adv._data_block(None)
    assert len(segs) == 2
    assert "【使用者親自提供的事實】" in segs[-1]
    assert "【使用者親自提供的事實】" not in segs[0]


def test_user_answer_does_not_invalidate_the_trials_prefix(adv_with_run_dir):
    """核心保證: 使用者回答新問題後, 前面那段 (含 50k+ trials) 必須逐字不變。

    修正前使用者事實排在 trials **之前**, 一回答就把整段前綴打掉。
    """
    adv, run_dir = adv_with_run_dir
    save_user_facts(str(run_dir), UserFacts(anatomy="eye"))
    core, tail = adv._data_block(None)

    save_user_facts(str(run_dir), UserFacts(anatomy="eye", modality="fundus"))
    core2, tail2 = adv._data_block(None)
    assert core2 == core        # ← 這一行就是這次改動的重點
    assert tail2 != tail


def test_trailing_segment_gets_its_own_breakpoint():
    """尾段很短, 但接在 50k 之後 —— 門檻要看累計長度, 不是單段長度。"""
    adv = LLMAdvisor(model="claude-opus-5")
    blocks = adv._user_content(["x" * 50_000, "y" * 120], "本輪指令", "1h")
    assert [("cache_control" in b) for b in blocks] == [True, True, False]
    assert len({b["cache_control"]["ttl"] for b in blocks[:2]}) == 1  # ttl 必須一致


def test_breakpoints_stay_within_budget():
    """一次請求最多 4 個; system 佔 1, user message 不得超過 3。"""
    adv = LLMAdvisor(model="claude-opus-5")
    blocks = adv._user_content(["x" * 10_000] * 6, "本輪指令", "5m")
    marked = [i for i, b in enumerate(blocks) if "cache_control" in b]
    assert len(marked) == LLMAdvisor._MAX_USER_BREAKPOINTS
    assert marked[-1] == len(blocks) - 2        # 保留離 volatile 最近的幾個

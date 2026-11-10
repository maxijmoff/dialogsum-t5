"""DialogSum AI — Streamlit demo of fine-tuned T5-small and T5-base.

Data: official DialogSum test set. Validation metrics: exported Kaggle results.
Inference: selected Hugging Face repositories, loaded on demand.
"""

from __future__ import annotations

import gc
import html
import json
import os
import re
import time
from pathlib import Path
from io import StringIO

import pandas as pd
import requests
import streamlit as st
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_URL = (
    "https://raw.githubusercontent.com/cylnlp/dialogsum/main/"
    "DialogSum_Data/dialogsum.test.jsonl"
)
MANIFEST_PATH = ROOT / "model_manifest.json"
VALIDATION_PATH = ROOT / "data" / "hasil_validasi_20_skenario.csv"
TEST_SUMMARY_PATH = ROOT / "data" / "hasil_uji_4_model.csv"
MAX_INPUT = 512
MAX_OUTPUT = 128
METRICS = ("ROUGE-1", "ROUGE-2", "ROUGE-L")
ROUGE_KEYS = {"ROUGE-1": "rouge1", "ROUGE-2": "rouge2", "ROUGE-L": "rougeL"}

st.set_page_config(
    page_title="DialogSum AI | Peringkasan Dialog Abstraktif",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
[data-testid="stAppViewContainer"] {background: #f5f6f4;}
.block-container {max-width: 1250px; padding-top: 1.35rem; padding-bottom: 3rem;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 18px;}
h1,h2,h3 {letter-spacing: -.025em;}
.dialog-header {background:#fff; padding:18px 22px; border-radius:18px;
                margin-bottom:18px; border:1px solid #e7e9e6;}
.dialog-header h2 {margin:0; font-size:1.55rem; color:#183c3b;}
.dialog-header p {margin:3px 0 0; color:#71817e; font-size:.88rem;}
.small-label {font-weight:700; color:#254b49; font-size:.94rem; margin:0 0 9px;}
.speaker-area {background:#f8faf9; padding:14px; border-radius:14px;
               border:1px solid #edf0ee;}
.bubble {padding:10px 12px; max-width:82%; display:inline-block;
         border-radius:15px; margin:5px 0; font-size:.92rem; line-height:1.5;}
.bubble-left {background:#e9edeb; color:#283636;}
.bubble-right {background:#e0eaf9; color:#253846;}
.bubble-name {font-size:.7rem; font-weight:700; margin-bottom:2px; opacity:.75;}
.bubble-row-left {text-align:left;}
.bubble-row-right {text-align:right;}
.meta-pill {padding:6px 11px; background:#f0f6f5; border:1px solid #dce9e5;
            border-radius:9px; font-size:.85rem; display:inline-block; margin:3px 5px 3px 0;}
.note {font-size:.83rem; color:#5c6a67;}
.section-title {font-size:1.1rem; font-weight:750; color:#173f3b; margin-bottom:10px;}
.metric-grid {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px;}
.metric-card {background:#eef7f6; border:1px solid #dcebe9; border-radius:13px;
              padding:12px 12px 9px; min-width:0;}
.metric-title {color:#546965; font-size:.77rem; font-weight:700;}
.metric-f1 {font-size:1.65rem; font-weight:800; color:#09665d; margin:4px 0 7px;}
.metric-sub {color:#57706b; font-size:.75rem; border-top:1px solid #d4e7e3;
              padding-top:7px; display:flex; justify-content:space-between; gap:4px;}
.summ-text {background:#f8faf9; border-left:4px solid #138175; border-radius:8px;
            padding:13px 15px; min-height:92px; line-height:1.55; margin:8px 0;}
.tag {font-size:.73rem; background:#e0f2ed; color:#096457; border-radius:40px;
      padding:4px 9px; font-weight:700; white-space:nowrap;}
@media (max-width:780px){.metric-grid{gap:5px;}.metric-card{padding:8px 6px;}
.metric-f1{font-size:1.22rem;}.metric-sub{font-size:.65rem;}}
</style>
    """,
    unsafe_allow_html=True,
)


def safe(value: object) -> str:
    return html.escape(str(value))


def format_score(value: float) -> str:
    return f"{float(value):.4f}".replace(".", ",")


def count_words(text: str) -> int:
    return len(text.split())


def cleaned(text: str) -> str:
    """Training notebook's clean_text: light whitespace normalization."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def topic_normalize(text: object) -> str:
    if pd.isna(text):
        return ""
    return " ".join(str(text).lower().split())


@st.cache_data(show_spinner="Memuat data uji DialogSum...")
def load_test() -> pd.DataFrame:
    local_jsonl = ROOT / "data" / "dialogsum.test.jsonl"
    if local_jsonl.exists():
        df = pd.read_json(local_jsonl, lines=True)
    else:
        response = requests.get(DATA_URL, timeout=45)
        response.raise_for_status()
        df = pd.read_json(StringIO(response.text), lines=True)
    required = ["dialogue", "summary1", "summary2", "summary3", "topic1", "topic2", "topic3"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom dataset hilang: {missing}")
    df = df.copy().reset_index(drop=True)
    if "fname" not in df.columns:
        df["fname"] = [f"test_{i}" for i in range(len(df))]
    df["fname"] = df["fname"].astype(str)
    if not df["fname"].is_unique:
        raise ValueError("fname pada data uji tidak unik")
    for col in ["dialogue", "summary1", "summary2", "summary3", "topic1", "topic2", "topic3"]:
        df[col] = df[col].fillna("").astype(str)
    for i in (1, 2, 3):
        df[f"topic_norm_{i}"] = df[f"topic{i}"].apply(topic_normalize)
    return df


@st.cache_data
def load_model_manifest() -> list[dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not isinstance(manifest, list):
        raise ValueError("model_manifest.json harus berbentuk array")
    return manifest


@st.cache_data
def load_validation_scores() -> pd.DataFrame:
    df = pd.read_csv(VALIDATION_PATH)
    return df


@st.cache_data
def load_test_summary() -> pd.DataFrame:
    return pd.read_csv(TEST_SUMMARY_PATH)


def access_token() -> str | None:
    # Do not require a secrets file for public model repos.
    try:
        v = st.secrets.get("HF_TOKEN")
        if v:
            return str(v)
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("HF_TOKEN") or None


@st.cache_resource(show_spinner=False, max_entries=1)
def load_model(repo_id: str, token: str | None):
    """Caches only one model to reduce server memory usage."""
    tok = AutoTokenizer.from_pretrained(repo_id, token=token)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo_id, token=token)
    model.eval().to("cpu")
    return tok, model


def generate(dialogue: str, repo_id: str, token: str | None) -> dict:
    start = time.perf_counter()
    tokenizer, model = load_model(repo_id, token)
    input_text = "summarize: " + dialogue
    inputs_full = tokenizer(input_text, truncation=False, add_special_tokens=True)
    original_tokens = len(inputs_full["input_ids"])
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_INPUT, truncation=True)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_length=MAX_OUTPUT, num_beams=1, do_sample=False
        )
    summary = tokenizer.decode(generated[0], skip_special_tokens=True)
    out_tokens = len(tokenizer(summary, add_special_tokens=True)["input_ids"])
    elapsed = time.perf_counter() - start
    return {
        "summary": summary,
        "elapsed": elapsed,
        "input_tokens": original_tokens,
        "output_tokens": out_tokens,
        "truncated": original_tokens > MAX_INPUT,
    }


def rouge_for_dialog(prediction: str, refs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replicates the Kaggle evaluation: ROUGE(3 refs) -> macro average P, R, F1 independently."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
    rows = []
    for index, reference in enumerate(refs, 1):
        reference = str(reference)
        if not reference.strip():
            continue
        scores = scorer.score(reference, prediction)
        for label, key in ROUGE_KEYS.items():
            s = scores[key]
            rows.append({
                "Referensi": f"Referensi {index}", "Metrik": label,
                "Precision": float(s.precision), "Recall": float(s.recall), "F1": float(s.fmeasure),
            })
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    detail = pd.DataFrame(rows)
    summary = (detail.groupby("Metrik", as_index=False)[["Precision", "Recall", "F1"]]
               .mean())
    summary["Metrik"] = pd.Categorical(summary["Metrik"], categories=list(METRICS), ordered=True)
    summary = summary.sort_values("Metrik").reset_index(drop=True)
    return summary, detail


def bubble_markup(dialogue: str) -> str:
    parts = []
    speaker_counter: dict[str, int] = {}
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^\s*(#?Person\d+#?|[^:]{1,45}):\s*(.*)$", line)
        if match:
            speaker, message = match.groups()
        else:
            speaker, message = "Dialog", line
        if speaker not in speaker_counter:
            speaker_counter[speaker] = len(speaker_counter)
        direction = "left" if speaker_counter[speaker] % 2 == 0 else "right"
        parts.append(
            f'<div class="bubble-row-{direction}"><div class="bubble bubble-{direction}">'
            f'<div class="bubble-name">{safe(speaker)}</div>{safe(message)}</div></div>'
        )
    return '<div class="speaker-area">' + "".join(parts) + "</div>"


def scenarios_for(manifest: list[dict], model_name: str) -> list[dict]:
    return sorted([x for x in manifest if x["model"] == model_name], key=lambda x: x["scenario_id"])


def scenario_label(item: dict) -> str:
    lr = f'{float(item["learning_rate"]):.0e}'.replace("+", "")
    mark = "★ Terbaik · " if item["is_best_validation"] else ("Baseline · " if item["is_baseline"] else "")
    status = "✅ Model terhubung" if item.get("hf_repo_id") else "— Belum terhubung"
    return f'{item["scenario_id"]} · {mark}LR {lr} · BS {item["batch_size"]} · {status}'


def scenario_score(item: dict, validation: pd.DataFrame, metric: str) -> float | None:
    tmp = validation[
        (validation["model"] == item["model"])
        & (validation["lr"] == round(float(item["learning_rate"]) / 1e-5))
        & (validation["bs"] == int(item["batch_size"]))
    ]
    return float(tmp.iloc[0][metric]) if len(tmp) == 1 else None


def result_card(item: dict, output: dict, dialog_words: int):
    label = safe(item["model"])
    lr = f'{float(item["learning_rate"]):.0e}'.replace("+", "")
    top_left, top_right = st.columns([3, 1])
    with top_left:
        st.markdown(f"**{label}** · {item['scenario_id']}")
    with top_right:
        if item["is_best_validation"]:
            st.markdown('<span class="tag">Terbaik</span>', unsafe_allow_html=True)
    st.caption(f"Learning rate {lr} · Batch size {item['batch_size']}")
    st.markdown(f'<div class="summ-text">{safe(output["summary"])}</div>', unsafe_allow_html=True)
    n_summary = count_words(output["summary"])
    ratio = 100.0 * (1.0 - n_summary / dialog_words) if dialog_words else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Panjang dialog", f"{dialog_words} kata")
    c2.metric("Panjang ringkasan", f"{n_summary} kata")
    c3.metric("Pemendekan", f"{ratio:.1f}%")
    if output["truncated"]:
        st.warning("Dialog melebihi 512 token T5 dan dipotong saat inferensi.")


def rouge_card(model_name: str, output: dict, references: list[str]):
    main, detail = rouge_for_dialog(output["summary"], references)
    if main.empty:
        st.info("Referensi kosong sehingga ROUGE tidak dapat dihitung.")
        return
    st.markdown(f"**{model_name} — rata-rata 3 referensi**")
    by_name = main.set_index("Metrik")
    blocks = []
    for metric in METRICS:
        x = by_name.loc[metric]
        blocks.append(
            '<div class="metric-card">'
            f'<div class="metric-title">{metric}</div>'
            f'<div class="metric-f1">{format_score(x["F1"])}</div>'
            '<div class="metric-sub"><span>F1-score</span><span>rata-rata</span></div>'
            f'<div class="metric-sub"><span>P {format_score(x["Precision"])}</span>'
            f'<span>R {format_score(x["Recall"])}</span></div>'
            '</div>'
        )
    st.markdown('<div class="metric-grid">' + ''.join(blocks) + '</div>', unsafe_allow_html=True)
    with st.expander("Lihat skor per referensi (Precision, Recall, F1)"):
        view = detail.copy()
        for col in ("Precision", "Recall", "F1"):
            view[col] = view[col].map(lambda v: f"{v:.4f}")
        st.dataframe(view, hide_index=True, use_container_width=True)
    st.caption("Nilai di atas adalah skor satu dialog, bukan rata-rata seluruh 500 data uji.")


# App data + manifest
try:
    manifest = load_model_manifest()
except Exception as exc:
    st.error(f"Gagal membaca file konfigurasi: {exc}")
    st.stop()

st.markdown(
    '<div class="dialog-header"><h2>💬 Peringkasan Dialog Abstraktif</h2>'
    '<p>Model: T5-small &amp; T5-base · Dataset: DialogSum · Evaluasi: ROUGE-1, ROUGE-2, ROUGE-L</p></div>',
    unsafe_allow_html=True,
)

if "result_state" not in st.session_state:
    st.session_state.result_state = None

# Compact controls, as in user's approved dataset/manual mockups.
with st.container(border=True):
    source_col, mode_col, scenario_col = st.columns([1.2, 1.05, 1.5], gap="medium")
    with source_col:
        st.markdown('<div class="small-label">Sumber dialog</div>', unsafe_allow_html=True)
        source = st.segmented_control(
            "Sumber", ["Dataset Uji", "Input Manual"], default="Dataset Uji",
            label_visibility="collapsed", key="source_control",
        )
        df = None
        row = None
        refs = None
        if source == "Dataset Uji":
            try:
                df = load_test()
            except Exception as exc:
                st.error("Data uji gagal dimuat. Periksa internet atau letakkan dialogsum.test.jsonl di folder data/.")
                with st.expander("Pesan teknis"):
                    st.code(str(exc))
                st.stop()
            # Count unique fname per topic, rather than count of the annotations.
            topic_pairs = pd.concat(
                [df[["fname", f"topic_norm_{i}"]].rename(columns={f"topic_norm_{i}": "topic"})
                 for i in (1, 2, 3)], ignore_index=True
            ).drop_duplicates()
            topic_counts = topic_pairs[topic_pairs["topic"] != ""].groupby("topic")["fname"].nunique().sort_values(ascending=False)
            topic_choices = ["Semua topik"] + list(topic_counts.index)
            chosen_topic = st.selectbox(
                "Topik", topic_choices,
                format_func=lambda t: ("Semua topik" if t == "Semua topik"
                                       else f"{t} ({topic_counts[t]} dialog)"),
                key="topic_control",
            )
            if chosen_topic == "Semua topik":
                filtered = df
            else:
                matching_ids = topic_pairs.loc[topic_pairs["topic"] == chosen_topic, "fname"].unique()
                filtered = df[df["fname"].isin(matching_ids)]
            if filtered.empty:
                st.warning("Tidak ada dialog untuk topik ini.")
                st.stop()
            available_ids = filtered["fname"].tolist()
            chosen_fname = st.selectbox(
                "Dialog", available_ids,
                format_func=lambda fid: (
                    f'{fid} — {filtered.loc[filtered["fname"] == fid, "dialogue"].iloc[0].splitlines()[0][:48]}'
                ), key="dialog_control",
            )
            row = df.loc[df["fname"] == chosen_fname].iloc[0]
            text_input = str(row["dialogue"])
            refs = [str(row[f"summary{i}"]) for i in (1, 2, 3)]
        else:
            text_input = ""
    with mode_col:
        st.markdown('<div class="small-label">Mode model</div>', unsafe_allow_html=True)
        model_mode = st.radio(
            "Model", ["T5-small saja", "T5-base saja", "Keduanya"],
            index=2, label_visibility="collapsed", key="model_mode",
        )
    with scenario_col:
        st.markdown('<div class="small-label">Skenario (LR–BS) tiap model</div>', unsafe_allow_html=True)
        choices = {}
        for model_name in ("T5-small", "T5-base"):
            scenario_options = scenarios_for(manifest, model_name)
            if not scenario_options:
                st.error(f"Tidak ada skenario untuk {model_name}")
                st.stop()
            preferred = next((i for i, x in enumerate(scenario_options) if x["is_best_validation"]), 0)
            ids = [x["scenario_id"] for x in scenario_options]
            chosen = st.selectbox(
                model_name, ids, index=preferred,
                format_func=lambda sid, opts=scenario_options: scenario_label(
                    next(x for x in opts if x["scenario_id"] == sid)),
                key=f"select_{model_name}",
                disabled=(model_mode == "T5-small saja" and model_name == "T5-base") or
                         (model_mode == "T5-base saja" and model_name == "T5-small"),
            )
            choices[model_name] = next(x for x in scenario_options if x["scenario_id"] == chosen)

    selected_models = (["T5-small"] if model_mode == "T5-small saja" else
                       ["T5-base"] if model_mode == "T5-base saja" else
                       ["T5-small", "T5-base"])

if source == "Input Manual":
    with st.container(border=True):
        st.markdown('<div class="section-title">Input dialog</div>', unsafe_allow_html=True)
        text_input = st.text_area(
            "Masukkan dialog", height=170,
            placeholder="#Person1#: Are we meeting tomorrow?\n#Person2#: Yes, at 10 AM.",
            label_visibility="collapsed", key="manual_dialog",
        )
        with st.expander("Pratinjau percakapan"):
            if text_input.strip():
                st.markdown(bubble_markup(text_input), unsafe_allow_html=True)

# Match the evaluation notebook: official test dialogues are passed through unchanged.
# Manual input gets only whitespace normalization like the training notebook.
input_clean = text_input if source == "Dataset Uji" else cleaned(text_input)
input_word_count = count_words(text_input)
chosen_scenarios = [choices[m] for m in selected_models]
result_key = (
    source, input_clean, tuple((m["scenario_id"], m.get("hf_repo_id")) for m in chosen_scenarios)
)
connected = all(bool(m.get("hf_repo_id")) for m in chosen_scenarios)

with st.container(border=True):
    action_left, action_right = st.columns([4, 1])
    with action_left:
        if not connected:
            not_connected = [m["scenario_id"] for m in chosen_scenarios if not m.get("hf_repo_id")]
            st.warning("Checkpoint belum dihubungkan: " + ", ".join(not_connected) + ".")
    with action_right:
        pressed = st.button(
            "Ringkas", type="primary", use_container_width=True,
            disabled=not input_clean or not connected,
        )

if pressed:
    try:
        token = access_token()
        all_outputs = {}
        for m in chosen_scenarios:
            with st.spinner(f"Memuat dan menjalankan {m['model']} ({m['scenario_id']})..."):
                all_outputs[m["model"]] = generate(input_clean, m["hf_repo_id"], token)
        st.session_state.result_state = {"key": result_key, "outputs": all_outputs}
    except Exception as exc:
        st.session_state.result_state = None
        st.error("Inferensi gagal. Pastikan repo Hugging Face benar, token memiliki izin baca, dan server memiliki cukup RAM.")
        with st.expander("Detail error"):
            st.code(str(exc))

if source == "Dataset Uji":
    with st.container(border=True):
        st.markdown('<div class="section-title">Input dialog</div>', unsafe_allow_html=True)
        preview_col, info_col = st.columns([2, 1], gap="medium")
        with preview_col:
            st.markdown(bubble_markup(text_input), unsafe_allow_html=True)
        with info_col:
            st.metric("Panjang dialog", f"{input_word_count} kata")
            st.markdown("**Anotasi topik**")
            st.caption(" · ".join([str(row[f"topic{i}"]) for i in (1, 2, 3)]))
            st.caption("Satu dialog dapat ditemukan melalui tiga anotasi topiknya.")

state = st.session_state.result_state
if state and state.get("key") == result_key:
    output_dict = state["outputs"]
    with st.container(border=True):
        st.markdown('<div class="section-title">Hasil ringkasan</div>', unsafe_allow_html=True)
        columns = st.columns(len(selected_models), gap="medium")
        for col, model_name in zip(columns, selected_models):
            with col:
                with st.container(border=True):
                    result_card(choices[model_name], output_dict[model_name], input_word_count)

    if refs is not None:
        with st.expander("Lihat ringkasan referensi (3)"):
            for i, ref in enumerate(refs, start=1):
                st.markdown(f"**Referensi {i}** · Topik: {row[f'topic{i}']}")
                st.write(ref)
        with st.container(border=True):
            st.markdown('<div class="section-title">Evaluasi ROUGE</div>', unsafe_allow_html=True)
            st.caption("Skor per dialog dihitung terhadap tiga ringkasan referensi, kemudian P, R, dan F1 dirata-rata secara terpisah.")
            for index, model_name in enumerate(selected_models):
                if index:
                    st.divider()
                rouge_card(model_name, output_dict[model_name], refs)
elif state is not None:
    st.info("Pilihan dialog/model telah berubah. Klik **Ringkas** untuk menghitung hasil yang baru.")


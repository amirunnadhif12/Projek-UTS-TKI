# =====================================================
#  MINI SEARCH ENGINE — Streamlit
#  Temu Kembali Informasi | Projek UTS
# =====================================================
#  Fitur:
#   • Pre-processing (case folding, punctuation removal,
#     stop-word removal, stemming Sastrawi)
#   • Inverted Index
#   • TF-IDF (log frequency weighting)
#   • Vector Space Model + Cosine Similarity
#   • Ranked Retrieval
#   • Evaluasi (Precision, Recall, F1)
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
import math
from pathlib import Path
from collections import Counter

# ---- Custom modules ----
from modules.preprocessing import preprocess, preprocess_steps
from modules.indexing import build_index
from modules.retrieval import build_tfidf, search
from modules.evaluation import evaluate

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Mini Search Engine — TF-IDF & Cosine Similarity",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---- Load external CSS ----
CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if CSS_PATH.exists():
    st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---- Google Font ----
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# =====================================================
# HEADER BANNER
# =====================================================
st.markdown(
    """
    <div class="header-banner">
        <h1> Mini Search Engine</h1>
        <p>Mesin pencari dokumen opini berbasis TF-IDF &amp; Vector Space Model</p>
        <span class="badge">Cosine Similarity · Log Frequency Weighting · Sastrawi Stemmer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =====================================================
# LOAD DATASET
# =====================================================
DEFAULT_DATASET = Path(__file__).parent / "dataset_opini.csv"


@st.cache_data(show_spinner=False)
def load_dataset(file_or_path):
    """Load dataset CSV, handles header quirks."""
    if hasattr(file_or_path, "seek"):
        file_or_path.seek(0)

    try:
        df = pd.read_csv(file_or_path, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        df = pd.read_csv(file_or_path, encoding="utf-8", header=2, on_bad_lines="skip")

    # Auto-detect if real header is on row 2
    if "Data" not in df.columns and "Unnamed: 2" in df.columns:
        first_value = str(df["Unnamed: 2"].iloc[0]).strip()
        if first_value == "Data":
            if hasattr(file_or_path, "seek"):
                file_or_path.seek(0)
            df = pd.read_csv(file_or_path, encoding="utf-8", header=2, on_bad_lines="skip")

    return df


uploaded = st.file_uploader("📂 Upload Dataset CSV (opsional)", type=["csv"])

if uploaded:
    df = load_dataset(uploaded)
    st.success("✅ Dataset berhasil di-upload!")
else:
    if DEFAULT_DATASET.exists():
        df = load_dataset(str(DEFAULT_DATASET))
        st.info("📁 Menggunakan dataset default: **dataset_opini.csv**")
    else:
        st.error("❌ Dataset default tidak ditemukan. Silakan upload file CSV.")
        st.stop()

# ---- Determine text column ----
if "Data" in df.columns:
    TEXT_COL = "Data"
elif "Unnamed: 2" in df.columns:
    TEXT_COL = "Unnamed: 2"
else:
    st.error("❌ Kolom teks tidak ditemukan. Pastikan CSV memiliki kolom **'Data'**.")
    st.stop()

documents = df[TEXT_COL].fillna("").astype(str).tolist()

# =====================================================
# PREPROCESSING & INDEXING (cached)
# =====================================================
@st.cache_data(show_spinner="⏳ Memproses dokumen…")
def process_all(docs):
    """Preprocess all documents, build index & TF-IDF."""
    processed = [preprocess(doc) for doc in docs]
    inv_index = build_index(processed)
    doc_matrix, vocab, idf_dict, tf_list = build_tfidf(processed, inv_index)
    return processed, inv_index, doc_matrix, vocab, idf_dict, tf_list


processed_docs, inverted_index, doc_matrix, vocabulary, idf_dict, tf_list = process_all(
    tuple(documents)
)

# ---- Stats ----
total_docs = len(documents)
total_terms = len(vocabulary)
avg_doc_len = round(np.mean([len(d) for d in processed_docs]), 1) if processed_docs else 0

# ---- Statistics Cards ----
st.markdown("---")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-icon">📄</div>
            <div class="stat-value">{total_docs}</div>
            <div class="stat-label">Total Dokumen</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-icon">🔤</div>
            <div class="stat-value">{total_terms}</div>
            <div class="stat-label">Term Unik</div>
        </div>""",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""<div class="stat-card">
            <div class="stat-icon">📏</div>
            <div class="stat-value">{avg_doc_len}</div>
            <div class="stat-label">Rata-rata Token/Dokumen</div>
        </div>""",
        unsafe_allow_html=True,
    )
st.markdown("---")

# =====================================================
# TABS
# =====================================================
tab_search, tab_tfidf, tab_dataset, tab_eval = st.tabs(
    ["🔍 Pencarian", "📊 TF-IDF Detail", "📚 Dataset & Index", "📈 Evaluasi"]
)

# =====================================================
# TAB 1: PENCARIAN
# =====================================================
with tab_search:
    st.markdown(
        '<div class="section-title"><span class="icon">🔍</span> Cari Dokumen</div>',
        unsafe_allow_html=True,
    )
    st.caption("Masukkan kata kunci untuk mencari dokumen yang relevan. Sistem menggunakan **TF-IDF + Cosine Similarity** untuk ranking.")

    with st.form("search_form", clear_on_submit=False):
        query_input = st.text_input(
            "Kueri Pencarian",
            placeholder="Contoh: imunisasi campak balita",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("🔎  Cari Sekarang", use_container_width=True)

    if submitted and query_input.strip():
        query_tokens = preprocess(query_input)

        if not query_tokens:
            st.warning("⚠️ Tidak ada term yang tersisa setelah preprocessing. Coba kata kunci lain.")
        else:
            # Show preprocessed query
            st.markdown(
                f"**Query setelah preprocessing:** `{' '.join(query_tokens)}`"
            )

            similarities, q_vector = search(query_tokens, vocabulary, idf_dict, doc_matrix)

            # Build results
            results_df = pd.DataFrame({
                "doc_id": range(len(documents)),
                "document": documents,
                "similarity": similarities,
            })
            results_df = results_df[results_df["similarity"] > 0].sort_values(
                "similarity", ascending=False
            ).reset_index(drop=True)

            if results_df.empty:
                st.info("😕 Tidak ada dokumen yang cocok dengan kueri Anda.")
            else:
                st.markdown(f"### 📋 Hasil Pencarian — {len(results_df)} dokumen ditemukan")

                # Store in session state for evaluation tab
                st.session_state["last_results"] = results_df
                st.session_state["last_query"] = query_input

                top_n = min(10, len(results_df))

                for rank_idx in range(top_n):
                    row = results_df.iloc[rank_idx]
                    score = row["similarity"]
                    doc_id = int(row["doc_id"])
                    doc_text = row["document"]

                    # Determine level
                    if score >= 0.3:
                        level = "high"
                        badge_class = "gold" if rank_idx == 0 else "silver" if rank_idx == 1 else "bronze"
                        level_label = "Sangat Relevan"
                    elif score >= 0.1:
                        level = "medium"
                        badge_class = "bronze"
                        level_label = "Relevan"
                    else:
                        level = "low"
                        badge_class = "bronze"
                        level_label = "Kurang Relevan"

                    # Truncate long docs
                    display_text = doc_text[:300] + "…" if len(doc_text) > 300 else doc_text

                    score_pct = min(score * 100, 100)

                    st.markdown(
                        f"""
                        <div class="result-card {level} animate-in" style="animation-delay: {rank_idx * 0.05}s">
                            <div style="display:flex; align-items:flex-start;">
                                <div class="rank-badge {badge_class}">#{rank_idx + 1}</div>
                                <div style="flex:1;">
                                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                                        <span style="font-weight:600; color:#e2e8f0;">Dokumen #{doc_id + 1}</span>
                                        <span class="step-pill">{level_label} · {score:.4f}</span>
                                    </div>
                                    <p style="color:rgba(255,255,255,0.7); font-size:0.9rem; line-height:1.6; margin:0;">
                                        {display_text}
                                    </p>
                                    <div class="score-bar">
                                        <div class="score-bar-fill {level}" style="width:{score_pct:.1f}%"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Show full results table
                with st.expander("📊 Lihat Tabel Lengkap Semua Hasil"):
                    display_df = results_df[["doc_id", "similarity", "document"]].copy()
                    display_df.columns = ["ID Dokumen", "Cosine Similarity", "Teks Dokumen"]
                    display_df["ID Dokumen"] = display_df["ID Dokumen"] + 1
                    st.dataframe(display_df, use_container_width=True)

    elif submitted:
        st.warning("⚠️ Masukkan kata kunci terlebih dahulu.")


# =====================================================
# TAB 2: TF-IDF DETAIL
# =====================================================
with tab_tfidf:
    st.markdown(
        '<div class="section-title"><span class="icon">📊</span> Detail TF-IDF</div>',
        unsafe_allow_html=True,
    )
    st.caption("Lihat bobot TF, IDF, dan TF-IDF untuk setiap term dalam koleksi dokumen.")

    # ---- IDF Table ----
    st.markdown("#### 📐 Tabel IDF (Inverse Document Frequency)")
    idf_data = sorted(idf_dict.items(), key=lambda x: x[1], reverse=True)
    idf_df = pd.DataFrame(idf_data, columns=["Term", "IDF"])
    idf_df["DF (doc freq)"] = idf_df["Term"].apply(lambda t: len(inverted_index.get(t, [])))
    idf_df = idf_df[["Term", "DF (doc freq)", "IDF"]]
    idf_df.index = range(1, len(idf_df) + 1)

    col_idf1, col_idf2 = st.columns([2, 1])
    with col_idf1:
        st.dataframe(idf_df.head(50), use_container_width=True, height=400)
    with col_idf2:
        st.markdown("**Rumus IDF:**")
        st.latex(r"IDF(t) = \log_{10}\left(\frac{N}{df(t)}\right)")
        st.markdown(f"- **N** (total dokumen) = `{total_docs}`")
        st.markdown(f"- **Total term unik** = `{total_terms}`")
        st.markdown("- Term dengan IDF tinggi → jarang muncul di banyak dokumen → lebih diskriminatif")

    # ---- TF-IDF per Document ----
    st.markdown("---")
    st.markdown("#### 📄 TF-IDF per Dokumen")
    doc_selector = st.selectbox(
        "Pilih dokumen untuk melihat detail TF-IDF:",
        options=range(total_docs),
        format_func=lambda x: f"Dokumen #{x + 1}: {documents[x][:80]}…" if len(documents[x]) > 80 else f"Dokumen #{x + 1}: {documents[x]}",
    )

    if doc_selector is not None:
        tf_doc = tf_list[doc_selector]
        detail_rows = []
        for term, tf_val in sorted(tf_doc.items(), key=lambda x: x[1], reverse=True):
            idf_val = idf_dict.get(term, 0)
            tfidf_val = tf_val * idf_val
            raw_count = Counter(processed_docs[doc_selector]).get(term, 0)
            detail_rows.append({
                "Term": term,
                "Raw TF": raw_count,
                "TF (1+log₁₀)": round(tf_val, 4),
                "IDF": round(idf_val, 4),
                "TF-IDF": round(tfidf_val, 4),
            })

        detail_df = pd.DataFrame(detail_rows)
        detail_df.index = range(1, len(detail_df) + 1)
        st.dataframe(detail_df, use_container_width=True, height=400)

        # Show formula
        st.markdown("**Rumus:**")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.latex(r"TF(t,d) = 1 + \log_{10}(f_{t,d})")
        with col_f2:
            st.latex(r"W(t,d) = TF(t,d) \times IDF(t)")

    # ---- Top Terms Bar Chart ----
    st.markdown("---")
    st.markdown("#### 📊 Top 20 Term (berdasarkan IDF)")
    top_terms = idf_df.head(20)
    st.bar_chart(top_terms.set_index("Term")["IDF"], use_container_width=True, height=350)


# =====================================================
# TAB 3: DATASET & INDEX
# =====================================================
with tab_dataset:
    st.markdown(
        '<div class="section-title"><span class="icon">📚</span> Dataset &amp; Inverted Index</div>',
        unsafe_allow_html=True,
    )

    # ---- Dataset Preview ----
    st.markdown("#### 📋 Preview Dataset")
    st.dataframe(df.head(20), use_container_width=True, height=400)

    st.markdown("---")

    # ---- Preprocessing Demo ----
    st.markdown("#### 🔬 Demo Preprocessing")
    st.caption("Pilih dokumen untuk melihat tahapan preprocessing secara detail.")
    demo_doc = st.selectbox(
        "Pilih dokumen:",
        options=range(total_docs),
        format_func=lambda x: f"Doc #{x + 1}: {documents[x][:70]}…" if len(documents[x]) > 70 else f"Doc #{x + 1}: {documents[x]}",
        key="demo_doc_select",
    )

    if demo_doc is not None:
        steps = preprocess_steps(documents[demo_doc])

        step_labels = [
            ("1️⃣ Original", steps["original"]),
            ("2️⃣ Case Folding", steps["case_folded"]),
            ("3️⃣ Punctuation Removal", steps["punctuation_removed"]),
            ("4️⃣ Tokenisasi", ", ".join(steps["tokens"])),
            ("5️⃣ Stop-word Removal", ", ".join(steps["stopword_removed"])),
            ("6️⃣ Stemming (Sastrawi)", ", ".join(steps["stemmed"])),
        ]

        for label, content in step_labels:
            with st.expander(label, expanded=False):
                st.text(content)

    st.markdown("---")

    # ---- Inverted Index ----
    st.markdown("#### 🗂️ Inverted Index")
    st.caption(f"Total **{total_terms}** term unik di-index dari **{total_docs}** dokumen.")

    search_term = st.text_input("🔍 Cari term di index:", placeholder="Ketik term…", key="idx_search")

    if search_term:
        filtered = {k: v for k, v in inverted_index.items() if search_term.lower() in k.lower()}
    else:
        filtered = dict(list(sorted(inverted_index.items()))[:30])

    if filtered:
        idx_rows = []
        for term, doc_ids in sorted(filtered.items()):
            idx_rows.append({
                "Term": term,
                "DF": len(doc_ids),
                "Document IDs": ", ".join(str(d + 1) for d in doc_ids[:20]) + ("…" if len(doc_ids) > 20 else ""),
            })
        idx_df = pd.DataFrame(idx_rows)
        idx_df.index = range(1, len(idx_df) + 1)
        st.dataframe(idx_df, use_container_width=True, height=400)
    else:
        st.info("Tidak ditemukan term yang cocok.")


# =====================================================
# TAB 4: EVALUASI
# =====================================================
with tab_eval:
    st.markdown(
        '<div class="section-title"><span class="icon">📈</span> Evaluasi Sistem</div>',
        unsafe_allow_html=True,
    )
    st.caption("Hitung **Precision**, **Recall**, dan **F1-Score** dengan membandingkan hasil retrieval terhadap ground truth.")

    if "last_results" in st.session_state and st.session_state["last_results"] is not None:
        last_q = st.session_state.get("last_query", "")
        results_df = st.session_state["last_results"]
        predicted_ids = results_df["doc_id"].head(10).tolist()

        st.markdown(f"**Kueri terakhir:** `{last_q}`")
        st.markdown(f"**Top-10 dokumen retrieved:** `{[int(x)+1 for x in predicted_ids]}`")

        st.markdown("---")
        st.markdown("#### Masukkan Ground Truth")
        st.caption("Tulis nomor dokumen yang benar-benar relevan (1-indexed), pisahkan dengan koma.")

        gt_input = st.text_input(
            "Ground Truth (nomor dokumen)",
            placeholder="Contoh: 1, 4, 10, 14",
            key="gt_input",
        )

        if gt_input:
            try:
                # Parse ground truth (user inputs 1-indexed, convert to 0-indexed)
                gt_ids_1indexed = [int(x.strip()) for x in gt_input.split(",") if x.strip().isdigit()]
                gt_ids = [x - 1 for x in gt_ids_1indexed if 1 <= x <= total_docs]

                if not gt_ids:
                    st.warning("⚠️ Masukkan nomor dokumen yang valid (1 s/d jumlah dokumen).")
                else:
                    metrics = evaluate(predicted_ids, gt_ids)

                    # Metric cards
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(
                            f"""<div class="metric-card metric-precision">
                                <div class="label">Precision</div>
                                <div class="value">{metrics['precision']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">TP / (TP + FP) = {metrics['tp']} / {metrics['tp'] + metrics['fp']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with m2:
                        st.markdown(
                            f"""<div class="metric-card metric-recall">
                                <div class="label">Recall</div>
                                <div class="value">{metrics['recall']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">TP / (TP + FN) = {metrics['tp']} / {metrics['tp'] + metrics['fn']}</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                    with m3:
                        st.markdown(
                            f"""<div class="metric-card metric-f1">
                                <div class="label">F1-Score</div>
                                <div class="value">{metrics['f1']}</div>
                                <div style="font-size:0.8rem; opacity:0.6;">2 × P × R / (P + R)</div>
                            </div>""",
                            unsafe_allow_html=True,
                        )

                    # Detail
                    st.markdown("---")
                    st.markdown("#### 🔍 Detail Evaluasi")
                    detail_col1, detail_col2 = st.columns(2)
                    with detail_col1:
                        st.markdown(f"- **True Positives (TP):** `{metrics['tp']}` — dokumen relevan yang berhasil ditemukan")
                        st.markdown(f"- **False Positives (FP):** `{metrics['fp']}` — dokumen tidak relevan yang masuk top-10")
                        st.markdown(f"- **False Negatives (FN):** `{metrics['fn']}` — dokumen relevan yang tidak masuk top-10")
                    with detail_col2:
                        st.markdown("**Rumus:**")
                        st.latex(r"Precision = \frac{TP}{TP + FP}")
                        st.latex(r"Recall = \frac{TP}{TP + FN}")
                        st.latex(r"F_1 = \frac{2 \cdot P \cdot R}{P + R}")

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("💡 Lakukan pencarian terlebih dahulu di tab **🔍 Pencarian**, lalu kembali ke sini untuk evaluasi.")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; padding:20px 0 10px; color:rgba(255,255,255,0.3); font-size:0.8rem;">
        Mini Search Engine — Projek UTS Temu Kembali Informasi<br>
        TF-IDF · Vector Space Model · Cosine Similarity · Sastrawi Stemmer
    </div>
    """,
    unsafe_allow_html=True,
)

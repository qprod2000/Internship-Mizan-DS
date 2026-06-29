# -*- coding: utf-8 -*-
"""
RFM Segmentation & Donor Ladder Intelligence — Dashboard Interaktif
Mizan Amanah | Kelompok 1

Pipeline analitik pada file ini meniru 1:1 logika pada notebook final
RFM_Mizan_Amanah_revisi.ipynb (section 1-10):
  1) Load data
  2) Data cleaning (dedup, hapus nominal<=0, normalisasi akad, dst.)
  3) Feature engineering RFM
  4) Deteksi Reactivated Donor (gap > 90 hari) + klasifikasi 9 segmen Tangga Donatur
  5) Rekomendasi strategi engagement per donor
  6) Tren bulanan, distribusi program, pola donasi
  7) Segment mapping table & ringkasan segmen
  8) Dashboard distribusi donor (visualisasi)
  9) Distribusi Recency, Frequency, Monetary
 10) Export final
"""

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ----------------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="RFM Donor Ladder — Mizan Amanah",
    page_icon="🕌",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMERALD = "#0F6B5C"
EMERALD_DARK = "#0B4F44"
GOLD = "#C9A24B"
SEGMENT_ORDER = [
    "Premium Donor", "Loyal Donor", "Reactivated Donor", "Active Donor",
    "Potential Donor", "New Donor", "Small/Occasional Donor",
    "Dormant Donor", "Lost Donor",
]
SEGMENT_COLORS = {
    "Premium Donor": "#0B4F44", "Loyal Donor": "#0F6B5C", "Reactivated Donor": "#2E9E8A",
    "Active Donor": "#55B5A3", "Potential Donor": "#8FCFC2", "New Donor": "#C9A24B",
    "Small/Occasional Donor": "#E0C27A", "Dormant Donor": "#B0B8B6", "Lost Donor": "#6E7674",
}

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "data_set_donasi_ma_2020_2025.csv"

st.markdown(
    f"""
    <style>
    .stMetric {{ background-color: #EEF5F3; border-radius: 10px; padding: 10px 14px; }}
    div[data-testid="stMetricValue"] {{ color: {EMERALD_DARK}; }}
    h1, h2, h3 {{ color: {EMERALD_DARK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# 1-2) LOAD DATA
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Memuat & membersihkan data donasi...")
def load_and_clean(file_bytes_or_path):
    df = pd.read_csv(file_bytes_or_path)

    # --- Data Cleaning (identik dengan notebook revisi) ---
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df = df.dropna(subset=["donor_id", "nominal"])
    df = df[df["nominal"] > 0]
    df = df.drop_duplicates(subset=["transaksi_id"])
    df["program"] = df["program"].fillna("Unknown")

    akad_map = {"Zakat Harta": "Zakat Harta (Maal)", "Zakat Uang": "Zakat Harta (Maal)"}
    df["akad"] = df["akad"].replace(akad_map)
    return df


@st.cache_data(show_spinner="Menjalankan feature engineering RFM & klasifikasi 9 segmen...")
def build_rfm(df: pd.DataFrame):
    # --- 3) Feature Engineering RFM ---
    snapshot_date = df["tanggal"].max() + pd.Timedelta(days=1)

    agg = df.groupby("donor_id").agg(
        Recency=("tanggal", lambda x: (snapshot_date - x.max()).days),
        Frequency=("transaksi_id", "count"),
        Monetary=("nominal", "sum"),
        Avg_Nominal=("nominal", "mean"),
        First_Donation=("tanggal", "min"),
        Last_Donation=("tanggal", "max"),
    ).reset_index()
    agg["Duration_Days"] = (agg["Last_Donation"] - agg["First_Donation"]).dt.days

    dom_program = df.groupby("donor_id")["program"].agg(lambda x: x.value_counts().idxmax())
    dom_akad = df.groupby("donor_id")["akad"].agg(lambda x: x.value_counts().idxmax())
    agg = agg.merge(dom_program.rename("Dominan_Program"), on="donor_id")
    agg = agg.merge(dom_akad.rename("Dominan_Akad"), on="donor_id")

    df = df.copy()
    df["year"] = df["tanggal"].dt.year
    yearly = df.groupby(["donor_id", "year"])["nominal"].sum().reset_index()
    max_yearly = yearly.groupby("donor_id")["nominal"].max().rename("Max_Yearly_Monetary")
    agg = agg.merge(max_yearly, on="donor_id")

    max_tx = df.groupby("donor_id")["nominal"].max().rename("Max_Single_Transaction")
    agg = agg.merge(max_tx, on="donor_id")

    # --- 4) Deteksi Reactivated Donor + Klasifikasi 9 Segmen ---
    df_sorted = df.sort_values(["donor_id", "tanggal"])

    def has_reactivation_gap(dates):
        if len(dates) < 2:
            return False
        diffs = dates.sort_values().diff().dt.days.dropna()
        return (diffs > 90).any()

    gap_flag = df_sorted.groupby("donor_id")["tanggal"].apply(has_reactivation_gap).rename("Has_Gap_90")
    agg = agg.merge(gap_flag, on="donor_id")

    def classify(row):
        r, f, m, my, gap = row["Recency"], row["Frequency"], row["Monetary"], row["Max_Yearly_Monetary"], row["Has_Gap_90"]
        if gap and r <= 30 and f >= 2:
            return "Reactivated Donor"
        if r <= 60 and f >= 2 and my >= 10_000_000:
            return "Premium Donor"
        if r <= 30 and f >= 3 and m >= 1_000_000:
            return "Loyal Donor"
        if r <= 60 and f >= 3 and 500_000 <= m <= 5_000_000:
            return "Potential Donor"
        if r <= 60 and f >= 2 and m >= 200_000:
            return "Active Donor"
        if r <= 30 and f == 1:
            return "New Donor"
        if r <= 90 and f in (1, 2) and m < 200_000:
            return "Small/Occasional Donor"
        if r > 180 and f == 1 and m < 500_000:
            return "Lost Donor"
        if r > 90:
            return "Dormant Donor"
        return "Active Donor" if f >= 2 else "Small/Occasional Donor"  # fallback zona abu-abu

    agg["Segment"] = agg.apply(classify, axis=1)
    agg["High_Value_Donor"] = (agg["Max_Yearly_Monetary"] >= 10_000_000) | (agg["Max_Single_Transaction"] >= 5_000_000)

    # --- 5) Rekomendasi Strategi Engagement per Donor ---
    seg_to_reco = {
        "New Donor": "Kirim ucapan terima kasih + onboarding info program. Follow-up dalam 7-14 hari untuk donasi kedua.",
        "Small/Occasional Donor": "Edukasi ringan via konten emosional (kisah mustahik). Ajak ikut campaign musiman (Ramadhan/Qurban).",
        "Active Donor": "Reminder rutin bulanan + edukasi manfaat donasi konsisten. Tawarkan program autodebet/langganan.",
        "Potential Donor": "Tawarkan program donasi rutin bulanan (subscription). Berikan laporan impact personal.",
        "Loyal Donor": "Kirim laporan dampak donasi (impact report) personal & komunikasi rutin. Pertahankan engagement.",
        "Premium Donor": "Pendekatan personal/relationship manager khusus. Undang ke event eksklusif & laporan dampak detail.",
        "Reactivated Donor": "Sambutan hangat (welcome back) + apresiasi. Pastikan pengalaman donasi mulus agar tidak vakum lagi.",
        "Dormant Donor": "Kampanye reaktivasi (win-back) bertarget: pengingat, cerita dampak terbaru, insentif kecil.",
        "Lost Donor": "Campaign storytelling & pengingat ringan (low cost). Evaluasi apakah layak diprioritaskan kembali.",
    }
    agg["Rekomendasi_Strategi"] = agg["Segment"].map(seg_to_reco)

    # --- 6) Pola Donasi ---
    def pattern_change(g):
        if len(g) < 2:
            return "Data Tidak Cukup"
        g = g.sort_values("tanggal")
        mid = len(g) // 2
        early_avg, late_avg = g.iloc[:mid]["nominal"].mean(), g.iloc[mid:]["nominal"].mean()
        if late_avg > early_avg * 1.1:
            return "Meningkat"
        elif late_avg < early_avg * 0.9:
            return "Menurun"
        return "Stabil"

    pattern = df_sorted.groupby("donor_id").apply(pattern_change, include_groups=False).rename("Pola_Donasi")
    agg = agg.merge(pattern, on="donor_id")

    # --- 7) Segment Mapping Table (referensi kriteria) ---
    segment_mapping = pd.DataFrame([
        ["New Donor", "≤30 hari", "1x", "Bebas", "Donatur baru. Fokus follow-up awal."],
        ["Small/Occasional Donor", "≤90 hari", "1-2x", "<Rp200rb", "Donatur momen tertentu. Pendekatan emosional ringan."],
        ["Active Donor", "≤60 hari", "≥2x", "≥Rp200rb", "Mulai rutin. Edukasi & reminder."],
        ["Potential Donor", "≤60 hari", "≥3x", "Rp500rb-5jt", "Konsisten, potensi naik kelas. Tawarkan program rutin."],
        ["Loyal Donor", "≤30 hari", "≥3x", "≥Rp1jt", "Stabil & rutin. Impact report personal."],
        ["Premium Donor", "≤60 hari", "≥2x", "≥Rp10jt/tahun", "Bernilai tinggi. Relasi personal eksklusif."],
        ["Reactivated Donor", "≤30 hari", "≥1x stlh vakum >90 hari", "Tidak wajib tinggi", "Kembali aktif. Sambut hangat."],
        ["Dormant Donor", ">90 hari", "≥1x", "Bebas", "Lama tidak aktif. Target reaktivasi."],
        ["Lost Donor", ">180 hari", "1x", "<Rp500rb", "Sekali donasi & hilang. Storytelling pengingat."],
    ], columns=["Segment", "Recency", "Frequency", "Monetary", "Strategi"])

    return df, agg, segment_mapping, seg_to_reco, snapshot_date


@st.cache_data(show_spinner=False)
def compute_trend_and_program(df: pd.DataFrame):
    # --- 6) Tren bulanan & distribusi program ---
    d = df.copy()
    d["year_month"] = d["tanggal"].dt.to_period("M").astype(str)
    monthly_trend = d.groupby("year_month").agg(
        Total_Nominal=("nominal", "sum"),
        Jumlah_Transaksi=("transaksi_id", "count"),
        Donor_Unik=("donor_id", "nunique"),
    ).reset_index()

    program_dist = d.groupby("program").agg(
        Total_Nominal=("nominal", "sum"),
        Jumlah_Transaksi=("transaksi_id", "count"),
    ).reset_index().sort_values("Total_Nominal", ascending=False)
    program_dist["Persentase_Nominal"] = (
        program_dist["Total_Nominal"] / program_dist["Total_Nominal"].sum() * 100
    ).round(2)
    return monthly_trend, program_dist


def rupiah(x):
    try:
        return f"Rp{x:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return x


def to_excel_bytes(final_table, segment_mapping, segment_summary, monthly_trend, program_dist):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        final_table.to_excel(writer, sheet_name="Tabel RFM", index=False)
        segment_mapping.to_excel(writer, sheet_name="Segment Mapping", index=False)
        segment_summary.to_excel(writer, sheet_name="Dashboard Summary", index=False)
        monthly_trend.to_excel(writer, sheet_name="Tren Bulanan", index=False)
        program_dist.to_excel(writer, sheet_name="Distribusi Program", index=False)
    return buf.getvalue()


# ----------------------------------------------------------------------------
# SIDEBAR — SUMBER DATA & FILTER
# ----------------------------------------------------------------------------
st.sidebar.title("🕌 RFM Donor Ladder")
st.sidebar.caption("Mizan Amanah · Kelompok 1")

uploaded = st.sidebar.file_uploader("Ganti dataset (opsional)", type=["csv"])
data_source = uploaded if uploaded is not None else DEFAULT_DATA_PATH

if uploaded is None and not DEFAULT_DATA_PATH.exists():
    st.error(
        f"Dataset bawaan tidak ditemukan di `{DEFAULT_DATA_PATH}`. "
        "Silakan upload file data_set_donasi_ma_2020_2025.csv melalui sidebar."
    )
    st.stop()

df_clean = load_and_clean(data_source)
df_full, agg, segment_mapping, seg_to_reco, snapshot_date = build_rfm(df_clean)
monthly_trend_all, program_dist_all = compute_trend_and_program(df_full)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter")
seg_filter = st.sidebar.multiselect("Segmen Donor", SEGMENT_ORDER, default=SEGMENT_ORDER)
akad_options = sorted(agg["Dominan_Akad"].unique().tolist())
akad_filter = st.sidebar.multiselect("Akad Dominan", akad_options, default=akad_options)

agg_f = agg[agg["Segment"].isin(seg_filter) & agg["Dominan_Akad"].isin(akad_filter)].copy()
donor_ids_f = set(agg_f["donor_id"])
df_f = df_full[df_full["donor_id"].isin(donor_ids_f)]
monthly_trend_f, program_dist_f = compute_trend_and_program(df_f) if len(df_f) else (monthly_trend_all.iloc[0:0], program_dist_all.iloc[0:0])

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Snapshot date (acuan Recency): **{snapshot_date.date()}**  \n"
    f"Baris transaksi setelah cleaning: **{len(df_full):,}**  \n"
    f"Total donor unik: **{len(agg):,}**"
)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.title("RFM Segmentation & Donor Ladder Intelligence")
st.caption(
    "Dashboard interaktif — mereplikasi pipeline analitik pada notebook final "
    "`RFM_Mizan_Amanah_revisi.ipynb` (9 segmen Tangga Donatur, bukan model generik e-commerce)."
)

# KPI row (berdasarkan filter aktif)
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Donor (terfilter)", f"{len(agg_f):,}", f"dari {len(agg):,} total")
k2.metric("Total Transaksi", f"{agg_f['Frequency'].sum():,}")
k3.metric("Total Nominal", rupiah(agg_f["Monetary"].sum()))
k4.metric("Rata-rata Nominal/Donor", rupiah(agg_f["Monetary"].mean()) if len(agg_f) else "Rp0")
high_value_n = int(agg_f["High_Value_Donor"].sum()) if len(agg_f) else 0
k5.metric("Donor High-Value", f"{high_value_n:,}")

st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan & Segmentasi",
    "📈 Tren & Program",
    "🔍 Distribusi RFM",
    "🗺️ Segment Mapping & Rekomendasi",
    "🗂️ Donor Explorer & Export",
])

# ============================================================ TAB 1 =========
with tab1:
    if agg_f.empty:
        st.warning("Tidak ada donor pada filter saat ini. Sesuaikan filter di sidebar.")
    else:
        segment_summary = agg_f.groupby("Segment").agg(
            Jumlah_Donor=("donor_id", "count"),
            Total_Monetary=("Monetary", "sum"),
            Rata2_Monetary=("Monetary", "mean"),
        ).reset_index()
        segment_summary["Persentase_Donor"] = (segment_summary["Jumlah_Donor"] / segment_summary["Jumlah_Donor"].sum() * 100).round(2)
        segment_summary["Persentase_Nominal"] = (segment_summary["Total_Monetary"] / segment_summary["Total_Monetary"].sum() * 100).round(2)
        order = [s for s in SEGMENT_ORDER if s in segment_summary["Segment"].values]
        segment_summary["Segment"] = pd.Categorical(segment_summary["Segment"], categories=order, ordered=True)
        segment_summary = segment_summary.sort_values("Jumlah_Donor", ascending=False)

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = px.bar(
                segment_summary.sort_values("Jumlah_Donor"),
                x="Jumlah_Donor", y="Segment", orientation="h",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                title="Jumlah Donor per Segmen", text="Jumlah_Donor",
            )
            fig_bar.update_layout(showlegend=False, height=420, yaxis_title="", xaxis_title="Jumlah Donor")
            st.plotly_chart(fig_bar, width="stretch")
        with c2:
            fig_pie = px.pie(
                segment_summary, values="Total_Monetary", names="Segment",
                color="Segment", color_discrete_map=SEGMENT_COLORS,
                title="Kontribusi Nominal per Segmen", hole=0.35,
            )
            fig_pie.update_layout(height=420)
            st.plotly_chart(fig_pie, width="stretch")

        st.subheader("Ringkasan Segmen")
        disp = segment_summary.copy()
        disp["Total_Monetary"] = disp["Total_Monetary"].apply(rupiah)
        disp["Rata2_Monetary"] = disp["Rata2_Monetary"].apply(rupiah)
        disp.columns = ["Segment", "Jumlah Donor", "Total Nominal", "Rata-rata Nominal", "% Donor", "% Nominal"]
        st.dataframe(disp, width="stretch", hide_index=True)

        top_seg = segment_summary.sort_values("Persentase_Nominal", ascending=False).iloc[0]
        st.info(
            f"💡 **Insight**: Segmen **{top_seg['Segment']}** menyumbang **{top_seg['Persentase_Nominal']}%** "
            f"dari total nominal pada filter saat ini, dari **{top_seg['Persentase_Donor']}%** populasi donor."
        )

# ============================================================ TAB 2 =========
with tab2:
    if monthly_trend_f.empty:
        st.warning("Tidak ada data transaksi pada filter saat ini.")
    else:
        fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
        fig_trend.add_trace(
            go.Scatter(x=monthly_trend_f["year_month"], y=monthly_trend_f["Total_Nominal"],
                       name="Total Nominal", mode="lines+markers", line=dict(color=EMERALD, width=2)),
            secondary_y=False,
        )
        fig_trend.add_trace(
            go.Scatter(x=monthly_trend_f["year_month"], y=monthly_trend_f["Jumlah_Transaksi"],
                       name="Jumlah Transaksi", mode="lines+markers", line=dict(color=GOLD, width=2)),
            secondary_y=True,
        )
        fig_trend.update_layout(title="Tren Donasi Bulanan (Nominal vs Jumlah Transaksi)", height=420,
                                 legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig_trend.update_yaxes(title_text="Total Nominal (Rp)", secondary_y=False)
        fig_trend.update_yaxes(title_text="Jumlah Transaksi", secondary_y=True)
        st.plotly_chart(fig_trend, width="stretch")

        c1, c2 = st.columns([3, 2])
        with c1:
            top10 = program_dist_f.head(10).sort_values("Persentase_Nominal")
            fig_prog = px.bar(
                top10, x="Persentase_Nominal", y="program", orientation="h",
                title="Top 10 Program (% Kontribusi Nominal)", text="Persentase_Nominal",
                color_discrete_sequence=[EMERALD],
            )
            fig_prog.update_traces(texttemplate="%{text}%")
            fig_prog.update_layout(height=420, yaxis_title="", xaxis_title="% Nominal")
            st.plotly_chart(fig_prog, width="stretch")
        with c2:
            pola_counts = agg_f["Pola_Donasi"].value_counts().reindex(
                ["Meningkat", "Stabil", "Menurun", "Data Tidak Cukup"]
            ).fillna(0).astype(int)
            fig_pola = px.bar(
                x=pola_counts.index, y=pola_counts.values,
                title="Pola Donasi Donor", color=pola_counts.index,
                color_discrete_map={"Meningkat": "#2E9E8A", "Stabil": GOLD, "Menurun": "#B0573A", "Data Tidak Cukup": "#B0B8B6"},
            )
            fig_pola.update_layout(height=420, showlegend=False, xaxis_title="", yaxis_title="Jumlah Donor")
            st.plotly_chart(fig_pola, width="stretch")

        with st.expander("Lihat tabel distribusi program lengkap"):
            disp_p = program_dist_f.copy()
            disp_p["Total_Nominal"] = disp_p["Total_Nominal"].apply(rupiah)
            disp_p.columns = ["Program", "Total Nominal", "Jumlah Transaksi", "% Nominal"]
            st.dataframe(disp_p, width="stretch", hide_index=True)

# ============================================================ TAB 3 =========
with tab3:
    if agg_f.empty:
        st.warning("Tidak ada donor pada filter saat ini.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            fig_r = px.histogram(agg_f, x="Recency", nbins=50, color_discrete_sequence=["#E07A5F"],
                                  title="Distribusi Recency (hari)")
            fig_r.update_layout(height=380, xaxis_title="Hari sejak donasi terakhir", yaxis_title="Jumlah Donor")
            st.plotly_chart(fig_r, width="stretch")
        with c2:
            freq_clip = agg_f["Frequency"].clip(upper=30)
            fig_f = px.histogram(x=freq_clip, nbins=30, color_discrete_sequence=["#3D7EA6"],
                                  title="Distribusi Frequency (maks. 30x)")
            fig_f.update_layout(height=380, xaxis_title="Jumlah Transaksi", yaxis_title="Jumlah Donor")
            st.plotly_chart(fig_f, width="stretch")
        with c3:
            log_monetary = np.log1p(agg_f["Monetary"])
            fig_m = px.histogram(x=log_monetary, nbins=50, color_discrete_sequence=["#3FA796"],
                                  title="Distribusi Monetary (skala log)")
            fig_m.update_layout(height=380, xaxis_title="log(1 + Nilai Total Donasi)", yaxis_title="Jumlah Donor")
            st.plotly_chart(fig_m, width="stretch")

        st.caption(
            "Mayoritas donor cenderung berada pada recency tinggi (tidak aktif) & frequency rendah, "
            "sejalan dengan dominasi segmen Lost & Dormant Donor pada hasil klasifikasi."
        )

# ============================================================ TAB 4 =========
with tab4:
    st.subheader("Kriteria Klasifikasi 9 Segmen (Segment Mapping)")
    st.dataframe(segment_mapping, width="stretch", hide_index=True)

    st.subheader("Rekomendasi Strategi Engagement per Segmen")
    pilih_segmen = st.selectbox("Pilih segmen untuk detail strategi:", SEGMENT_ORDER)
    st.success(f"**{pilih_segmen}** → {seg_to_reco[pilih_segmen]}")

    reco_df = pd.DataFrame({"Segment": list(seg_to_reco.keys()), "Strategi": list(seg_to_reco.values())})
    reco_df["Segment"] = pd.Categorical(reco_df["Segment"], categories=SEGMENT_ORDER, ordered=True)
    reco_df = reco_df.sort_values("Segment")
    with st.expander("Lihat semua rekomendasi strategi (9 segmen)"):
        st.dataframe(reco_df, width="stretch", hide_index=True)

# ============================================================ TAB 5 =========
with tab5:
    st.subheader("Donor Explorer")
    search_id = st.text_input("Cari berdasarkan donor_id (opsional):").strip()

    final_cols = [
        "donor_id", "Recency", "Frequency", "Monetary", "Avg_Nominal",
        "First_Donation", "Last_Donation", "Duration_Days",
        "Dominan_Program", "Dominan_Akad", "Max_Yearly_Monetary",
        "High_Value_Donor", "Pola_Donasi", "Segment", "Rekomendasi_Strategi",
    ]
    final_table_full = agg[final_cols]
    table_view = agg_f[final_cols]
    if search_id:
        table_view = table_view[table_view["donor_id"].str.contains(search_id, case=False, na=False)]

    st.dataframe(table_view, width="stretch", hide_index=True, height=420)
    st.caption(f"Menampilkan {len(table_view):,} dari {len(agg):,} total donor (sesuai filter aktif & pencarian).")

    st.markdown("---")
    st.subheader("Export Deliverable Final")
    colA, colB = st.columns(2)

    with colA:
        csv_bytes = final_table_full.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download rfm_output_FINAL.csv (seluruh donor)",
            data=csv_bytes, file_name="rfm_output_FINAL.csv", mime="text/csv",
            width="stretch",
        )
    with colB:
        seg_summary_full = agg.groupby("Segment").agg(
            Jumlah_Donor=("donor_id", "count"), Total_Monetary=("Monetary", "sum"),
            Rata2_Monetary=("Monetary", "mean"),
        ).reset_index()
        seg_summary_full["Persentase_Donor"] = (seg_summary_full["Jumlah_Donor"] / seg_summary_full["Jumlah_Donor"].sum() * 100).round(2)
        seg_summary_full["Persentase_Nominal"] = (seg_summary_full["Total_Monetary"] / seg_summary_full["Total_Monetary"].sum() * 100).round(2)
        xlsx_bytes = to_excel_bytes(final_table_full, segment_mapping, seg_summary_full, monthly_trend_all, program_dist_all)
        st.download_button(
            "⬇️ Download Kebutuhan_informasi_data_HASIL.xlsx (5 sheet)",
            data=xlsx_bytes, file_name="Kebutuhan_informasi_data_HASIL.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )

st.markdown("---")
st.caption(
    "Dibangun mengikuti pipeline & threshold yang tervalidasi pada RFM_Mizan_Amanah_revisi.ipynb · "
    "Kelompok 1 — Hengkie Wirawijaya, Tazkia Shafaluna Indihani, Beta Elok Yuanata, Afiani Dewi Rizky"
)

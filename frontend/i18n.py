"""
Bilingual string registry — Indonesian (ID) and English (EN).

Usage:
    from i18n import t, lang_selector
    lang_selector()          # renders toggle in sidebar
    label = t("btn_login")   # returns string in current language
    msg = t("showing_analyses", 5)  # supports .format() args
"""
from __future__ import annotations
import streamlit as st

# ── Translation dictionary ────────────────────────────────────────────────────

_STRINGS: dict[str, dict[str, str]] = {
    "ID": {
        # Common / sidebar
        "logout": "🚪 Keluar",

        # Auth page
        "auth_subtitle": "Analisis keterlibatan siswa berbasis AI dari video kelas",
        "tab_login": "🔐 Masuk",
        "tab_signup": "✨ Daftar",
        "login_welcome": "Selamat datang kembali! Masuk ke akun Anda.",
        "signup_welcome": "Buat akun baru untuk memulai.",
        "label_email": "Email",
        "label_password": "Kata Sandi",
        "label_fullname": "Nama Lengkap (opsional)",
        "label_confirm_pw": "Konfirmasi Kata Sandi",
        "ph_password": "Kata sandi Anda",
        "ph_confirm_pw": "Masukkan ulang kata sandi",
        "ph_fullname": "Nama Lengkap",
        "btn_login": "Masuk",
        "btn_create": "Buat Akun",
        "err_fill_all": "Harap isi semua kolom.",
        "err_fill_required": "Harap isi semua kolom yang wajib.",
        "err_pw_mismatch": "Kata sandi tidak cocok.",
        "err_pw_short": "Kata sandi minimal 6 karakter.",
        "err_backend": "Tidak dapat terhubung ke server. Apakah server FastAPI sudah berjalan di port 8000?",
        "err_timeout": "Permintaan habis waktu. Silakan coba lagi.",
        "err_login_fail": "Login gagal: {}",
        "err_signup_fail": "Pendaftaran gagal: {}",
        "spinner_login": "Sedang masuk…",
        "spinner_signup": "Membuat akun…",
        "success_login": "Berhasil masuk!",
        "success_signup_confirm": "Akun dibuat! Silakan cek email untuk konfirmasi, lalu masuk.",
        "success_signup": "Akun dibuat! Anda sudah masuk.",
        "require_auth_warn": "Silakan masuk untuk mengakses halaman ini.",

        # Home
        "home_subtitle": "Analisis keterlibatan siswa dari video kelas menggunakan deteksi, pelacakan, dan klasifikasi berbasis AI.",
        "feat_upload_title": "Upload Video",
        "feat_upload_desc": "Upload video kelas dan biarkan AI menganalisis tingkat keterlibatan setiap siswa.",
        "feat_results_title": "Lihat Hasil",
        "feat_results_desc": "Lihat laporan keterlibatan per siswa dengan grafik interaktif dan video beranotasi.",
        "feat_history_title": "Riwayat",
        "feat_history_desc": "Telusuri semua analisis sebelumnya, bandingkan hasil, dan unduh laporan.",
        "backend_ok": "✅ Backend online — model dimuat (perangkat: {})",
        "backend_loading": "⏳ Backend sedang menyala — model masih dimuat…",
        "backend_error": "❌ Tidak dapat terhubung ke backend API. Pastikan FastAPI berjalan di port 8000.",

        # Upload page
        "upload_title": "Upload Video Kelas",
        "upload_subtitle": "Upload video (maks {} MB) dan biarkan AI menganalisis keterlibatan siswa",
        "how_it_works": "Cara Kerja",
        "step1_title": "Upload",
        "step1_desc": "Pilih file video kelas",
        "step2_title": "Proses",
        "step2_desc": "AI mendeteksi, melacak &amp; mengklasifikasi siswa",
        "step3_title": "Hasil",
        "step3_desc": "Lihat laporan keterlibatan per siswa",
        "select_video": "Pilih Video",
        "btn_analyze": "🚀 Analisis Keterlibatan",
        "upload_timeout": (
            "⏱️ **Server sedang memproses video Anda.**  \n"
            "Koneksi ke server timeout, tapi video kemungkinan besar "
            "sudah diterima dan sedang diproses.  \n\n"
            "Silakan cek halaman **Riwayat** untuk memantau status analisis."
        ),
        "upload_conn_err": (
            "❌ **Tidak bisa terhubung ke server.**  \n"
            "Pastikan backend sudah berjalan di `http://localhost:8000`."
        ),
        "upload_fail": "❌ **Upload gagal.**  \nDetail teknis: `{}`  \n\nCoba lagi atau cek halaman Riwayat.",
        "btn_open_history": "📋 Buka Riwayat",
        "btn_check_history": "📋 Cek Riwayat",
        "upload_success": "✅ **Video berhasil dikirim!**  \nPipeline analisis sedang berjalan di background.  \nID Analisis: `{}`",
        "upload_redirecting": "🔄 Anda akan diarahkan ke halaman Riwayat untuk memantau progress…",

        # Results page
        "results_no_analysis": "Tidak ada analisis terpilih",
        "results_no_analysis_sub": "Unggah video atau pilih analisis dari Riwayat untuk melihat hasil.",
        "btn_go_upload": "📤 Pergi ke Upload",
        "btn_go_history": "📋 Pergi ke Riwayat",
        "results_load_err": "Tidak dapat memuat hasil: {}",
        "results_not_ready": "Status analisis: **{}**. Hasil belum siap.",
        "results_title": "Hasil Analisis",
        "metric_students": "👥 Jumlah Siswa",
        "metric_frames": "🎞️ Jumlah Frame",
        "metric_confidence": "🎯 Rata-rata Confidence",
        "metric_time": "⏱️ Waktu Pemrosesan",
        "metric_detections": "📊 Total Deteksi",
        "section_class_summary": "Ringkasan Keterlibatan Kelas",
        "majority_vote_note": "Berdasarkan <b>voting mayoritas</b> di semua frame untuk masing-masing dari <b>{}</b> siswa yang terlacak.",
        "section_video": "Video Beranotasi",
        "section_per_student": "Keterlibatan per Siswa",
        "tab_table": "📋 Tabel",
        "tab_bar": "📊 Diagram Batang",
        "tab_stack": "📈 Rincian Voting",
        "col_student_id": "ID Siswa",
        "col_engagement": "Tingkat Keterlibatan",
        "col_engaged_votes": "Frame Terlibat",
        "col_not_engaged_votes": "Frame Tidak Terlibat",
        "col_total_frames": "Total Frame",
        "col_avg_conf": "Rata-rata Confidence",
        "col_majority_vote": "Majority Vote %",
        "no_student_data": "Data siswa tidak tersedia.",
        "section_downloads": "Unduhan",
        "download_csv_title": "Unduh CSV Mentah",
        "download_csv_sub": "Data pelacakan per-frame",
        "download_video_title": "Unduh Video Beranotasi",
        "download_video_sub": "Video dengan kotak pembatas &amp; label",
        "no_csv": "CSV tidak tersedia.",
        "no_video": "Video beranotasi tidak tersedia.",
        "btn_back_history": "← Kembali ke Riwayat",

        # History page
        "history_title": "Riwayat Analisis",
        "history_subtitle": "Telusuri dan kelola riwayat analisis Anda",
        "history_load_err": "Tidak dapat memuat riwayat: {}",
        "showing_analyses": "Menampilkan <b>{}</b> analisis",
        "showing_processing_suffix": " — 🔴 ada yang sedang diproses, halaman akan refresh otomatis",
        "btn_refresh": "🔄 Refresh",
        "empty_title": "Belum ada analisis",
        "empty_sub": "Upload video kelas untuk memulai!",
        "btn_upload_video": "📤 Upload Video",
        "students_label": "siswa",
        "btn_view_results": "📊 Lihat Hasil",
        "btn_delete": "🗑️ Hapus",
        "delete_success": "Berhasil dihapus!",
        "delete_fail": "Hapus gagal: {}",
        "processing_note": "⏳ Sedang memproses video{} — halaman akan refresh otomatis setiap 10 detik.",

        # Profile page
        "profile_title": "Profil Anda",
        "profile_subtitle": "Informasi akun dan pengaturan",
        "profile_label_email": "Email",
        "profile_label_uid": "User ID",

        # Engagement labels (used in table & cards)
        "label_engaged": "Terlibat",
        "label_not_engaged": "Tidak Terlibat",
    },

    "EN": {
        # Common / sidebar
        "logout": "🚪 Logout",

        # Auth page
        "auth_subtitle": "AI-powered student engagement analysis from classroom videos",
        "tab_login": "🔐 Login",
        "tab_signup": "✨ Sign Up",
        "login_welcome": "Welcome back! Sign in to your account.",
        "signup_welcome": "Create a new account to get started.",
        "label_email": "Email",
        "label_password": "Password",
        "label_fullname": "Full Name (optional)",
        "label_confirm_pw": "Confirm Password",
        "ph_password": "Your password",
        "ph_confirm_pw": "Re-enter password",
        "ph_fullname": "John Doe",
        "btn_login": "Login",
        "btn_create": "Create Account",
        "err_fill_all": "Please fill in all fields.",
        "err_fill_required": "Please fill in all required fields.",
        "err_pw_mismatch": "Passwords do not match.",
        "err_pw_short": "Password must be at least 6 characters.",
        "err_backend": "Cannot reach the backend. Is the FastAPI server running on port 8000?",
        "err_timeout": "Request timed out. Please try again.",
        "err_login_fail": "Login failed: {}",
        "err_signup_fail": "Sign up failed: {}",
        "spinner_login": "Logging in…",
        "spinner_signup": "Creating account…",
        "success_login": "Logged in!",
        "success_signup_confirm": "Account created! Please check your email to confirm, then log in.",
        "success_signup": "Account created! You are now logged in.",
        "require_auth_warn": "Please log in to access this page.",

        # Home
        "home_subtitle": "Analyze student engagement from classroom videos using AI-powered detection, tracking, and classification.",
        "feat_upload_title": "Upload Video",
        "feat_upload_desc": "Upload a classroom video and let AI analyze each student's engagement level.",
        "feat_results_title": "View Results",
        "feat_results_desc": "See detailed per-student engagement reports with interactive charts and annotated video.",
        "feat_history_title": "History",
        "feat_history_desc": "Browse all your past analyses, compare results, and download reports.",
        "backend_ok": "✅ Backend online — models loaded (device: {})",
        "backend_loading": "⏳ Backend is starting up — models are still loading…",
        "backend_error": "❌ Cannot reach the backend API. Make sure FastAPI is running on port 8000.",

        # Upload page
        "upload_title": "Upload Classroom Video",
        "upload_subtitle": "Upload a video (max {} MB) and let AI analyze student engagement",
        "how_it_works": "How It Works",
        "step1_title": "Upload",
        "step1_desc": "Select a classroom video file",
        "step2_title": "Process",
        "step2_desc": "AI detects, tracks &amp; classifies students",
        "step3_title": "Results",
        "step3_desc": "View per-student engagement reports",
        "select_video": "Select Video",
        "btn_analyze": "🚀 Analyze Engagement",
        "upload_timeout": (
            "⏱️ **Server is processing your video.**  \n"
            "The connection timed out, but the video was likely received and is being processed.  \n\n"
            "Please check the **History** page to monitor the analysis status."
        ),
        "upload_conn_err": (
            "❌ **Cannot connect to the server.**  \n"
            "Make sure the backend is running at `http://localhost:8000`."
        ),
        "upload_fail": "❌ **Upload failed.**  \nTechnical details: `{}`  \n\nTry again or check History.",
        "btn_open_history": "📋 Open History",
        "btn_check_history": "📋 Check History",
        "upload_success": "✅ **Video submitted!**  \nAnalysis pipeline is running in the background.  \nAnalysis ID: `{}`",
        "upload_redirecting": "🔄 You will be redirected to History to monitor progress…",

        # Results page
        "results_no_analysis": "No analysis selected",
        "results_no_analysis_sub": "Upload a video or select an analysis from History to view results.",
        "btn_go_upload": "📤 Go to Upload",
        "btn_go_history": "📋 Go to History",
        "results_load_err": "Could not load results: {}",
        "results_not_ready": "Analysis status: **{}**. Results not ready yet.",
        "results_title": "Analysis Results",
        "metric_students": "👥 Students",
        "metric_frames": "🎞️ Total Frames",
        "metric_confidence": "🎯 Avg Confidence",
        "metric_time": "⏱️ Processing Time",
        "metric_detections": "📊 Total Detections",
        "section_class_summary": "Class Engagement Summary",
        "majority_vote_note": "Based on <b>majority voting</b> across all frames for each of <b>{}</b> tracked students.",
        "section_video": "Annotated Video",
        "section_per_student": "Per-Student Engagement",
        "tab_table": "📋 Table",
        "tab_bar": "📊 Bar Chart",
        "tab_stack": "📈 Vote Breakdown",
        "col_student_id": "Student ID",
        "col_engagement": "Engagement Level",
        "col_engaged_votes": "Engaged Frames",
        "col_not_engaged_votes": "Not-Engaged Frames",
        "col_total_frames": "Total Frames",
        "col_avg_conf": "Avg Confidence",
        "col_majority_vote": "Majority Vote %",
        "no_student_data": "No student data available.",
        "section_downloads": "Downloads",
        "download_csv_title": "Download Raw CSV",
        "download_csv_sub": "Per-frame tracking data",
        "download_video_title": "Download Annotated Video",
        "download_video_sub": "Video with bounding boxes &amp; labels",
        "no_csv": "CSV not available.",
        "no_video": "Annotated video not available.",
        "btn_back_history": "← Back to History",

        # History page
        "history_title": "Analysis History",
        "history_subtitle": "Browse and manage your analysis history",
        "history_load_err": "Could not load history: {}",
        "showing_analyses": "Showing <b>{}</b> past analyses",
        "showing_processing_suffix": " — 🔴 some are being processed, page will auto-refresh",
        "btn_refresh": "🔄 Refresh",
        "empty_title": "No analyses yet",
        "empty_sub": "Upload a classroom video to get started!",
        "btn_upload_video": "📤 Upload Video",
        "students_label": "students",
        "btn_view_results": "📊 View Results",
        "btn_delete": "🗑️ Delete",
        "delete_success": "Deleted successfully!",
        "delete_fail": "Delete failed: {}",
        "processing_note": "⏳ Processing video{} — page auto-refreshes every 10 seconds.",

        # Profile page
        "profile_title": "Your Profile",
        "profile_subtitle": "Account information and settings",
        "profile_label_email": "Email",
        "profile_label_uid": "User ID",

        # Engagement labels
        "label_engaged": "Engaged",
        "label_not_engaged": "Not Engaged",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_lang() -> str:
    return st.session_state.get("lang", "ID")


def t(key: str, *args: object) -> str:
    lang = get_lang()
    s = _STRINGS.get(lang, _STRINGS["ID"]).get(key) or _STRINGS["ID"].get(key, key)
    return s.format(*args) if args else s


def lang_selector() -> None:
    """Render a compact language toggle in the sidebar. Call once per page render."""
    options = ["🇮🇩 Indonesia", "🇬🇧 English"]
    current_idx = 0 if get_lang() == "ID" else 1
    chosen = st.sidebar.radio(
        "🌐",
        options,
        index=current_idx,
        horizontal=True,
        label_visibility="collapsed",
        key="__lang_radio",
    )
    new_lang = "ID" if "Indonesia" in chosen else "EN"
    if new_lang != get_lang():
        st.session_state["lang"] = new_lang
        st.rerun()

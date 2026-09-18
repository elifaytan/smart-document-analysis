import os
import html
import json
import hashlib
import re
import unicodedata
from pathlib import Path
from difflib import SequenceMatcher

import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.suggested_questions import get_suggested_questions
from src.source_verifier import verify_source_text
from src.analysis_statistics import (
    calculate_analysis_statistics,
)

from src.analysis_engine import (
    AnalysisEngineError,
    DocumentAnalysis,
    analyze_document,
)
from src.keyword_search import (
    KeywordSearchError,
    KeywordSearchResult,
    search_document_pages,
)
from src.document_comparator import (
    ComparisonFinding,
    DocumentComparison,
    DocumentComparisonError,
    RiskChange,
    compare_documents,
    compare_documents_local,
)
from src.history_manager import (
    HistoryManagerError,
    clear_document_questions,
    delete_document,
    initialize_database,
    list_recent_documents,
    load_analysis,
    load_questions,
    read_saved_pdf,
    save_analysis,
    save_document,
    save_question,
    update_document_type,
)
from src.pdf_page_viewer import (
    PagePreviewError,
    render_pdf_page,
)
from src.pdf_reader import PDFReaderError, read_pdf
from src.report_generator import (
    ReportGeneratorError,
    create_report_filename,
    generate_analysis_report,
)
from src.question_answer import (
    QuestionAnswerError,
    QuestionAnswerResult,
    answer_document_question,
)
from src.text_splitter import (
    TextSplitterError,
    split_pdf_pages,
)
from src.vector_store import (
    VectorStoreError,
    create_vector_store,
)


load_dotenv(override=True)


st.set_page_config(
    page_title="TSE Akıllı Doküman Asistanı",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


DOCUMENT_TYPES = [
    "Teknik Şartname",
    "Prosedür",
    "Denetim Dokümanı",
    "Başvuru veya Belgelendirme Rehberi",
    "Standartla İlgili Rehber",
    "Diğer",
]


COMPARISON_CACHE_DIRECTORY = Path("data/comparison_cache")
AI_INTERPRETATION_CACHE_DIRECTORY = Path("data/ai_interpretation_cache")


def create_comparison_cache_key(
    old_file_bytes: bytes,
    new_file_bytes: bytes,
    model_name: str,
) -> str:
    """Aynı iki PDF ve model için değişmeyen önbellek anahtarı üretir."""

    digest = hashlib.sha256()
    digest.update(old_file_bytes)
    digest.update(b"\0TSE-COMPARISON\0")
    digest.update(new_file_bytes)
    digest.update(b"\0MODEL\0")
    digest.update(model_name.encode("utf-8"))
    return digest.hexdigest()


def load_cached_comparison(cache_key: str) -> DocumentComparison | None:
    """Daha önce üretilmiş karşılaştırmayı yerel diskten yükler."""

    cache_file = COMPARISON_CACHE_DIRECTORY / f"{cache_key}.json"
    if not cache_file.exists():
        return None

    try:
        return DocumentComparison.model_validate_json(
            cache_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        # Bozulmuş önbellek uygulamayı durdurmamalıdır.
        try:
            cache_file.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def save_cached_comparison(
    cache_key: str,
    comparison: DocumentComparison,
) -> None:
    """Karşılaştırma sonucunu sonraki kullanımlar için kaydeder."""

    try:
        COMPARISON_CACHE_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )
        cache_file = COMPARISON_CACHE_DIRECTORY / f"{cache_key}.json"
        cache_file.write_text(
            comparison.model_dump_json(indent=2),
            encoding="utf-8",
        )
    except OSError:
        # Önbellek yardımcı özelliktir; kayıt başarısız olsa bile sonuç gösterilir.
        return


def create_ai_interpretation_cache_key(
    title: str,
    explanation: str,
    old_text: str,
    new_text: str,
    model_name: str,
) -> str:
    """Tek bir değişiklik yorumu için kalıcı önbellek anahtarı üretir."""

    digest = hashlib.sha256()
    for value in (title, explanation, old_text, new_text, model_name):
        digest.update((value or "").encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def load_cached_ai_interpretation(cache_key: str) -> str | None:
    """Daha önce oluşturulan AI yorumunu yerel diskten yükler."""

    cache_file = AI_INTERPRETATION_CACHE_DIRECTORY / f"{cache_key}.txt"
    if not cache_file.exists():
        return None

    try:
        value = cache_file.read_text(encoding="utf-8").strip()
        return value or None
    except OSError:
        return None


def save_cached_ai_interpretation(cache_key: str, interpretation: str) -> None:
    """AI yorumunu aynı madde için yeniden kota harcanmaması amacıyla kaydeder."""

    try:
        AI_INTERPRETATION_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        cache_file = AI_INTERPRETATION_CACHE_DIRECTORY / f"{cache_key}.txt"
        cache_file.write_text(interpretation.strip(), encoding="utf-8")
    except OSError:
        return


def generate_ai_change_interpretation(
    title: str,
    explanation: str,
    old_text: str,
    new_text: str,
    api_key: str,
    model_name: str,
) -> str:
    """Yalnızca seçilen değişiklik için kısa ve kaynakla sınırlı Gemini yorumu üretir."""

    if not api_key:
        raise DocumentComparisonError(
            "Gemini API anahtarı bulunamadı. .env dosyasındaki "
            "GOOGLE_API_KEY değerini kontrol edin."
        )

    model = ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0,
    )

    prompt = f"""
Sen teknik prosedür ve standart revizyonlarını inceleyen dikkatli bir TSE doküman uzmanısın.
Yalnızca aşağıdaki başlık, yerel açıklama ve kaynak metinleri kullan. Dokümanlarda bulunmayan bilgi üretme.
Değişikliği Türkçe, kısa ve açık biçimde yorumla.

Çıktıyı tam olarak şu başlıklarla ver:
**Değişikliğin anlamı:** 1-2 cümle
**Muhtemel etkisi:** 1-2 cümle
**Önerilen kontrol:** 1 kısa ve uygulanabilir madde

Başlık: {title}
Karşılaştırma açıklaması: {explanation}

ESKİ KAYNAK:
{old_text or "Eski dokümanda kaynak bulunmuyor."}

YENİ KAYNAK:
{new_text or "Yeni dokümanda kaynak bulunmuyor."}
"""

    response = model.invoke(prompt)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        content = "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        )

    interpretation = str(content).strip()
    if not interpretation:
        raise DocumentComparisonError("Gemini boş bir yorum döndürdü.")
    return interpretation


def display_ai_interpretation_control(
    finding: ComparisonFinding,
    finding_key: str,
) -> None:
    """Seçilen değişiklik için isteğe bağlı, tek istekli AI yorum alanını gösterir."""

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    model_name = os.getenv(
        "GEMINI_CHAT_MODEL",
        "gemini-flash-latest",
    ).strip()

    cache_key = create_ai_interpretation_cache_key(
        title=finding.title,
        explanation=finding.explanation,
        old_text=finding.old_source_text,
        new_text=finding.new_source_text,
        model_name=model_name,
    )

    interpretation = st.session_state.ai_interpretations.get(cache_key)
    if not interpretation:
        interpretation = load_cached_ai_interpretation(cache_key)
        if interpretation:
            st.session_state.ai_interpretations[cache_key] = interpretation

    if interpretation:
        meaning = interpretation
        impact = ""
        control = ""

        meaning_match = re.search(
            r"\*\*Değişikliğin anlamı:\*\*\s*(.*?)(?=\n\*\*Muhtemel etkisi:|$)",
            interpretation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        impact_match = re.search(
            r"\*\*Muhtemel etkisi:\*\*\s*(.*?)(?=\n\*\*Önerilen kontrol:|$)",
            interpretation,
            flags=re.DOTALL | re.IGNORECASE,
        )
        control_match = re.search(
            r"\*\*Önerilen kontrol:\*\*\s*(.*)$",
            interpretation,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if meaning_match:
            meaning = meaning_match.group(1).strip()
        if impact_match:
            impact = impact_match.group(1).strip()
        if control_match:
            control = control_match.group(1).strip().lstrip("-• ")

        with st.expander("🤖 Yapay Zekâ Yorumu", expanded=False):
            with st.container(border=True):
                st.markdown("**📌 Değişikliğin anlamı**")
                st.write(meaning)
                if impact:
                    st.markdown("**⚠️ Olası etkisi**")
                    st.write(impact)
                if control:
                    st.markdown("**✅ Önerilen kontrol**")
                    st.write(control)
        return

    button_clicked = st.button(
        "🤖 Yapay Zekâ Yorumu Oluştur",
        key=f"ai_interpret_{finding_key}_{cache_key[:12]}",
        use_container_width=True,
    )

    if not button_clicked:
        return

    try:
        with st.spinner("Değişiklik değerlendiriliyor..."):
            interpretation = generate_ai_change_interpretation(
                title=finding.title,
                explanation=finding.explanation,
                old_text=finding.old_source_text,
                new_text=finding.new_source_text,
                api_key=api_key,
                model_name=model_name,
            )
            save_cached_ai_interpretation(cache_key, interpretation)
            st.session_state.ai_interpretations[cache_key] = interpretation
        st.rerun()
    except Exception as exc:
        st.error(get_user_friendly_comparison_error(exc))


def get_user_friendly_comparison_error(error: Exception) -> str:
    """Gemini ve bağlantı hatalarını kullanıcı dostu Türkçe mesaja çevirir."""

    error_text = str(error)
    normalized = error_text.lower()

    if (
        "resource_exhausted" in normalized
        or "quota exceeded" in normalized
        or "429" in normalized
    ):
        return (
            "Gemini ücretsiz API kullanım kotası doldu. "
            "Bu durum PDF dosyalarından veya uygulama kodundan kaynaklanmaz. "
            "Kota yenilendikten sonra tekrar deneyin. Aynı iki dokümanın daha "
            "önce oluşturulmuş sonucu varsa uygulama bunu önbellekten açar."
        )

    if (
        "api key" in normalized
        or "permission_denied" in normalized
        or "unauthenticated" in normalized
        or "401" in normalized
        or "403" in normalized
    ):
        return (
            "Gemini API anahtarı doğrulanamadı. .env dosyasındaki "
            "GOOGLE_API_KEY değerini kontrol edin."
        )

    if (
        "timeout" in normalized
        or "connection" in normalized
        or "network" in normalized
        or "unavailable" in normalized
    ):
        return (
            "Gemini hizmetine şu anda ulaşılamıyor. İnternet bağlantınızı "
            "kontrol edip kısa süre sonra yeniden deneyin."
        )

    return f"Doküman karşılaştırması tamamlanamadı: {error_text}"


def initialize_session_state() -> None:
    """
    Streamlit oturumunda korunacak değişkenleri oluşturur.
    """

    default_values = {
        "pdf_result": None,
        "uploaded_file_name": None,
        "uploaded_file_bytes": None,
        "document_chunks": [],
        "vector_store": None,
        "vector_store_result": None,
        "document_type": None,
        "analysis_result": None,
        "question_history": [],
        "current_document_id": None,
        "history_mode": False,
        "history_file_name": None,
        "document_type_selector": DOCUMENT_TYPES[0],
        "keyword_search_query": "",
        "keyword_search_results": [],
        "keyword_preview_page": None,
        "keyword_preview_query": "",
        "app_mode": "analysis",
        "comparison_result": None,
        "comparison_old_file_name": None,
        "comparison_new_file_name": None,
        "comparison_notice": None,
        "ai_interpretations": {},
    }

    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state() -> None:
    """
    Açık dokümana ait oturum verilerini temizler.
    """

    st.session_state.pdf_result = None
    st.session_state.uploaded_file_name = None
    st.session_state.uploaded_file_bytes = None
    st.session_state.document_chunks = []
    st.session_state.vector_store = None
    st.session_state.vector_store_result = None
    st.session_state.document_type = None
    st.session_state.analysis_result = None
    st.session_state.question_history = []
    st.session_state.current_document_id = None
    st.session_state.history_mode = False
    st.session_state.history_file_name = None
    st.session_state.keyword_search_query = ""
    st.session_state.keyword_search_results = []
    st.session_state.keyword_preview_page = None
    st.session_state.keyword_preview_query = ""


def format_history_date(
    iso_date: str,
) -> str:
    """
    ISO tarihini kullanıcı dostu biçime dönüştürür.
    """

    if not iso_date:
        return ""

    try:
        date_part = iso_date.split("T")[0]
        year, month, day = date_part.split("-")

        return f"{day}.{month}.{year}"

    except ValueError:
        return iso_date


def load_saved_analysis_and_questions(
    document_id: str,
) -> None:
    """
    Kaydedilmiş analiz ve soruları oturuma yükler.
    """

    saved_analysis_json = load_analysis(
        document_id=document_id
    )

    if saved_analysis_json:
        st.session_state.analysis_result = (
            DocumentAnalysis.model_validate_json(
                saved_analysis_json
            )
        )

    else:
        st.session_state.analysis_result = None

    st.session_state.question_history = load_questions(
        document_id=document_id
    )


def open_history_document(
    document_id: str,
) -> None:
    """
    Geçmişte kayıtlı PDF'yi ana ekrana yükler.
    """

    try:
        file_bytes, document = read_saved_pdf(
            document_id=document_id
        )

        pdf_result = read_pdf(
            file_bytes=file_bytes,
            file_name=document.file_name,
        )

        if pdf_result.word_count == 0:
            raise PDFReaderError(
                "Kayıtlı PDF'den okunabilir metin çıkarılamadı."
            )

        document_chunks = split_pdf_pages(
            page_texts=pdf_result.page_texts,
            chunk_size=900,
            chunk_overlap=150,
        )

        reset_document_state()

        st.session_state.pdf_result = pdf_result
        st.session_state.uploaded_file_name = (
            document.file_name
        )
        st.session_state.uploaded_file_bytes = file_bytes
        st.session_state.document_chunks = document_chunks
        st.session_state.document_type = (
            document.document_type
        )
        st.session_state.document_type_selector = (
            document.document_type
        )
        st.session_state.current_document_id = (
            document.document_id
        )
        st.session_state.history_mode = True
        st.session_state.history_file_name = (
            document.file_name
        )

        load_saved_analysis_and_questions(
            document_id=document.document_id
        )

        st.rerun()

    except (
        HistoryManagerError,
        PDFReaderError,
        TextSplitterError,
    ) as exc:
        st.error(str(exc))

    except Exception as exc:
        st.error(
            f"Geçmiş doküman açılırken hata oluştu: {exc}"
        )


def remove_history_document(
    document_id: str,
) -> None:
    """
    Geçmiş kaydı siler.
    """

    try:
        delete_document(
            document_id=document_id
        )

        if (
            st.session_state.current_document_id
            == document_id
        ):
            reset_document_state()

        st.rerun()

    except HistoryManagerError as exc:
        st.error(str(exc))


def display_recent_documents() -> None:
    """
    Son dokümanları sol menüde gösterir.
    """

    try:
        recent_documents = list_recent_documents(
            limit=8
        )

    except HistoryManagerError as exc:
        st.warning(str(exc))
        return

    if not recent_documents:
        return

    st.divider()
    st.subheader("Geçmiş dokümanlar")

    for document in recent_documents:
        with st.expander(
            document.file_name,
            expanded=False,
        ):
            st.write(
                f"**Tür:** {document.document_type}"
            )

            st.write(
                f"**Sayfa:** {document.page_count}"
            )

            st.write(
                f"**Soru sayısı:** "
                f"{document.question_count}"
            )

            analysis_status = (
                "Tamamlandı"
                if document.has_analysis
                else "Analiz edilmedi"
            )

            st.write(
                f"**Analiz:** {analysis_status}"
            )

            st.caption(
                f"Son işlem: "
                f"{format_history_date(document.updated_at)}"
            )

            open_button = st.button(
                "📂 Dokümanı Aç",
                key=(
                    f"open_history_"
                    f"{document.document_id}"
                ),
                use_container_width=True,
            )

            if open_button:
                open_history_document(
                    document.document_id
                )

            delete_button = st.button(
                "🗑️ Geçmişten Sil",
                key=(
                    f"delete_history_"
                    f"{document.document_id}"
                ),
                use_container_width=True,
            )

            if delete_button:
                remove_history_document(
                    document.document_id
                )


def sidebar() -> None:
    """
    Sol menüyü oluşturur.
    """

    with st.sidebar:
        st.title("📄 TSE Doküman Asistanı")

        st.caption(
            "Yapay Zekâ Destekli Doküman Analiz "
            "ve Soru-Cevap Sistemi"
        )

        st.divider()
        st.subheader("Uygulama")

        analysis_mode_button = st.button(
            "📄 Doküman Analizi",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.app_mode == "analysis"
                else "secondary"
            ),
        )

        if analysis_mode_button:
            st.session_state.app_mode = "analysis"
            st.rerun()

        comparison_mode_button = st.button(
            "📑 Doküman Karşılaştırma",
            use_container_width=True,
            type=(
                "primary"
                if st.session_state.app_mode == "comparison"
                else "secondary"
            ),
        )

        if comparison_mode_button:
            st.session_state.app_mode = "comparison"
            st.rerun()

        st.divider()
        st.subheader("Analiz kapsamı")

        st.markdown(
            """
            - Yönetici özeti
            - Standart referansları
            - Zorunlu hükümler
            - Gerekli belgeler
            - Test ve kontrol kriterleri
            - Riskli hükümler
            - Tarihler ve süreler
            - Sorumlu taraflar
            - Kontrol listesi
            - Dokümana soru sorma
            """
        )

        if st.session_state.pdf_result is not None:
            result = st.session_state.pdf_result

            st.divider()
            st.subheader("Açık doküman")

            st.write(
                f"**Dosya:** {result.file_name}"
            )

            st.write(
                f"**Sayfa:** {result.page_count}"
            )

            if st.session_state.document_type:
                st.write(
                    f"**Tür:** "
                    f"{st.session_state.document_type}"
                )

            if st.session_state.analysis_result is not None:
                st.success("✅ Analiz tamamlandı")

            elif st.session_state.vector_store_result is not None:
                st.info("Soru-cevap sistemine hazır")

            else:
                st.info("Doküman okundu")

            if st.session_state.question_history:
                st.write(
                    f"**Kaydedilen soru:** "
                    f"{len(st.session_state.question_history)}"
                )

            close_document_button = st.button(
                "Yeni Doküman Aç",
                key="close_current_document",
                use_container_width=True,
            )

            if close_document_button:
                reset_document_state()
                st.rerun()

        display_recent_documents()

        st.divider()

        st.info(
            "Bu uygulama karar destek amacıyla geliştirilmiş "
            "bir prototiptir. Resmî TSE değerlendirmesi veya "
            "uzman görüşü yerine geçmez."
        )


def header() -> None:
    """
    Ana başlık alanını oluşturur.
    """

    st.title(
        "TSE Akıllı Doküman Analiz ve Soru-Cevap Asistanı"
    )

    st.markdown(
        """
        PDF dokümanınızı yükleyin, önemli hükümleri analiz edin
        ve belge hakkında sorular sorun.
        """
    )

    st.divider()


def display_document_information() -> None:
    """
    Temel doküman bilgilerini gösterir.
    """

    result = st.session_state.pdf_result

    if result is None:
        return

    st.subheader("Doküman bilgileri")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sayfa sayısı",
            result.page_count,
        )

    with col2:
        st.metric(
            "Doküman türü",
            st.session_state.document_type,
        )

    with col3:
        text_status = (
            "Okunabilir"
            if result.empty_page_count == 0
            else "Kısmen okunabilir"
        )

        st.metric(
            "Metin durumu",
            text_status,
        )

    with col4:
        if st.session_state.analysis_result is not None:
            status = "Tamamlandı"

        elif st.session_state.vector_store_result is not None:
            status = "Analize hazır"

        else:
            status = "Bekliyor"

        st.metric(
            "Analiz durumu",
            status,
        )

    st.caption(
        f"Yüklenen dosya: {result.file_name}"
    )


def display_source_verification(
    page_number: int,
    source_text: str,
) -> None:
    """Analiz kaynağını gerçek PDF sayfa metniyle karşılaştırır."""

    pdf_result = st.session_state.pdf_result

    if pdf_result is None:
        return

    verification = verify_source_text(
        page_texts=pdf_result.page_texts,
        page_number=page_number,
        source_text=source_text,
    )

    if verification.status == "Doğrulandı":
        st.caption("✅ Kaynak doğrulandı")
    elif verification.status == "Büyük ölçüde doğrulandı":
        st.caption("🟡 Kaynak büyük ölçüde doğrulandı")
    else:
        st.caption("⚠️ Kaynak manuel olarak incelenmeli")

    with st.expander("Kaynak doğrulama ayrıntısı"):
        st.write(verification.message)
        st.progress(min(max(verification.score, 0.0), 1.0))
        st.caption(f"Metin eşleşme puanı: %{verification.score * 100:.0f}")


def display_source_finding(
    finding,
) -> None:
    """
    Genel analiz bulgusunu gösterir.
    """

    st.markdown(
        f"#### {finding.title}"
    )

    st.write(
        finding.explanation
    )

    st.caption(
        f"Kaynak: PDF sayfa {finding.page_number}"
    )

    display_source_verification(
        page_number=finding.page_number,
        source_text=finding.source_text,
    )

    with st.expander(
        "Kaynak metni görüntüle"
    ):
        st.write(
            finding.source_text
        )


def display_empty_result(
    message: str,
) -> None:
    """
    Boş analiz kategorisini gösterir.
    """

    st.info(message)



def display_analysis_statistics(
    analysis: DocumentAnalysis,
) -> None:
    """
    Analiz sonuçlarını kullanıcı dostu özet kartlarıyla gösterir.
    """

    statistics = calculate_analysis_statistics(
        analysis=analysis
    )

    st.subheader("Analiz özeti")

    first_row_col1, first_row_col2, first_row_col3 = st.columns(3)

    with first_row_col1:
        st.metric(
            "Standart referansı",
            statistics.standard_count,
            help=(
                "Dokümanda açıkça tespit edilen standart "
                "referanslarının sayısı."
            ),
        )

    with first_row_col2:
        st.metric(
            "Zorunlu hüküm",
            statistics.mandatory_item_count,
            help=(
                "Uyulması zorunlu olduğu değerlendirilen "
                "hükümlerin sayısı."
            ),
        )

    with first_row_col3:
        st.metric(
            "Gerekli belge",
            statistics.required_document_count,
            help=(
                "Dokümanda istenen belge, rehber, rapor "
                "ve sertifikaların sayısı."
            ),
        )

    second_row_col1, second_row_col2, second_row_col3 = st.columns(3)

    with second_row_col1:
        st.metric(
            "Riskli hüküm",
            statistics.risk_count,
            help="Dokümanda tespit edilen toplam risk sayısı.",
        )

        if statistics.critical_risk_count > 0:
            st.caption(
                f"🔴 Kritik: {statistics.critical_risk_count}"
            )

        if statistics.high_risk_count > 0:
            st.caption(
                f"🟠 Yüksek: {statistics.high_risk_count}"
            )

    with second_row_col2:
        st.metric(
            "Tarih ve süre",
            statistics.date_count,
            help=(
                "Dokümanda tespit edilen tarih, süre, "
                "geçerlilik ve revizyon bilgileri."
            ),
        )

    with second_row_col3:
        st.metric(
            "Kontrol maddesi",
            statistics.checklist_count,
            help=(
                "Dokümandaki yükümlülüklere göre oluşturulan "
                "kontrol listesi maddeleri."
            ),
        )

    with st.expander("Diğer analiz istatistikleri"):
        detail_col1, detail_col2 = st.columns(2)

        with detail_col1:
            st.metric(
                "Test ve kontrol kriteri",
                statistics.test_criteria_count,
            )

        with detail_col2:
            st.metric(
                "Sorumlu taraf",
                statistics.responsibility_count,
            )


def display_report_download(
    analysis: DocumentAnalysis,
) -> None:
    """Analiz sonuçlarını indirilebilir PDF raporuna dönüştürür."""

    pdf_result = st.session_state.pdf_result

    if pdf_result is None:
        return

    try:
        report_bytes = generate_analysis_report(
            analysis=analysis,
            file_name=pdf_result.file_name,
            document_type=(
                st.session_state.document_type or "Belirtilmedi"
            ),
            page_count=pdf_result.page_count,
            question_history=st.session_state.question_history,
        )

        st.download_button(
            label="📥 PDF Analiz Raporunu İndir",
            data=report_bytes,
            file_name=create_report_filename(
                pdf_result.file_name
            ),
            mime="application/pdf",
            use_container_width=True,
            key=(
                "download_analysis_report_"
                f"{st.session_state.current_document_id}"
            ),
        )

        st.caption(
            "Rapor; analiz özeti, bulgular, riskler, kontrol "
            "listesi ve kayıtlı soru-cevap geçmişini içerir."
        )

    except ReportGeneratorError as exc:
        st.error(str(exc))


def display_analysis_results(
    analysis: DocumentAnalysis,
) -> None:
    """
    Analiz sonuçlarını sekmelerde gösterir.
    """

    st.divider()
    st.subheader("Analiz sonuçları")

    st.success(
        "Doküman analizi başarıyla tamamlandı."
    )

    display_analysis_statistics(
        analysis=analysis
    )

    display_report_download(
        analysis=analysis
    )

    st.divider()

    summary_tab, requirements_tab, risks_tab, details_tab = st.tabs(
        [
            "📋 Genel Bakış",
            "📌 Hükümler ve Belgeler",
            "⚠️ Riskler",
            "📅 Tarihler ve Sorumluluklar",
        ]
    )

    with summary_tab:
        st.markdown("### Doküman başlığı")
        st.write(analysis.document_title)

        st.markdown("### Dokümanın amacı")
        st.write(analysis.document_purpose)

        st.markdown("### Yönetici özeti")
        st.write(analysis.executive_summary)

        st.divider()
        st.markdown("### Standart referansları")

        if analysis.standards:
            for standard in analysis.standards:
                display_source_finding(standard)
                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir standart "
                "referansı tespit edilmedi."
            )

    with requirements_tab:
        st.markdown("### Zorunlu hükümler")

        if analysis.mandatory_items:
            for item in analysis.mandatory_items:
                display_source_finding(item)
                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir zorunlu "
                "hüküm tespit edilmedi."
            )

        st.markdown("### Gerekli belgeler")

        if analysis.required_documents:
            for item in analysis.required_documents:
                display_source_finding(item)
                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir belge "
                "gereksinimi tespit edilmedi."
            )

        st.markdown("### Test ve kontrol kriterleri")

        if analysis.test_criteria:
            for item in analysis.test_criteria:
                display_source_finding(item)
                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir test veya kontrol "
                "kriteri tespit edilmedi."
            )

    with risks_tab:
        if analysis.risks:
            sorted_risks = sorted(
                analysis.risks,
                key=lambda risk: {
                    "Kritik": 0,
                    "Yüksek": 1,
                    "Orta": 2,
                    "Düşük": 3,
                }.get(
                    risk.level,
                    4,
                ),
            )

            for risk in sorted_risks:
                level_icon = {
                    "Kritik": "🔴",
                    "Yüksek": "🟠",
                    "Orta": "🟡",
                    "Düşük": "🟢",
                }.get(
                    risk.level,
                    "⚪",
                )

                st.markdown(
                    f"### {level_icon} {risk.title}"
                )

                st.write(
                    f"**Risk seviyesi:** {risk.level}"
                )

                st.write(
                    risk.explanation
                )

                st.caption(
                    f"Kaynak: PDF sayfa "
                    f"{risk.page_number}"
                )

                display_source_verification(
                    page_number=risk.page_number,
                    source_text=risk.source_text,
                )

                with st.expander(
                    "Kaynak metni görüntüle"
                ):
                    st.write(
                        risk.source_text
                    )

                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir riskli "
                "hüküm tespit edilmedi."
            )

    with details_tab:
        st.markdown("### Tarihler ve süreler")

        if analysis.dates:
            for date_item in analysis.dates:
                st.markdown(
                    f"#### 📅 {date_item.date_or_period}"
                )

                st.write(
                    date_item.purpose
                )

                st.caption(
                    f"Kaynak: PDF sayfa "
                    f"{date_item.page_number}"
                )

                display_source_verification(
                    page_number=date_item.page_number,
                    source_text=date_item.source_text,
                )

                with st.expander(
                    "Kaynak metni görüntüle"
                ):
                    st.write(
                        date_item.source_text
                    )

                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir tarih veya "
                "süre tespit edilmedi."
            )

        st.markdown("### Sorumlu taraflar")

        if analysis.responsibilities:
            for responsibility in analysis.responsibilities:
                st.markdown(
                    f"#### 👤 {responsibility.party}"
                )

                st.write(
                    responsibility.responsibility
                )

                st.caption(
                    f"Kaynak: PDF sayfa "
                    f"{responsibility.page_number}"
                )

                display_source_verification(
                    page_number=responsibility.page_number,
                    source_text=responsibility.source_text,
                )

                with st.expander(
                    "Kaynak metni görüntüle"
                ):
                    st.write(
                        responsibility.source_text
                    )

                st.divider()

        else:
            display_empty_result(
                "Dokümanda açık bir sorumlu "
                "taraf tespit edilmedi."
            )

        st.markdown("### Kontrol listesi")

        if analysis.checklist:
            for index, item in enumerate(
                analysis.checklist,
                start=1,
            ):
                checked = st.checkbox(
                    item.task,
                    key=(
                        f"checklist_"
                        f"{st.session_state.current_document_id}_"
                        f"{index}_"
                        f"{item.page_number}"
                    ),
                )

                priority_text = (
                    f"Öncelik: {item.priority} · "
                    f"Kaynak: PDF sayfa "
                    f"{item.page_number}"
                )

                if checked:
                    st.caption(
                        f"✅ Tamamlandı · {priority_text}"
                    )

                else:
                    st.caption(priority_text)

                with st.expander(
                    f"{index}. madde kaynak metni"
                ):
                    st.write(
                        item.source_text
                    )

        else:
            display_empty_result(
                "Dokümandan kontrol listesi oluşturulamadı."
            )



def highlight_search_term(
    text: str,
    query: str,
) -> str:
    """
    Aranan ifadeyi güvenli HTML kullanarak görünür biçimde vurgular.
    """

    import html
    import re

    escaped_text = html.escape(text)
    escaped_query = html.escape(query.strip())

    if not escaped_query:
        return escaped_text

    pattern = re.compile(
        re.escape(escaped_query),
        flags=re.IGNORECASE,
    )

    return pattern.sub(
        lambda match: (
            "<mark style='padding: 0.08rem 0.22rem; "
            "border-radius: 0.22rem;'>"
            f"{match.group(0)}"
            "</mark>"
        ),
        escaped_text,
    )


def display_keyword_search_results(
    query: str,
    results: list[KeywordSearchResult],
) -> None:
    """Anahtar kelime sonuçlarını metin ve gerçek PDF sayfasıyla gösterir."""

    if not results:
        st.warning(
            f'“{query}” ifadesi yüklenen dokümanda bulunamadı.'
        )
        return

    total_occurrences = sum(
        result.occurrence_count
        for result in results
    )

    page_numbers = [
        result.page_number
        for result in results
    ]

    st.markdown("### Arama sonuçları")
    st.success(
        f'“{query}” ifadesi {len(results)} sayfada '
        f'toplam {total_occurrences} kez bulundu.'
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:
        st.metric(
            "Eşleşme sayısı",
            total_occurrences,
        )

    with summary_col2:
        st.metric(
            "Bulunan sayfa",
            len(results),
        )

    formatted_pages = ", ".join(
        str(page_number)
        for page_number in page_numbers
    )

    st.caption(
        f"Eşleşen PDF sayfaları: {formatted_pages}"
    )

    for result in results:
        preview_is_selected = (
            st.session_state.keyword_preview_page
            == result.page_number
            and st.session_state.keyword_preview_query
            == query
        )

        expander_title = (
            f"📄 Sayfa {result.page_number} · "
            f"{result.occurrence_count} eşleşme"
        )

        with st.expander(
            expander_title,
            expanded=(
                result.page_number == page_numbers[0]
                or preview_is_selected
            ),
        ):
            st.caption(
                "Aşağıdaki kısa bölümler kelimenin geçtiği "
                "yerleri hızlıca gösterir."
            )

            for snippet_index, snippet in enumerate(
                result.snippets,
                start=1,
            ):
                highlighted_snippet = highlight_search_term(
                    text=snippet,
                    query=query,
                )

                st.markdown(
                    (
                        "<div style='padding: 0.85rem 1rem; "
                        "border: 1px solid rgba(128,128,128,0.25); "
                        "border-radius: 0.55rem; "
                        "line-height: 1.65; margin-bottom: 0.6rem;'>"
                        f"{highlighted_snippet}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

            preview_button = st.button(
                "📄 Bu sayfayı PDF üzerinde görüntüle",
                key=(
                    f"preview_pdf_page_"
                    f"{st.session_state.current_document_id}_"
                    f"{result.page_number}_"
                    f"{query}"
                ),
                use_container_width=True,
            )

            if preview_button:
                st.session_state.keyword_preview_page = (
                    result.page_number
                )
                st.session_state.keyword_preview_query = query
                st.rerun()

            if preview_is_selected:
                st.markdown(
                    f"#### PDF sayfa {result.page_number} ön izlemesi"
                )

                try:
                    preview_result = render_highlighted_pdf_page(
                        file_bytes=(
                            st.session_state.uploaded_file_bytes
                        ),
                        page_number=result.page_number,
                        query=query,
                    )

                    if preview_result.highlighted_count > 0:
                        st.caption(
                            f"Sayfa üzerinde "
                            f"{preview_result.highlighted_count} "
                            "eşleşme vurgulandı."
                        )
                    else:
                        st.warning(
                            "Metin aramasında sonuç bulundu ancak PDF "
                            "sayfa görüntüsünde otomatik vurgulama "
                            "yapılamadı. Sayfa yine de aşağıda gösteriliyor."
                        )

                    st.image(
                        preview_result.image_bytes,
                        caption=(
                            f"PDF sayfa {result.page_number} — "
                            f'“{query}” eşleşmeleri'
                        ),
                        use_container_width=True,
                    )

                    close_preview_button = st.button(
                        "Ön izlemeyi kapat",
                        key=(
                            f"close_pdf_preview_"
                            f"{result.page_number}_"
                            f"{query}"
                        ),
                        use_container_width=True,
                    )

                    if close_preview_button:
                        st.session_state.keyword_preview_page = None
                        st.session_state.keyword_preview_query = ""
                        st.rerun()

                except PagePreviewError as exc:
                    st.error(str(exc))


def display_keyword_search_section() -> None:
    """
    PDF metninde doğrudan anahtar kelime ve ifade araması yapar.
    Bu işlem Gemini API kullanmaz.
    """

    result = st.session_state.pdf_result

    if result is None:
        return

    st.divider()
    st.subheader("3. Doküman içinde arama yapın")

    st.write(
        "Bir kelime veya ifade yazın; geçtiği sayfaları ve "
        "eşleşmenin çevresindeki gerçek PDF metnini görüntüleyin."
    )

    with st.form(
        "keyword_search_form",
        clear_on_submit=False,
    ):
        query = st.text_input(
            "Aranacak kelime veya ifade",
            value=st.session_state.keyword_search_query,
            placeholder=(
                "Örnek: ISO, logo, askıya alma veya FSSC 22000"
            ),
        )

        search_button = st.form_submit_button(
            "🔍 Dokümanda Ara",
            type="primary",
            use_container_width=True,
        )

    if search_button:
        try:
            search_results = search_document_pages(
                page_texts=result.page_texts,
                query=query,
                maximum_results=30,
                maximum_snippets_per_page=3,
            )

            st.session_state.keyword_search_query = query.strip()
            st.session_state.keyword_search_results = search_results
            st.session_state.keyword_preview_page = None
            st.session_state.keyword_preview_query = ""

        except KeywordSearchError as exc:
            st.error(str(exc))
            st.session_state.keyword_search_results = []
            st.session_state.keyword_preview_page = None
            st.session_state.keyword_preview_query = ""

    if st.session_state.keyword_search_query:
        clear_col1, clear_col2 = st.columns([4, 1])

        with clear_col2:
            clear_search_button = st.button(
                "Aramayı Temizle",
                use_container_width=True,
                key="clear_keyword_search",
            )

        if clear_search_button:
            st.session_state.keyword_search_query = ""
            st.session_state.keyword_search_results = []
            st.session_state.keyword_preview_page = None
            st.session_state.keyword_preview_query = ""
            st.rerun()

        display_keyword_search_results(
            query=st.session_state.keyword_search_query,
            results=st.session_state.keyword_search_results,
        )


def display_answer_sources(
    result: QuestionAnswerResult,
) -> None:
    """
    Soru cevabının kaynaklarını gösterir.
    """

    if not result.answer_found or not result.sources:
        return

    page_numbers = sorted(
        {
            source.page_number
            for source in result.sources
            if source.page_number > 0
        }
    )

    if page_numbers:
        formatted_pages = ", ".join(
            str(page_number)
            for page_number in page_numbers
        )

        st.caption(
            f"İlgili PDF sayfaları: {formatted_pages}"
        )

    with st.expander(
        f"Kaynakları görüntüle "
        f"({len(result.sources)} bölüm)"
    ):
        for index, source in enumerate(
            result.sources,
            start=1,
        ):
            st.markdown(
                f"**Kaynak {index} — "
                f"PDF sayfa {source.page_number}**"
            )

            st.write(
                source.text
            )

            if index < len(result.sources):
                st.divider()


def display_question_history() -> None:
    """
    Soru-cevap geçmişini gösterir.
    """

    if not st.session_state.question_history:
        return

    st.markdown("### Soru geçmişi")

    for result in reversed(
        st.session_state.question_history
    ):
        with st.container(border=True):
            st.markdown(
                f"**Soru:** {result.question}"
            )

            if result.answer_found:
                st.markdown("**Cevap:**")
                st.write(result.answer)

                display_answer_sources(result)

            else:
                st.warning(result.answer)


def prepare_vector_store() -> None:
    """
    Açık doküman için vektör arama sistemini hazırlar.
    """

    api_key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    ).strip()

    embedding_model = os.getenv(
        "GEMINI_EMBEDDING_MODEL",
        "gemini-embedding-001",
    ).strip()

    if not api_key:
        st.error(
            "Gemini API anahtarı bulunamadı."
        )
        return

    try:
        with st.spinner(
            "Doküman soru-cevap sistemine hazırlanıyor..."
        ):
            vector_store, vector_store_result = (
                create_vector_store(
                    chunks=(
                        st.session_state.document_chunks
                    ),
                    file_name=(
                        st.session_state.pdf_result.file_name
                    ),
                    document_type=(
                        st.session_state.document_type
                    ),
                    file_bytes=(
                        st.session_state.uploaded_file_bytes
                    ),
                    api_key=api_key,
                    embedding_model=embedding_model,
                )
            )

            st.session_state.vector_store = vector_store
            st.session_state.vector_store_result = (
                vector_store_result
            )

        st.rerun()

    except VectorStoreError as exc:
        st.error(str(exc))

def process_document_question(
    question: str,
) -> None:
    """
    Yazılan veya önerilerden seçilen soruyu cevaplar
    ve sonucu kalıcı geçmişe kaydeder.
    """

    api_key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    ).strip()

    model_name = os.getenv(
        "GEMINI_CHAT_MODEL",
        "gemini-flash-latest",
    ).strip()

    if not api_key:
        st.error(
            "Gemini API anahtarı bulunamadı. "
            ".env dosyasındaki GOOGLE_API_KEY "
            "alanını kontrol edin."
        )
        return

    if st.session_state.vector_store is None:
        st.error(
            "Dokümanın soru-cevap sistemi hazır değil."
        )
        return

    try:
        with st.spinner(
            "Dokümanda ilgili bölümler aranıyor "
            "ve cevap hazırlanıyor..."
        ):
            answer_result = answer_document_question(
                question=question,
                vector_store=st.session_state.vector_store,
                api_key=api_key,
                model_name=model_name,
                result_count=5,
            )

            save_question(
                document_id=(
                    st.session_state.current_document_id
                ),
                result=answer_result,
            )

            st.session_state.question_history.append(
                answer_result
            )

        st.rerun()

    except QuestionAnswerError as exc:
        st.error(str(exc))

    except HistoryManagerError as exc:
        st.error(
            "Cevap oluşturuldu ancak geçmişe "
            f"kaydedilemedi: {exc}"
        )

    except Exception as exc:
        st.error(
            f"Soru cevaplanırken beklenmeyen "
            f"bir hata oluştu: {exc}"
        )

def display_question_answer_section() -> None:
    """
    Dokümana soru sorma, önerilen sorular ve
    kalıcı soru geçmişi bölümünü gösterir.
    """

    st.divider()
    st.subheader("4. Dokümana soru sorun")

    if st.session_state.vector_store is None:
        st.info(
            "Yeni soru sorabilmek için dokümanın "
            "arama sistemini hazırlayın."
        )

        prepare_button = st.button(
            "Soru-Cevap Sistemini Hazırla",
            type="primary",
            use_container_width=True,
        )

        if prepare_button:
            prepare_vector_store()

        display_question_history()
        return

    st.write(
        "Sorular yalnızca yüklenen PDF içeriğine göre cevaplanır."
    )

    analysis = st.session_state.analysis_result

    suggested_questions = get_suggested_questions(
        analysis=analysis
    )

    if suggested_questions:
        st.markdown("### Önerilen sorular")

        st.caption(
            "Hazır sorulardan birine basarak dokümanı "
            "hızlıca inceleyebilirsiniz."
        )

        for row_start in range(
            0,
            len(suggested_questions),
            2,
        ):
            question_columns = st.columns(2)

            row_questions = suggested_questions[
                row_start:row_start + 2
            ]

            for column_index, question in enumerate(
                row_questions
            ):
                with question_columns[column_index]:
                    suggestion_button = st.button(
                        question,
                        key=(
                            f"suggested_question_"
                            f"{row_start}_"
                            f"{column_index}_"
                            f"{st.session_state.current_document_id}"
                        ),
                        use_container_width=True,
                    )

                    if suggestion_button:
                        process_document_question(
                            question=question
                        )

        st.divider()

    st.markdown("### Kendi sorunuzu yazın")

    with st.form(
        "document_question_form",
        clear_on_submit=True,
    ):
        question = st.text_input(
            "Sorunuz",
            placeholder=(
                "Örnek: Belge askıya alınırsa "
                "marka kullanımına ne olur?"
            ),
        )

        ask_button = st.form_submit_button(
            "🔎 Soruyu Cevapla",
            type="primary",
            use_container_width=True,
        )

    if ask_button:
        process_document_question(
            question=question
        )

    if st.session_state.question_history:
        clear_col1, clear_col2 = st.columns(
            [4, 1]
        )

        with clear_col2:
            clear_button = st.button(
                "Soru Geçmişini Temizle",
                use_container_width=True,
            )

        if clear_button:
            try:
                clear_document_questions(
                    document_id=(
                        st.session_state.current_document_id
                    )
                )

                st.session_state.question_history = []

                st.rerun()

            except HistoryManagerError as exc:
                st.error(str(exc))

        display_question_history()


def analyze_current_document() -> None:
    """
    Açık dokümanı Gemini ile analiz eder.
    """

    api_key = os.getenv(
        "GOOGLE_API_KEY",
        "",
    ).strip()

    model_name = os.getenv(
        "GEMINI_CHAT_MODEL",
        "gemini-flash-latest",
    ).strip()

    if not api_key:
        st.error(
            "Gemini API anahtarı bulunamadı."
        )
        return

    try:
        if st.session_state.vector_store is None:
            prepare_vector_store()

            return

        with st.spinner(
            "Gemini dokümanı analiz ediyor..."
        ):
            analysis_result = analyze_document(
                pdf_result=(
                    st.session_state.pdf_result
                ),
                document_type=(
                    st.session_state.document_type
                ),
                api_key=api_key,
                model_name=model_name,
            )

            save_analysis(
                document_id=(
                    st.session_state.current_document_id
                ),
                analysis_json=(
                    analysis_result.model_dump_json()
                ),
            )

            st.session_state.analysis_result = (
                analysis_result
            )

        st.rerun()

    except (
        AnalysisEngineError,
        HistoryManagerError,
    ) as exc:
        st.error(str(exc))




def _normalize_diff_text(text: str) -> str:
    """Karşılaştırma metnini güvenli ve okunabilir hâle getirir."""

    return " ".join((text or "").split())


def calculate_text_similarity(
    old_text: str,
    new_text: str,
) -> float:
    """İki kaynak metnin kelime dizilimine göre benzerliğini hesaplar."""

    old_words = _normalize_diff_text(old_text).lower().split()
    new_words = _normalize_diff_text(new_text).lower().split()

    if not old_words and not new_words:
        return 1.0

    if not old_words or not new_words:
        return 0.0

    return SequenceMatcher(
        None,
        old_words,
        new_words,
        autojunk=False,
    ).ratio()


def classify_text_change(
    old_text: str,
    new_text: str,
    similarity: float,
) -> tuple[str, str]:
    """Benzerlik oranına göre kullanıcı dostu değişiklik etiketi döndürür."""

    old_clean = _normalize_diff_text(old_text)
    new_clean = _normalize_diff_text(new_text)

    if not old_clean and new_clean:
        return "Tamamen yeni madde", "new"

    if old_clean and not new_clean:
        return "Tamamen kaldırılan madde", "removed"

    if similarity >= 0.90:
        return "Küçük düzenleme", "minor"

    if similarity >= 0.65:
        return "Kısmi değişiklik", "partial"

    if similarity >= 0.35:
        return "Büyük ölçüde yeniden yazıldı", "major"

    return "İçerik önemli ölçüde değiştirildi", "critical"


def _build_word_diff_html(
    old_text: str,
    new_text: str,
) -> tuple[str, str]:
    """Eski ve yeni metni kelime düzeyinde renklendirir."""

    old_words = _normalize_diff_text(old_text).split()
    new_words = _normalize_diff_text(new_text).split()

    matcher = SequenceMatcher(
        None,
        old_words,
        new_words,
        autojunk=False,
    )

    old_parts: list[str] = []
    new_parts: list[str] = []

    for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        old_segment = " ".join(old_words[old_start:old_end])
        new_segment = " ".join(new_words[new_start:new_end])

        if tag == "equal":
            escaped_old = html.escape(old_segment)
            escaped_new = html.escape(new_segment)
            if escaped_old:
                old_parts.append(escaped_old)
            if escaped_new:
                new_parts.append(escaped_new)

        elif tag == "delete":
            if old_segment:
                old_parts.append(
                    '<span class="diff-delete">'
                    f'{html.escape(old_segment)}'
                    '</span>'
                )

        elif tag == "insert":
            if new_segment:
                new_parts.append(
                    '<span class="diff-insert">'
                    f'{html.escape(new_segment)}'
                    '</span>'
                )

        elif tag == "replace":
            if old_segment:
                old_parts.append(
                    '<span class="diff-delete">'
                    f'{html.escape(old_segment)}'
                    '</span>'
                )
            if new_segment:
                new_parts.append(
                    '<span class="diff-insert">'
                    f'{html.escape(new_segment)}'
                    '</span>'
                )

    return " ".join(old_parts), " ".join(new_parts)


def display_text_diff(
    old_text: str,
    new_text: str,
) -> None:
    """Eski ve yeni kaynak metni yan yana fark görünümünde gösterir."""

    old_clean = _normalize_diff_text(old_text)
    new_clean = _normalize_diff_text(new_text)

    if not old_clean and not new_clean:
        st.info("Karşılaştırılabilir kaynak metni bulunamadı.")
        return

    similarity = calculate_text_similarity(
        old_text=old_clean,
        new_text=new_clean,
    )

    change_label, change_class = classify_text_change(
        old_text=old_clean,
        new_text=new_clean,
        similarity=similarity,
    )

    old_html, new_html = _build_word_diff_html(
        old_text=old_clean,
        new_text=new_clean,
    )

    if not old_clean:
        old_html = '<span class="diff-empty">Eski dokümanda bulunmuyor.</span>'
    if not new_clean:
        new_html = '<span class="diff-empty">Yeni dokümanda bulunmuyor.</span>'

    st.markdown(
        """
        <style>
        .diff-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 4px 0 14px 0;
        }
        .diff-badge {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 6px 11px;
            font-size: 0.88rem;
            font-weight: 700;
            border: 1px solid transparent;
        }
        .diff-badge.similarity {
            background: #eef2ff;
            border-color: #c7d2fe;
            color: #3730a3;
        }
        .diff-badge.new {
            background: #dcfce7;
            border-color: #86efac;
            color: #166534;
        }
        .diff-badge.removed {
            background: #fee2e2;
            border-color: #fca5a5;
            color: #991b1b;
        }
        .diff-badge.minor {
            background: #ecfeff;
            border-color: #a5f3fc;
            color: #155e75;
        }
        .diff-badge.partial {
            background: #fef9c3;
            border-color: #fde047;
            color: #854d0e;
        }
        .diff-badge.major {
            background: #ffedd5;
            border-color: #fdba74;
            color: #9a3412;
        }
        .diff-badge.critical {
            background: #ffe4e6;
            border-color: #fda4af;
            color: #9f1239;
        }
        .diff-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-top: 8px;
        }
        .diff-panel {
            border: 1px solid #d8dee8;
            border-radius: 10px;
            padding: 14px;
            background: #ffffff;
            min-height: 120px;
        }
        .diff-panel-title {
            font-weight: 700;
            margin-bottom: 10px;
            color: #1f2937;
        }
        .diff-text {
            line-height: 1.75;
            overflow-wrap: anywhere;
        }
        .diff-delete {
            background: #ffe2e2;
            color: #9f1d1d;
            text-decoration: line-through;
            border-radius: 4px;
            padding: 2px 4px;
        }
        .diff-insert {
            background: #dcfce7;
            color: #166534;
            border-radius: 4px;
            padding: 2px 4px;
        }
        .diff-empty {
            color: #6b7280;
            font-style: italic;
        }
        @media (max-width: 850px) {
            .diff-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    similarity_percentage = round(similarity * 100)

    st.markdown(
        f"""
        <div class="diff-summary">
            <span class="diff-badge {change_class}">{html.escape(change_label)}</span>
            <span class="diff-badge similarity">Metin benzerliği: %{similarity_percentage}</span>
        </div>
        <div class="diff-grid">
            <div class="diff-panel">
                <div class="diff-panel-title">➖ Eski metin</div>
                <div class="diff-text">{old_html}</div>
            </div>
            <div class="diff-panel">
                <div class="diff-panel-title">➕ Yeni metin</div>
                <div class="diff-text">{new_html}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def normalize_search_text(value: str) -> str:
    """Arama sırasında Türkçe karakter ve büyük/küçük harf farklarını azaltır."""
    normalized = unicodedata.normalize("NFKD", value or "")
    return "".join(
        character for character in normalized
        if not unicodedata.combining(character)
    ).casefold()


def filter_comparison_findings(
    findings: list[ComparisonFinding],
    search_query: str,
) -> list[ComparisonFinding]:
    """Karşılaştırma bulgularını başlık, açıklama ve kaynak metne göre süzer."""
    normalized_query = normalize_search_text(search_query.strip())
    if not normalized_query:
        return findings

    filtered: list[ComparisonFinding] = []
    for finding in findings:
        searchable_text = " ".join(
            [
                finding.title or "",
                finding.explanation or "",
                finding.old_source_text or "",
                finding.new_source_text or "",
            ]
        )
        if normalized_query in normalize_search_text(searchable_text):
            filtered.append(finding)
    return filtered


def filter_risk_changes(
    risk_changes: list[RiskChange],
    search_query: str,
) -> list[RiskChange]:
    """Risk değişikliklerini kullanıcı aramasına göre süzer."""
    normalized_query = normalize_search_text(search_query.strip())
    if not normalized_query:
        return risk_changes

    filtered: list[RiskChange] = []
    for risk in risk_changes:
        searchable_text = " ".join(
            [
                risk.title or "",
                risk.explanation or "",
                risk.change_type or "",
                risk.risk_level or "",
                risk.source_text or "",
            ]
        )
        if normalized_query in normalize_search_text(searchable_text):
            filtered.append(risk)
    return filtered


def clean_comparison_summary(summary: str) -> str:
    """Son kullanıcıya yönelik olmayan teknik ifadeleri özet metninden kaldırır."""
    cleaned = summary or ""
    replacements = {
        "Yerel karşılaştırma sonucunda": "Doküman karşılaştırması sonucunda",
        "yerel karşılaştırma sonucunda": "doküman karşılaştırması sonucunda",
        "tamamen yerel olarak": "",
        "Gemini API kotası veya internet bağlantısı kullanılmadı.": "",
    }
    for old_text, new_text in replacements.items():
        cleaned = cleaned.replace(old_text, new_text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def format_finding_title(title: str, change_label: str) -> str:
    """Kart başlıklarını daha kurumsal ve tutarlı bir biçimde gösterir."""
    cleaned_title = (title or "").strip()
    prefix_patterns = [
        r"^Yeni madde\s*:\s*",
        r"^Eklenen madde\s*:\s*",
        r"^Kaldırılan madde\s*:\s*",
        r"^Değiştirilen madde\s*:\s*",
        r"^Tarih/süre değişikliği\s*:\s*",
        r"^Sorumluluk değişikliği\s*:\s*",
        r"^Standart değişikliği\s*:\s*",
    ]
    for pattern in prefix_patterns:
        cleaned_title = re.sub(pattern, "", cleaned_title, flags=re.IGNORECASE)
    return f"{change_label}: {cleaned_title}" if cleaned_title else change_label



def _build_preview_query(source_text: str) -> str:
    """Kaynak metinden PDF üzerinde aranabilecek kısa bir ifade üretir."""
    cleaned = re.sub(r"\s+", " ", source_text or "").strip()
    if not cleaned:
        return ""

    words = cleaned.split()
    # Uzun cümlelerin PDF aramasında birebir eşleşmeme ihtimalini azaltmak
    # için ilk anlamlı kelime grubunu kullanırız.
    return " ".join(words[:10])


def display_comparison_page_preview(
    *,
    document_side: str,
    page_number: int | None,
    source_text: str,
    finding_key: str,
) -> None:
    """Karşılaştırma bulgusunun ilgili eski/yeni PDF sayfasını gösterir."""
    if not page_number:
        return

    if document_side == "old":
        label = f"📄 Eski dokümanda sayfa {page_number}'i aç"
        file_bytes = st.session_state.get("comparison_old_file_bytes")
        document_label = "Eski doküman"
    else:
        label = f"📄 Yeni dokümanda sayfa {page_number}'i aç"
        file_bytes = st.session_state.get("comparison_new_file_bytes")
        document_label = "Yeni doküman"

    preview_state_key = (
        f"comparison_preview_{document_side}_{finding_key}_{page_number}"
    )

    if st.button(
        label,
        key=f"open_{preview_state_key}",
        use_container_width=True,
    ):
        st.session_state[preview_state_key] = not st.session_state.get(
            preview_state_key,
            False,
        )

    if not st.session_state.get(preview_state_key, False):
        return

    if not file_bytes:
        st.warning(
            "PDF sayfa ön izlemesi için dokümanı yeniden yükleyip "
            "karşılaştırmayı tekrar çalıştırın."
        )
        return

    try:
        preview = render_pdf_page(
            file_bytes=file_bytes,
            page_number=page_number,
        )

        st.caption(f"{document_label} · PDF sayfa {page_number}")
        st.image(
            preview.image_bytes,
            caption=f"{document_label} — PDF sayfa {page_number}",
            use_container_width=True,
        )

    except PagePreviewError as exc:
        st.warning(str(exc))

def display_comparison_finding(
    finding: ComparisonFinding,
    change_label: str,
    finding_key: str,
) -> None:
    """Karşılaştırma bulgusunu kaynaklarıyla gösterir."""

    display_title = format_finding_title(finding.title, change_label)
    st.markdown(f"#### {display_title}")
    st.write(finding.explanation)

    source_parts: list[str] = []
    if finding.old_page_number:
        source_parts.append(
            f"Eski doküman: sayfa {finding.old_page_number}"
        )
    if finding.new_page_number:
        source_parts.append(
            f"Yeni doküman: sayfa {finding.new_page_number}"
        )

    if source_parts:
        st.caption(" · ".join(source_parts))

    if finding.old_source_text or finding.new_source_text:
        with st.expander(f"{change_label} kaynaklarını ve metin farkını görüntüle"):
            display_text_diff(
                old_text=finding.old_source_text,
                new_text=finding.new_source_text,
            )

            preview_columns = st.columns(2)
            with preview_columns[0]:
                display_comparison_page_preview(
                    document_side="old",
                    page_number=finding.old_page_number,
                    source_text=finding.old_source_text,
                    finding_key=finding_key,
                )
            with preview_columns[1]:
                display_comparison_page_preview(
                    document_side="new",
                    page_number=finding.new_page_number,
                    source_text=finding.new_source_text,
                    finding_key=finding_key,
                )

            with st.expander("Ham kaynak metinlerini göster"):
                if finding.old_source_text:
                    st.markdown("**Eski dokümandaki kaynak**")
                    st.write(finding.old_source_text)
                if finding.new_source_text:
                    st.markdown("**Yeni dokümandaki kaynak**")
                    st.write(finding.new_source_text)

    st.divider()
    display_ai_interpretation_control(
        finding=finding,
        finding_key=finding_key,
    )


def display_comparison_list(
    title: str,
    findings: list[ComparisonFinding],
    empty_message: str,
    change_label: str,
) -> None:
    """Karşılaştırma bulgularının bir kategorisini gösterir."""

    st.markdown(f"### {title}")

    if not findings:
        st.info(empty_message)
        return

    for finding_index, finding in enumerate(findings, start=1):
        with st.container(border=True):
            display_comparison_finding(
                finding=finding,
                change_label=change_label,
                finding_key=(
                    f"{change_label}_{finding_index}_"
                    f"{finding.old_page_number}_{finding.new_page_number}"
                ),
            )


def display_risk_changes(
    risk_changes: list[RiskChange],
) -> None:
    """Risk değişikliklerini önem seviyesine göre renkli biçimde gösterir."""
    st.markdown("### Risk değişiklikleri")

    if not risk_changes:
        st.info("Arama ölçütüne uyan bir risk değişikliği bulunamadı.")
        return

    risk_styles = {
        "Kritik": ("🔴", "#fee2e2", "#991b1b", "#fecaca"),
        "Yüksek": ("🟠", "#ffedd5", "#9a3412", "#fed7aa"),
        "Orta": ("🟡", "#fef9c3", "#854d0e", "#fde68a"),
        "Düşük": ("🟢", "#dcfce7", "#166534", "#bbf7d0"),
    }

    for risk in risk_changes:
        icon, background, foreground, border = risk_styles.get(
            risk.risk_level,
            ("⚪", "#f3f4f6", "#374151", "#d1d5db"),
        )
        with st.container(border=True):
            st.markdown(f"#### {icon} {risk.title}")
            badge_html = (
                '<span style="display:inline-block;padding:0.28rem 0.7rem;'
                f'border-radius:999px;background:{background};color:{foreground};'
                f'border:1px solid {border};font-weight:700;margin-bottom:0.35rem;">'
                f'{html.escape(risk.risk_level)} risk</span>'
            )
            st.markdown(badge_html, unsafe_allow_html=True)
            st.write(f"**Değişiklik türü:** {risk.change_type}")
            st.write(risk.explanation)

            pages: list[str] = []
            if risk.old_page_number:
                pages.append(f"Eski doküman: sayfa {risk.old_page_number}")
            if risk.new_page_number:
                pages.append(f"Yeni doküman: sayfa {risk.new_page_number}")
            if pages:
                st.caption(" · ".join(pages))

            if risk.source_text:
                with st.expander("Risk kaynağını görüntüle"):
                    st.write(risk.source_text)


def render_comparison_metric_cards(comparison: DocumentComparison) -> None:
    """Karşılaştırma sayılarını Streamlit'in yerleşik metrikleriyle gösterir."""

    total_changes = (
        len(comparison.added_items)
        + len(comparison.removed_items)
        + len(comparison.changed_items)
    )

    first_row = st.columns(4)
    first_row[0].metric("📊 Toplam değişiklik", total_changes)
    first_row[1].metric("➕ Eklenen", len(comparison.added_items))
    first_row[2].metric("➖ Kaldırılan", len(comparison.removed_items))
    first_row[3].metric("✏️ Değiştirilen", len(comparison.changed_items))

    second_row = st.columns(4)
    second_row[0].metric(
        "📅 Tarih / süre",
        len(comparison.date_and_period_changes),
    )
    second_row[1].metric(
        "👥 Sorumluluk",
        len(comparison.responsibility_changes),
    )
    second_row[2].metric(
        "📚 Standart",
        len(comparison.standard_changes),
    )
    second_row[3].metric("⚠️ Risk", len(comparison.risk_changes))


def display_revision_status(comparison: DocumentComparison) -> None:
    """Değişiklik sayısına göre dokümanın revizyon büyüklüğünü özetler."""

    total_changes = (
        len(comparison.added_items)
        + len(comparison.removed_items)
        + len(comparison.changed_items)
    )

    if total_changes <= 5:
        label = "🟢 Küçük revizyon"
        description = "Dokümanda sınırlı sayıda değişiklik tespit edildi."
    elif total_changes <= 15:
        label = "🟠 Orta kapsamlı revizyon"
        description = (
            "Dokümanda dikkate alınması gereken birden fazla değişiklik "
            "tespit edildi."
        )
    else:
        label = "🔴 Büyük kapsamlı revizyon"
        description = (
            "Dokümanda çok sayıda değişiklik bulundu. İlgili birimlerin "
            "değişiklikleri ayrıntılı incelemesi önerilir."
        )

    with st.container(border=True):
        st.markdown(f"#### {label}")
        st.write(description)


def display_comparison_results(
    comparison: DocumentComparison,
) -> None:
    """Gemini karşılaştırma sonuçlarını sekmelerde gösterir."""

    st.divider()
    st.subheader("Karşılaştırma sonuçları")
    st.success("Doküman karşılaştırması başarıyla tamamlandı.")

    total_changes = (
        len(comparison.added_items)
        + len(comparison.removed_items)
        + len(comparison.changed_items)
    )

    render_comparison_metric_cards(comparison)
    display_revision_status(comparison)

    st.markdown("### Revizyon özeti")
    st.markdown(
        f"""
- **{len(comparison.added_items)}** yeni hüküm eklenmiştir.
- **{len(comparison.removed_items)}** hüküm kaldırılmıştır.
- **{len(comparison.changed_items)}** hüküm güncellenmiştir.
- **{len(comparison.risk_changes)}** risk başlığı etkilenmiştir.
- **{len(comparison.standard_changes)}** standart veya referans değişikliği tespit edilmiştir.
        """
    )
    if comparison.comparison_summary:
        st.info(clean_comparison_summary(comparison.comparison_summary))

    st.markdown("### Revizyon değerlendirmesi")
    st.write(
        "Eski ve yeni doküman karşılaştırılarak eklenen, kaldırılan ve "
        "değiştirilen hükümler kategori bazında belirlenmiştir. "
        "Kaynak metinler ve sayfa bilgileri her değişiklik kartında incelenebilir."
    )

    st.caption(
        f"Eski doküman: {comparison.old_document_title} · "
        f"Yeni doküman: {comparison.new_document_title}"
    )


    added_tab, removed_tab, changed_tab, detail_tab = st.tabs(
        [
            "➕ Eklenenler",
            "➖ Kaldırılanlar",
            "✏️ Değiştirilenler",
            "📋 Diğer değişiklikler",
        ]
    )

    with added_tab:
        added_search = st.text_input(
            "Bu bölümde ara",
            placeholder="Değişikliklerde ara...",
            key="comparison_search_added",
            label_visibility="collapsed",
        )
        filtered_added_items = filter_comparison_findings(
            comparison.added_items, added_search
        )
        if added_search.strip():
            st.caption(f"{len(filtered_added_items)} sonuç bulundu.")
        display_comparison_list(
            title="Eklenen hükümler",
            findings=filtered_added_items,
            empty_message="Yeni dokümana eklenen anlamlı bir madde bulunmadı.",
            change_label="Eklenen Hüküm",
        )

    with removed_tab:
        removed_search = st.text_input(
            "Bu bölümde ara",
            placeholder="Değişikliklerde ara...",
            key="comparison_search_removed",
            label_visibility="collapsed",
        )
        filtered_removed_items = filter_comparison_findings(
            comparison.removed_items, removed_search
        )
        if removed_search.strip():
            st.caption(f"{len(filtered_removed_items)} sonuç bulundu.")
        display_comparison_list(
            title="Kaldırılan hükümler",
            findings=filtered_removed_items,
            empty_message="Yeni dokümanda kaldırılan anlamlı bir madde bulunmadı.",
            change_label="Kaldırılan Hüküm",
        )

    with changed_tab:
        changed_search = st.text_input(
            "Bu bölümde ara",
            placeholder="Değişikliklerde ara...",
            key="comparison_search_changed",
            label_visibility="collapsed",
        )
        filtered_changed_items = filter_comparison_findings(
            comparison.changed_items, changed_search
        )
        if changed_search.strip():
            st.caption(f"{len(filtered_changed_items)} sonuç bulundu.")
        display_comparison_list(
            title="Değiştirilen hükümler",
            findings=filtered_changed_items,
            empty_message="İçeriği anlamlı biçimde değişen bir madde bulunmadı.",
            change_label="Değiştirilen Hüküm",
        )

    with detail_tab:
        detail_search = st.text_input(
            "Bu bölümde ara",
            placeholder="Değişikliklerde ara...",
            key="comparison_search_detail",
            label_visibility="collapsed",
        )
        filtered_date_changes = filter_comparison_findings(
            comparison.date_and_period_changes, detail_search
        )
        filtered_responsibility_changes = filter_comparison_findings(
            comparison.responsibility_changes, detail_search
        )
        filtered_standard_changes = filter_comparison_findings(
            comparison.standard_changes, detail_search
        )
        filtered_risk_changes = filter_risk_changes(
            comparison.risk_changes, detail_search
        )
        if detail_search.strip():
            detail_count = (
                len(filtered_date_changes)
                + len(filtered_responsibility_changes)
                + len(filtered_standard_changes)
                + len(filtered_risk_changes)
            )
            st.caption(f"{detail_count} sonuç bulundu.")
        display_comparison_list(
            title="Tarih ve süre değişiklikleri",
            findings=filtered_date_changes,
            empty_message="Tarih veya süre değişikliği tespit edilmedi.",
            change_label="Tarih ve süre değişikliği",
        )
        display_comparison_list(
            title="Sorumluluk değişiklikleri",
            findings=filtered_responsibility_changes,
            empty_message="Sorumluluk değişikliği tespit edilmedi.",
            change_label="Sorumluluk değişikliği",
        )
        display_comparison_list(
            title="Standart değişiklikleri",
            findings=filtered_standard_changes,
            empty_message="Standart referansı değişikliği tespit edilmedi.",
            change_label="Standart değişikliği",
        )
        display_risk_changes(filtered_risk_changes)


def display_document_comparison_section() -> None:
    """İki PDF yükleme ve kullanıcı odaklı karşılaştırma ekranını oluşturur."""

    st.subheader("Dokümanları karşılaştırın")
    st.write(
        "Eski ve yeni revizyonu yükleyin; eklenen, kaldırılan ve "
        "değiştirilen hükümleri inceleyin."
    )

    old_column, new_column = st.columns(2)

    with old_column:
        st.markdown("### Eski doküman")
        old_file = st.file_uploader(
            "Eski PDF dosyası",
            type=["pdf"],
            key="comparison_old_pdf",
        )

    with new_column:
        st.markdown("### Yeni doküman")
        new_file = st.file_uploader(
            "Yeni PDF dosyası",
            type=["pdf"],
            key="comparison_new_pdf",
        )

    if old_file is not None and new_file is not None:
        info_col1, info_col2 = st.columns(2)
        info_col1.success(f"Eski doküman seçildi: {old_file.name}")
        info_col2.success(f"Yeni doküman seçildi: {new_file.name}")

    compare_button = st.button(
        "📑 Dokümanları Karşılaştır",
        type="primary",
        use_container_width=True,
        disabled=(old_file is None or new_file is None),
    )

    if compare_button:
        try:
            old_file_bytes = old_file.getvalue()
            new_file_bytes = new_file.getvalue()
            st.session_state.comparison_old_file_bytes = old_file_bytes
            st.session_state.comparison_new_file_bytes = new_file_bytes
            cache_key = create_comparison_cache_key(
                old_file_bytes=old_file_bytes,
                new_file_bytes=new_file_bytes,
                model_name="local-v2",
            )

            cached_result = load_cached_comparison(cache_key)

            if cached_result is not None:
                st.session_state.comparison_result = cached_result
                st.session_state.comparison_old_file_name = old_file.name
                st.session_state.comparison_new_file_name = new_file.name
                st.session_state.comparison_notice = "Karşılaştırma sonucu hazır."
                st.rerun()

            with st.spinner("Dokümanlar karşılaştırılıyor..."):
                old_result = read_pdf(
                    file_bytes=old_file_bytes,
                    file_name=old_file.name,
                )
                new_result = read_pdf(
                    file_bytes=new_file_bytes,
                    file_name=new_file.name,
                )

                comparison_result = compare_documents_local(
                    old_pdf_result=old_result,
                    new_pdf_result=new_result,
                )

                save_cached_comparison(
                    cache_key=cache_key,
                    comparison=comparison_result,
                )

                st.session_state.comparison_result = comparison_result
                st.session_state.comparison_old_file_name = old_file.name
                st.session_state.comparison_new_file_name = new_file.name
                st.session_state.comparison_notice = "Karşılaştırma sonucu hazır."

            st.rerun()

        except (PDFReaderError, DocumentComparisonError) as exc:
            st.error(get_user_friendly_comparison_error(exc))
        except Exception as exc:
            st.error(get_user_friendly_comparison_error(exc))

    if st.session_state.comparison_result is not None:
        if st.session_state.comparison_notice:
            st.success(st.session_state.comparison_notice)

        clear_col1, clear_col2 = st.columns([4, 1])
        with clear_col2:
            clear_button = st.button(
                "Karşılaştırmayı Temizle",
                use_container_width=True,
            )
        if clear_button:
            clear_comparison_state()
            st.rerun()

        display_comparison_results(
            st.session_state.comparison_result
        )


def display_open_document() -> None:
    """
    Açık dokümanın ana ekranını gösterir.
    """

    result = st.session_state.pdf_result

    if result is None:
        return

    st.subheader("1. Açık doküman")

    st.success(
        f"Doküman açıldı: {result.file_name}"
    )

    selected_type = st.selectbox(
        "Doküman türü",
        options=DOCUMENT_TYPES,
        key="document_type_selector",
    )

    if selected_type != st.session_state.document_type:
        st.session_state.document_type = selected_type

        try:
            update_document_type(
                document_id=(
                    st.session_state.current_document_id
                ),
                document_type=selected_type,
            )

        except HistoryManagerError as exc:
            st.warning(str(exc))

    display_document_information()

    st.divider()
    st.subheader("2. Dokümanı analiz edin")

    if st.session_state.analysis_result is None:
        analyze_button = st.button(
            "🤖 Dokümanı Analiz Et",
            type="primary",
            use_container_width=True,
        )

        if analyze_button:
            analyze_current_document()

    else:
        col1, col2 = st.columns(
            [3, 1]
        )

        with col1:
            st.success(
                "Analiz sonuçları hazır."
            )

        with col2:
            reanalyze_button = st.button(
                "Analizi Yenile",
                use_container_width=True,
            )

        if reanalyze_button:
            st.session_state.analysis_result = None
            st.rerun()

        display_analysis_results(
            st.session_state.analysis_result
        )

        display_keyword_search_section()

        display_question_answer_section()


def display_upload_section() -> None:
    """
    Yeni PDF yükleme alanını gösterir.
    """

    st.subheader("1. Dokümanı yükleyin")

    left_column, right_column = st.columns(
        [2, 1]
    )

    with left_column:
        uploaded_file = st.file_uploader(
            "PDF dosyasını sürükleyip bırakın "
            "veya bilgisayarınızdan seçin.",
            type=["pdf"],
            accept_multiple_files=False,
        )

    with right_column:
        document_type = st.selectbox(
            "Doküman türü",
            options=DOCUMENT_TYPES,
            key="new_document_type",
        )

    if uploaded_file is None:
        st.warning(
            "Analize devam edebilmek için "
            "bir PDF dosyası yükleyin."
        )

        return

    st.success(
        f"Dosya seçildi: {uploaded_file.name}"
    )

    read_button = st.button(
        "PDF Dosyasını Aç",
        type="primary",
        use_container_width=True,
    )

    if not read_button:
        return

    try:
        with st.spinner(
            "PDF okunuyor ve geçmişe kaydediliyor..."
        ):
            file_bytes = uploaded_file.getvalue()

            pdf_result = read_pdf(
                file_bytes=file_bytes,
                file_name=uploaded_file.name,
            )

            if pdf_result.word_count == 0:
                raise PDFReaderError(
                    "PDF dosyasından okunabilir "
                    "metin çıkarılamadı."
                )

            document_chunks = split_pdf_pages(
                page_texts=pdf_result.page_texts,
                chunk_size=900,
                chunk_overlap=150,
            )

            document_id = save_document(
                file_bytes=file_bytes,
                file_name=uploaded_file.name,
                document_type=document_type,
                page_count=pdf_result.page_count,
            )

            reset_document_state()

            st.session_state.pdf_result = pdf_result
            st.session_state.uploaded_file_name = (
                uploaded_file.name
            )
            st.session_state.uploaded_file_bytes = (
                file_bytes
            )
            st.session_state.document_chunks = (
                document_chunks
            )
            st.session_state.document_type = (
                document_type
            )
            st.session_state.document_type_selector = (
                document_type
            )
            st.session_state.current_document_id = (
                document_id
            )

            load_saved_analysis_and_questions(
                document_id=document_id
            )

        st.rerun()

    except (
        PDFReaderError,
        TextSplitterError,
        HistoryManagerError,
    ) as exc:
        st.error(str(exc))


def main() -> None:
    """
    Ana uygulama fonksiyonudur.
    """

    initialize_database()
    initialize_session_state()

    sidebar()
    header()

    if st.session_state.app_mode == "comparison":
        display_document_comparison_section()
    elif st.session_state.pdf_result is None:
        display_upload_section()
    else:
        display_open_document()


if __name__ == "__main__":
    main()
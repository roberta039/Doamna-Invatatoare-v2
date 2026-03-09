import streamlit as st
from google import genai
from google.genai import types as genai_types
import edge_tts
import asyncio
from io import BytesIO
from supabase import create_client, Client
import uuid
import time
import tempfile
import ast
import re



# === APP INSTANCE ID ===
# Separă datele între instanțe diferite ale aceleiași aplicații (același Supabase, app-uri diferite)
# Setează APP_INSTANCE_ID în secrets.toml: APP_INSTANCE_ID = "profesor_v1"
def get_app_id() -> str:
    try:
        return str(st.secrets.get("APP_INSTANCE_ID", "default")).strip() or "default"
    except Exception:
        return "default"

# === CONSTANTE PENTRU LIMITE (FIX MEMORY LEAK) ===
MAX_MESSAGES_IN_MEMORY = 100
MAX_MESSAGES_TO_SEND_TO_AI = 20
MAX_MESSAGES_IN_DB_PER_SESSION = 500
CLEANUP_DAYS_OLD = 7

# === ISTORIC CONVERSAȚII ===
def get_session_list(limit: int = 20) -> list[dict]:
    """Returneaza lista sesiunilor — 2 query-uri totale in loc de N*2.

    FIX CACHE: Cache-ul de 30s e invalidat imediat dupa operatii care modifica sesiunile
    (mesaj nou, sesiune stearsa, sesiune noua). Astfel evitam date invechite fara
    sa interogam DB la fiecare rerun minor.
    """
    cache_ts  = st.session_state.get("_sess_list_ts", 0)
    cache_val = st.session_state.get("_sess_list_cache", None)
    force_refresh = st.session_state.pop("_sess_cache_dirty", False)  # FIX: flag de invalidare

    if not force_refresh and cache_val is not None and (time.time() - cache_ts) < 30:
        return cache_val

    try:
        supabase = get_supabase_client()

        # Query 1: sesiunile
        resp = (
            supabase.table("sessions")
            .select("session_id, last_active")
            .eq("app_id", get_app_id())
            .order("last_active", desc=True)
            .limit(limit)
            .execute()
        )
        sessions = resp.data or []
        if not sessions:
            return []

        session_ids = [s["session_id"] for s in sessions]

        # Query 2: primul mesaj user + count per sesiune (un singur query)
        hist_resp = (
            supabase.table("history")
            .select("session_id, role, content, timestamp")
            .in_("session_id", session_ids)
            .eq("role", "user")
            .order("timestamp", desc=False)
            .execute()
        )
        hist_rows = hist_resp.data or []

        # Agregare în Python — fără query suplimentare
        first_msg: dict[str, str] = {}
        msg_count: dict[str, int] = {}
        for row in hist_rows:
            sid = row["session_id"]
            msg_count[sid] = msg_count.get(sid, 0) + 1
            if sid not in first_msg:
                txt = row["content"][:60]
                first_msg[sid] = txt + ("..." if len(row["content"]) > 60 else "")

        result = []
        for s in sessions:
            sid = s["session_id"]
            cnt = msg_count.get(sid, 0)
            if cnt > 0:
                result.append({
                    "session_id": sid,
                    "last_active": s["last_active"],
                    "preview": first_msg.get(sid, "Conversație nouă"),
                    "msg_count": cnt,
                })

        st.session_state["_sess_list_cache"] = result
        st.session_state["_sess_list_ts"]    = time.time()
        return result

    except Exception as e:
        _log("Eroare la încărcarea sesiunilor", "silent", e)
        return cache_val or []


def switch_session(new_session_id: str):
    """Comută la o altă sesiune."""
    st.session_state.session_id = new_session_id
    st.session_state.messages = []
    st.query_params["sid"] = new_session_id
    invalidate_session_cache()  # FIX: forțează refresh la switch
    inject_session_js()


def invalidate_session_cache():
    """Marchează cache-ul sesiunilor ca expirat — apelat după orice modificare."""
    st.session_state["_sess_cache_dirty"] = True


def format_time_ago(timestamp) -> str:
    """Formatează timestamp ca timp relativ (ex: '2 ore în urmă'). Acceptă float sau ISO string."""
    # FIX: Supabase poate returna ISO string în loc de float
    if isinstance(timestamp, str):
        try:
            from datetime import datetime, timezone
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            timestamp = dt.timestamp()
        except Exception:
            return "necunoscut"
    try:
        diff = time.time() - float(timestamp)
    except (TypeError, ValueError):
        return "necunoscut"
    if diff < 60:
        return "acum"
    elif diff < 3600:
        mins = int(diff / 60)
        return f"{mins} min în urmă"
    elif diff < 86400:
        hours = int(diff / 3600)
        return f"{hours}h în urmă"
    else:
        days = int(diff / 86400)
        return f"{days} zile în urmă"




# === SUPABASE CLIENT + FALLBACK ===
@st.cache_resource
def get_supabase_client() -> Client | None:
    """Returnează clientul Supabase (conexiunea e lazy, fără query de test)."""
    try:
        url = st.secrets.get("SUPABASE_URL", "")
        key = st.secrets.get("SUPABASE_KEY", "")
        if not url or not key:
            return None
        return create_client(url, key)
    except Exception:
        return None


def is_supabase_available() -> bool:
    """Returnează statusul Supabase din cache — nu face request la fiecare apel.
    Statusul se actualizează doar când o operație reală eșuează sau reușește."""
    return st.session_state.get("_sb_online", True)


def _mark_supabase_offline():
    """Marchează Supabase ca offline și notifică utilizatorul."""
    was_online = st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = False
    if was_online:
        st.toast("⚠️ Baza de date offline — modul local activat.", icon="📴")


def _mark_supabase_online():
    """Marchează Supabase ca online și golește coada offline."""
    was_offline = not st.session_state.get("_sb_online", True)
    st.session_state["_sb_online"] = True
    if was_offline:
        st.toast("✅ Conexiunea restabilită!", icon="🟢")
        _flush_offline_queue()


# --- Coadă offline: mesaje salvate local când Supabase e down ---
def _get_offline_queue() -> list:
    return st.session_state.setdefault("_offline_queue", [])


def _flush_offline_queue():
    """Trimite mesajele din coada offline la Supabase când revine online."""
    queue = _get_offline_queue()
    if not queue:
        return
    client = get_supabase_client()
    if not client:
        return
    failed = []
    for item in queue:
        try:
            client.table("history").insert(item).execute()
        except Exception:
            failed.append(item)
    st.session_state["_offline_queue"] = failed
    if not failed:
        st.toast(f"✅ {len(queue)} mesaje sincronizate cu baza de date.", icon="☁️")

# === VOCI EDGE TTS (VOCE BĂRBAT) ===
VOICE_MALE_RO = "ro-RO-EmilNeural"
VOICE_FEMALE_RO = "ro-RO-AlinaNeural"


st.set_page_config(page_title="Profesor Liceu", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# Aplică tema dark/light imediat la fiecare rerun
if st.session_state.get("dark_mode", False):
    st.markdown("""
    <script>
    (function() {
        function applyDark() {
            const root = window.parent.document.documentElement;
            root.setAttribute('data-theme', 'dark');
            // Streamlit's internal theme toggle
            const btn = window.parent.document.querySelector('[data-testid="baseButton-headerNoPadding"]');
        }
        applyDark();
        // Re-apply after Streamlit re-renders
        setTimeout(applyDark, 100);
        setTimeout(applyDark, 500);
    })();
    </script>
    <style>
        /* Manual dark mode overrides pentru elementele principale */
        :root { color-scheme: dark; }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0e1117 !important;
            color: #fafafa !important;
        }
        [data-testid="stSidebar"] {
            background-color: #161b22 !important;
        }
        .stChatMessage {
            background-color: #1a1f2e !important;
        }
        .stTextArea textarea, .stTextInput input {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
            border-color: #444 !important;
        }
        .stSelectbox > div, .stRadio > div {
            background-color: #1a1f2e !important;
            color: #fafafa !important;
        }
        p, h1, h2, h3, h4, h5, h6, li, label, span {
            color: #fafafa !important;
        }
        .stButton > button {
            border-color: #555 !important;
        }
        hr { border-color: #333 !important; }
        .stExpander { border-color: #333 !important; }
        [data-testid="stChatInput"] {
            background-color: #1a1f2e !important;
        }
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
    .stChatMessage { font-size: 16px; }
    footer { visibility: hidden; }

    /* SVG container - light mode */
    .svg-container {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #ddd;
        text-align: center;
        margin: 15px 0;
        overflow: auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        max-width: 100%;
    }
    .svg-container svg { max-width: 100%; height: auto; }

    /* Dark mode */
    [data-theme="dark"] .svg-container {
        background-color: #1e1e2e;
        border-color: #444;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }



    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 10px 4px;
        font-size: 14px;
        color: #888;
    }
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    .typing-dots span {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #888;
        animation: typing-bounce 1.2s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0.7); opacity: 0.4; }
        40%            { transform: scale(1.0); opacity: 1.0; }
    }
</style>
""", unsafe_allow_html=True)


# === DATABASE FUNCTIONS (SUPABASE) ===

# ÎMBUNĂTĂȚIRE 3: Logger centralizat — afișează toast utilizatorului ȘI loghează în consolă.
# Niveluri: "info" (toast albastru), "warning" (toast portocaliu), "error" (toast roșu).
# Erorile silențioase de fundal (cleanup, trim) folosesc doar consola.
def _log(msg: str, level: str = "silent", exc: Exception = None):
    """Loghează un mesaj și opțional afișează un toast în interfață.
    
    level:
        "silent"  — doar print în consolă (erori de fundal, nu deranjează utilizatorul)
        "info"    — toast verde, pentru operații reușite/informative
        "warning" — toast portocaliu, pentru degradări non-critice
        "error"   — toast roșu, pentru erori vizibile utilizatorului
    """
    full_msg = f"{msg}: {exc}" if exc else msg
    print(full_msg)
    icon_map = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}
    if level in icon_map:
        try:
            st.toast(msg, icon=icon_map[level])
        except Exception:
            pass  # st.toast poate eșua în contexte fără sesiune activă


def init_db():
    """Verifică conexiunea la Supabase. Dacă e offline, activează modul local."""
    online = is_supabase_available()
    if not online:
        st.warning("📴 **Modul offline activ** — conversația se păstrează în memorie. "
                   "Istoricul va fi sincronizat automat când conexiunea revine.", icon="⚠️")


def cleanup_old_sessions(days_old: int = CLEANUP_DAYS_OLD):
    """Șterge sesiunile vechi — rulează cel mult o dată pe zi."""
    if time.time() - st.session_state.get("_last_cleanup", 0) < 86400:
        return
    st.session_state["_last_cleanup"] = time.time()
    try:
        supabase = get_supabase_client()
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        supabase.table("history").delete().lt("timestamp", cutoff_time).eq("app_id", get_app_id()).execute()
        supabase.table("sessions").delete().lt("last_active", cutoff_time).eq("app_id", get_app_id()).execute()
    except Exception as e:
        _log("Eroare la curățarea sesiunilor vechi", "silent", e)


def save_message_to_db(session_id, role, content):
    """Salvează un mesaj în Supabase. Dacă e offline, pune în coada locală."""
    record = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "app_id": get_app_id()
    }
    if not is_supabase_available():
        _get_offline_queue().append(record)
        return
    try:
        client = get_supabase_client()
        client.table("history").insert(record).execute()
        _mark_supabase_online()
    except Exception as e:
        _log("Mesajul nu a putut fi salvat", "warning", e)
        _mark_supabase_offline()
        _get_offline_queue().append(record)


def load_history_from_db(session_id, limit: int = MAX_MESSAGES_IN_MEMORY):
    """Încarcă istoricul din Supabase. Fallback: returnează ce e deja în session_state."""
    if not is_supabase_available():
        # Offline: întoarce mesajele deja în memorie (dacă există)
        return st.session_state.get("messages", [])[-limit:]
    try:
        client = get_supabase_client()
        response = (
            client.table("history")
            .select("role, content, timestamp")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .order("timestamp", desc=False)
            .limit(limit)
            .execute()
        )
        return [{"role": row["role"], "content": row["content"]} for row in response.data]
    except Exception as e:
        _log("Eroare la încărcarea istoricului", "silent", e)
        return st.session_state.get("messages", [])[-limit:]


def clear_history_db(session_id):
    """Șterge istoricul pentru o sesiune din Supabase."""
    try:
        supabase = get_supabase_client()
        supabase.table("history").delete().eq("session_id", session_id).eq("app_id", get_app_id()).execute()
        invalidate_session_cache()  # FIX: sesiune ștearsă = cache invalid
    except Exception as e:
        _log("Istoricul nu a putut fi șters", "warning", e)


def trim_db_messages(session_id: str):
    """Limitează mesajele din DB pentru o sesiune (FIX MEMORY LEAK)."""
    try:
        supabase = get_supabase_client()

        # Numără mesajele sesiunii
        count_resp = (
            supabase.table("history")
            .select("id", count="exact")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .execute()
        )
        count = count_resp.count or 0

        if count > MAX_MESSAGES_IN_DB_PER_SESSION:
            to_delete = count - MAX_MESSAGES_IN_DB_PER_SESSION
            # Obține ID-urile celor mai vechi mesaje
            old_resp = (
                supabase.table("history")
                .select("id")
                .eq("session_id", session_id)
                .eq("app_id", get_app_id())
                .order("timestamp", desc=False)
                .limit(to_delete)
                .execute()
            )
            ids_to_delete = [row["id"] for row in old_resp.data]
            if ids_to_delete:
                supabase.table("history").delete().in_("id", ids_to_delete).execute()
    except Exception as e:
        _log("Eroare la curățarea DB", "silent", e)


# === SESSION MANAGEMENT (SUPABASE) ===
def generate_unique_session_id() -> str:
    """Generează un session ID garantat unic."""
    uuid_part = uuid.uuid4().hex[:16]
    time_part = hex(int(time.time() * 1000000))[2:][-8:]
    random_part = uuid.uuid4().hex[:8]
    return f"{uuid_part}{time_part}{random_part}"


# Regex precompilat pentru validarea session_id — doar hex lowercase, 16-64 caractere
_SESSION_ID_RE = re.compile(r'^[a-f0-9]{16,64}$')

def is_valid_session_id(sid: str) -> bool:
    """Validează session_id: doar hex lowercase, lungime 16-64 caractere.
    
    FIX: Fără validare, un sid malițios din URL (?sid=../../../etc) putea
    ajunge direct în query-urile Supabase ca parametru nevalidat.
    """
    if not sid or not isinstance(sid, str):
        return False
    return bool(_SESSION_ID_RE.match(sid))


def session_exists_in_db(session_id: str) -> bool:
    """Verifică dacă un session_id există deja în Supabase."""
    try:
        supabase = get_supabase_client()
        response = (
            supabase.table("sessions")
            .select("session_id")
            .eq("session_id", session_id)
            .eq("app_id", get_app_id())
            .limit(1)
            .execute()
        )
        return len(response.data) > 0
    except Exception:
        return False


def register_session(session_id: str):
    """Înregistrează o sesiune nouă în Supabase. Silent dacă offline."""
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        now = time.time()
        client.table("sessions").upsert({
            "session_id": session_id,
            "created_at": now,
            "last_active": now,
            "app_id": get_app_id()
        }).execute()
    except Exception as e:
        _log("Eroare la înregistrarea sesiunii", "silent", e)


def update_session_activity(session_id: str):
    """Actualizează timestamp-ul activității — cel mult o dată la 5 minute."""
    last = st.session_state.get("_last_activity_update", 0)
    if time.time() - last < 300:
        return
    st.session_state["_last_activity_update"] = time.time()
    if not is_supabase_available():
        return
    try:
        client = get_supabase_client()
        client.table("sessions").update({
            "last_active": time.time()
        }).eq("session_id", session_id).execute()
    except Exception as e:
        _log("Eroare la actualizarea sesiunii", "silent", e)


def inject_session_js():
    """
    Injectează JS care sincronizează session_id și API key cu localStorage.
    - session_id: persistă între sesiuni pe același browser
    - API key: persistă între refresh-uri pe același browser (localStorage)
    """
    import streamlit.components.v1 as components
    components.html("""
    <script>
    (function() {
        const SID_KEY    = 'profesor_session_id';
        const APIKEY_KEY = 'profesor_api_key';
        const params     = new URLSearchParams(window.parent.location.search);

        // ── SESSION ID ──
        // Logică: fiecare browser are propriul session_id în localStorage
        // NU expunem session_id în URL (ar permite partajarea istoricului prin link)
        const sidFromUrl = params.get('sid');
        const storedSid  = localStorage.getItem(SID_KEY);

        if (sidFromUrl && sidFromUrl.length >= 16) {
            // Sid vine din URL — salvăm în localStorage și SCOATEM din URL
            // (poate fi propriul sid restaurat de Streamlit la switch_session)
            localStorage.setItem(SID_KEY, sidFromUrl);
            params.delete('sid');
        } else if (!storedSid) {
            // Prima vizită pe acest browser — Streamlit va genera un sid nou
            // Nu facem nimic, lăsăm Streamlit să creeze și să trimită sid via rerun
        }
        // Nu punem niciodată sid în URL de la noi — previne partajarea istoricului

        // ── API KEY ──
        const keyFromUrl = params.get('apikey');
        if (keyFromUrl && keyFromUrl.startsWith('AIza')) {
            localStorage.setItem(APIKEY_KEY, keyFromUrl);
            params.delete('apikey');
        } else {
            const storedKey = localStorage.getItem(APIKEY_KEY);
            if (storedKey && storedKey.startsWith('AIza') && !params.get('apikey')) {
                params.set('apikey', storedKey);
            }
        }

        // Actualizează URL-ul fără să reîncărce pagina
        const newSearch = params.toString();
        const newUrl = window.parent.location.pathname +
            (newSearch ? '?' + newSearch : '');
        if (window.parent.location.href !== window.parent.location.origin + newUrl) {
            window.parent.history.replaceState(null, '', newUrl);
        }
    })();
    </script>

    <script>
    window._clearStoredApiKey = function() {
        localStorage.removeItem('profesor_api_key');
    };
    </script>
    """, height=0)


def get_or_create_session_id() -> str:
    """
    Obține session ID din: session_state → ?sid= (restaurat din localStorage de JS) → sesiune nouă.
    
    IZOLARE: Fiecare browser are propriul session_id stocat în localStorage.
    session_id nu apare niciodată în URL-ul vizibil (previne partajarea istoricului prin link).
    """
    # 1. Deja în sesiunea curentă Streamlit (refresh normal)
    if "session_id" in st.session_state:
        existing_id = st.session_state.session_id
        if is_valid_session_id(existing_id):
            return existing_id

    # 2. Restaurat din localStorage via ?sid= în URL
    # JS-ul pune ?sid= în URL DOAR când face switch_session sau la primul load cu sid existent
    if "sid" in st.query_params:
        sid_from_storage = st.query_params["sid"]
        if is_valid_session_id(sid_from_storage):
            if session_exists_in_db(sid_from_storage):
                # Scoate sid din URL după ce l-am citit (nu rămâne vizibil)
                try:
                    st.query_params.pop("sid", None)
                except Exception:
                    pass
                return sid_from_storage
            # Sid invalid/expirat — ignorăm și creăm sesiune nouă

    # 3. Creează sesiune nouă (primul load pe browser nou)
    for _ in range(10):
        new_id = generate_unique_session_id()
        if not session_exists_in_db(new_id):
            register_session(new_id)
            # Trimite sid la JS via URL ca să-l salveze în localStorage
            # JS îl scoate din URL imediat după ce îl salvează
            try:
                st.query_params["sid"] = new_id
            except Exception:
                pass
            return new_id

    fallback_id = uuid.uuid4().hex + uuid.uuid4().hex[:8]
    register_session(fallback_id)
    return fallback_id


# === MEMORY MANAGEMENT (FIX MEMORY LEAK) ===
def trim_session_messages():
    """Limitează mesajele din session_state pentru a preveni memory leak."""
    if "messages" in st.session_state:
        current_count = len(st.session_state.messages)
        
        if current_count > MAX_MESSAGES_IN_MEMORY:
            excess = current_count - MAX_MESSAGES_IN_MEMORY
            st.session_state.messages = st.session_state.messages[excess:]
            st.toast(f"📝 Am arhivat {excess} mesaje vechi pentru performanță.", icon="📦")


def get_context_for_ai(messages: list) -> list:
    """Pregătește contextul pentru AI cu limită de mesaje."""
    if len(messages) <= MAX_MESSAGES_TO_SEND_TO_AI:
        return messages  # FIX: eliminat [:-1] care tăia ultimul mesaj din context
    
    first_message = messages[0] if messages else None
    recent_messages = messages[-MAX_MESSAGES_TO_SEND_TO_AI:]
    
    if first_message and first_message not in recent_messages:
        return [first_message] + recent_messages[1:]
    return recent_messages


def save_message_with_limits(session_id: str, role: str, content: str):
    """Salvează mesaj și verifică limitele."""
    save_message_to_db(session_id, role, content)
    invalidate_session_cache()  # FIX: un mesaj nou înseamnă date noi în sidebar
    
    if len(st.session_state.get("messages", [])) % 10 == 0:
        trim_db_messages(session_id)
    
    trim_session_messages()




# === AUDIO / TTS FUNCTIONS ===

# --- Tabele de date pentru clean_text_for_audio ---

# Unități: (sufix, pronunție) — ordonate de la lung la scurt pentru a evita match greșit
_UNITS: list[tuple[str, str]] = [
    # Rezistență
    ("GΩ", "gigaohmi"), ("MΩ", "megaohmi"), ("kΩ", "kiloohmi"),
    ("mΩ", "miliohmi"), ("μΩ", "microohmi"), ("nΩ", "nanoohmi"), ("Ω", "ohmi"),
    # Temperatură
    ("°C", "grade Celsius"), ("°F", "grade Fahrenheit"), ("°K", "Kelvin"), ("K", "Kelvin"), ("°", "grade"),
    # Tensiune
    ("MV", "megavolți"), ("kV", "kilovolți"), ("mV", "milivolți"), ("μV", "microvolți"), ("V", "volți"),
    # Curent
    ("kA", "kiloamperi"), ("mA", "miliamperi"), ("μA", "microamperi"), ("nA", "nanoamperi"), ("A", "amperi"),
    # Putere
    ("GW", "gigawați"), ("MW", "megawați"), ("kW", "kilowați"), ("mW", "miliwați"), ("μW", "microwați"), ("W", "wați"),
    # Frecvență
    ("THz", "terahertzi"), ("GHz", "gigahertzi"), ("MHz", "megahertzi"), ("kHz", "kilohertzi"), ("mHz", "milihertzi"), ("Hz", "hertzi"),
    # Capacitate
    ("mF", "milifarazi"), ("μF", "microfarazi"), ("nF", "nanofarazi"), ("pF", "picofarazi"), ("F", "farazi"),
    # Inductanță
    ("mH", "milihenry"), ("μH", "microhenry"), ("nH", "nanohenry"), ("H", "henry"),
    # Sarcină electrică
    ("mC", "milicoulombi"), ("μC", "microcoulombi"), ("nC", "nanocoulombi"), ("C", "coulombi"),
    # Câmp magnetic
    ("Wb", "weberi"), ("mT", "militesla"), ("μT", "microtesla"), ("T", "tesla"),
    # Forță
    ("MN", "meganewtoni"), ("kN", "kilonewtoni"), ("mN", "milinewtoni"), ("N", "newtoni"),
    # Energie
    ("kWh", "kilowatt oră"), ("Wh", "watt oră"),
    ("GeV", "gigaelectronvolți"), ("MeV", "megaelectronvolți"), ("keV", "kiloelectronvolți"), ("eV", "electronvolți"),
    ("kcal", "kilocalorii"), ("cal", "calorii"),
    ("GJ", "gigajouli"), ("MJ", "megajouli"), ("kJ", "kilojouli"), ("mJ", "milijouli"), ("J", "jouli"),
    # Presiune
    ("GPa", "gigapascali"), ("MPa", "megapascali"), ("kPa", "kilopascali"), ("hPa", "hectopascali"), ("Pa", "pascali"),
    ("mmHg", "milimetri coloană de mercur"), ("atm", "atmosfere"), ("bar", "bari"),
    # Lungime
    ("km", "kilometri"), ("dm", "decimetri"), ("cm", "centimetri"), ("mm", "milimetri"),
    ("μm", "micrometri"), ("nm", "nanometri"), ("pm", "picometri"), ("Å", "angstromi"), ("m", "metri"),
    # Masă
    ("kg", "kilograme"), ("mg", "miligrame"), ("μg", "micrograme"), ("ng", "nanograme"), ("g", "grame"), ("t", "tone"),
    # Volum
    ("mL", "mililitri"), ("ml", "mililitri"), ("μL", "microlitri"), ("L", "litri"), ("l", "litri"),
    ("dm³", "decimetri cubi"), ("cm³", "centimetri cubi"), ("mm³", "milimetri cubi"), ("m³", "metri cubi"),
    # Timp
    ("ms", "milisecunde"), ("μs", "microsecunde"), ("ns", "nanosecunde"), ("ps", "picosecunde"),
    ("min", "minute"), ("s", "secunde"), ("h", "ore"),
    # Suprafață
    ("km²", "kilometri pătrați"), ("m²", "metri pătrați"), ("dm²", "decimetri pătrați"),
    ("cm²", "centimetri pătrați"), ("mm²", "milimetri pătrați"), ("ha", "hectare"),
    # Viteză & derivate
    ("m/s²", "metri pe secundă la pătrat"), ("m/s", "metri pe secundă"), ("km/h", "kilometri pe oră"),
    ("km/s", "kilometri pe secundă"), ("cm/s", "centimetri pe secundă"),
    ("rad/s", "radiani pe secundă"), ("rpm", "rotații pe minut"),
    # Densitate, presiune compusă
    ("kg/m³", "kilograme pe metru cub"), ("g/cm³", "grame pe centimetru cub"), ("g/mL", "grame pe mililitru"),
    ("N/m²", "newtoni pe metru pătrat"), ("N/m", "newtoni pe metru"),
    ("J/kg", "jouli pe kilogram"), ("J/mol", "jouli pe mol"),
    ("W/m²", "wați pe metru pătrat"), ("V/m", "volți pe metru"), ("A/m", "amperi pe metru"),
    # Chimie
    ("mol/L", "moli pe litru"), ("mol/l", "moli pe litru"),
    ("g/mol", "grame pe mol"), ("kg/mol", "kilograme pe mol"),
    ("mol", "moli"), ("M", "molar"),
    # Radiație & optică
    ("Bq", "becquereli"), ("Gy", "gray"), ("Sv", "sievert"),
    ("cd", "candele"), ("lm", "lumeni"), ("lx", "lucși"),
    # Unghiuri
    ("rad", "radiani"), ("sr", "steradiani"),
]

# Simboluri și combinații speciale: (literal, înlocuitor)
_SYMBOLS: dict[str, str] = {
    ">=": " mai mare sau egal cu ", "<=": " mai mic sau egal cu ",
    "!=": " diferit de ", "==": " egal cu ", "<>": " diferit de ",
    ">>": " mult mai mare decât ", "<<": " mult mai mic decât ",
    "->": " implică ", "<-": " provine din ", "<->": " echivalent cu ", "=>": " rezultă că ",
    "...": " ", "…": " ", "N·m": " newton metri ", "N*m": " newton metri ", "kW·h": " kilowatt oră ",
    "α": " alfa ", "β": " beta ", "γ": " gama ", "δ": " delta ", "ε": " epsilon ",
    "ζ": " zeta ", "η": " eta ", "θ": " teta ", "ι": " iota ", "κ": " kapa ",
    "λ": " lambda ", "μ": " miu ", "ν": " niu ", "ξ": " csi ", "ο": " omicron ",
    "π": " pi ", "ρ": " ro ", "σ": " sigma ", "ς": " sigma ", "τ": " tau ",
    "υ": " ipsilon ", "φ": " fi ", "χ": " hi ", "ψ": " psi ", "ω": " omega ",
    "Α": " alfa ", "Β": " beta ", "Γ": " gama ", "Δ": " delta ", "Ε": " epsilon ",
    "Ζ": " zeta ", "Η": " eta ", "Θ": " teta ", "Ι": " iota ", "Κ": " kapa ",
    "Λ": " lambda ", "Μ": " miu ", "Ν": " niu ", "Ξ": " csi ", "Ο": " omicron ",
    "Π": " pi ", "Ρ": " ro ", "Σ": " sigma ", "Τ": " tau ", "Υ": " ipsilon ",
    "Φ": " fi ", "Χ": " hi ", "Ψ": " psi ", "Ω": " omega ",
    "∞": " infinit ", "∑": " suma ", "∏": " produsul ", "∫": " integrala ",
    "∂": " derivata parțială ", "√": " radical din ", "∛": " radical de ordin 3 din ",
    "∜": " radical de ordin 4 din ", "±": " plus minus ", "∓": " minus plus ",
    "×": " ori ", "÷": " împărțit la ", "≠": " diferit de ", "≈": " aproximativ egal cu ",
    "≡": " identic cu ", "≤": " mai mic sau egal cu ", "≥": " mai mare sau egal cu ",
    "≪": " mult mai mic decât ", "≫": " mult mai mare decât ", "∝": " proporțional cu ",
    "∈": " aparține lui ", "∉": " nu aparține lui ", "⊂": " inclus în ", "⊃": " include ",
    "⊆": " inclus sau egal cu ", "⊇": " include sau egal cu ",
    "∪": " reunit cu ", "∩": " intersectat cu ", "∅": " mulțimea vidă ",
    "∀": " pentru orice ", "∃": " există ", "∄": " nu există ",
    "∴": " deci ", "∵": " deoarece ",
    "→": " implică ", "←": " rezultă din ", "↔": " echivalent cu ",
    "⇒": " rezultă că ", "⇐": " provine din ", "⇔": " dacă și numai dacă ",
    "↑": " crește ", "↓": " scade ", "°": " grade ", "′": " ", "″": " ",
    "‰": " la mie ", "∠": " unghiul ", "⊥": " perpendicular pe ", "∥": " paralel cu ",
    "△": " triunghiul ", "□": " ", "○": " ", "★": " ", "☆": " ",
    "✓": " corect ", "✗": " greșit ", "✘": " greșit ",
    ">": " mai mare decât ", "<": " mai mic decât ", "=": " egal ",
    "+": " plus ", "−": " minus ", "—": " ", "–": " ",
    "·": " ori ", "•": " ", "∙": " ori ", "⋅": " ori ",
    "⁰": " la puterea 0 ", "¹": " la puterea 1 ", "²": " la pătrat ", "³": " la cub ",
    "⁴": " la puterea 4 ", "⁵": " la puterea 5 ", "⁶": " la puterea 6 ",
    "⁷": " la puterea 7 ", "⁸": " la puterea 8 ", "⁹": " la puterea 9 ",
    "⁺": " plus ", "⁻": " minus ", "⁼": " egal ",
    "₀": " indice 0 ", "₁": " indice 1 ", "₂": " indice 2 ", "₃": " indice 3 ",
    "₄": " indice 4 ", "₅": " indice 5 ", "₆": " indice 6 ", "₇": " indice 7 ",
    "₈": " indice 8 ", "₉": " indice 9 ", "₊": " plus ", "₋": " minus ", "₌": " egal ",
    "ₐ": " indice a ", "ₑ": " indice e ", "ₕ": " indice h ", "ᵢ": " indice i ",
    "ⱼ": " indice j ", "ₖ": " indice k ", "ₗ": " indice l ", "ₘ": " indice m ",
    "ₙ": " indice n ", "ₒ": " indice o ", "ₚ": " indice p ", "ᵣ": " indice r ",
    "ₛ": " indice s ", "ₜ": " indice t ", "ᵤ": " indice u ", "ᵥ": " indice v ", "ₓ": " indice x ",
    "ᵦ": " indice beta ", "ᵧ": " indice gama ", "ᵨ": " indice ro ", "ᵩ": " indice fi ", "ᵪ": " indice hi ",
    "ᵃ": " la puterea a ", "ᵇ": " la puterea b ", "ᶜ": " la puterea c ", "ᵈ": " la puterea d ",
    "ᵉ": " la puterea e ", "ᶠ": " la puterea f ", "ᵍ": " la puterea g ", "ʰ": " la puterea h ",
    "ⁱ": " la puterea i ", "ʲ": " la puterea j ", "ᵏ": " la puterea k ", "ˡ": " la puterea l ",
    "ᵐ": " la puterea m ", "ⁿ": " la puterea n ", "ᵒ": " la puterea o ", "ᵖ": " la puterea p ",
    "ʳ": " la puterea r ", "ˢ": " la puterea s ", "ᵗ": " la puterea t ", "ᵘ": " la puterea u ",
    "ᵛ": " la puterea v ", "ʷ": " la puterea w ", "ˣ": " la puterea x ", "ʸ": " la puterea y ", "ᶻ": " la puterea z ",
    "½": " o doime ", "⅓": " o treime ", "⅔": " două treimi ", "¼": " un sfert ", "¾": " trei sferturi ",
    "⅕": " o cincime ", "⅖": " două cincimi ", "⅗": " trei cincimi ", "⅘": " patru cincimi ",
    "⅙": " o șesime ", "⅚": " cinci șesimi ", "⅛": " o optime ", "⅜": " trei optimi ",
    "⅝": " cinci optimi ", "⅞": " șapte optimi ",
    "%": " procent ", "&": " și ", "#": " numărul ", "~": " aproximativ ",
    "≅": " congruent cu ", "≃": " aproximativ egal cu ", "|": " ", "‖": " ", "⋯": " ",
    "∧": " și ", "∨": " sau ", "¬": " negația lui ", "∎": " ",
    "ℕ": " mulțimea numerelor naturale ", "ℤ": " mulțimea numerelor întregi ",
    "ℚ": " mulțimea numerelor raționale ", "ℝ": " mulțimea numerelor reale ",
    "ℂ": " mulțimea numerelor complexe ", "℃": " grade Celsius ", "℉": " grade Fahrenheit ",
    "Å": " angstrom ", "№": " numărul ",
}

# Comenzi LaTeX: (pattern, replacement)
_LATEX_PATTERNS: list[tuple[str, str]] = [
    (r'\\sqrt\[(\d+)\]\{([^}]+)\}', r' radical de ordin \1 din \2 '),
    (r'\\sqrt\{([^}]+)\}', r' radical din \1 '),
    (r'\\d?frac\{([^}]+)\}\{([^}]+)\}', r' \1 supra \2 '),
    (r'\^\{([^}]+)\}', r' la puterea \1 '), (r'\^(\d+)', r' la puterea \1 '),
    (r'_\{([^}]+)\}', r' indice \1 '),     (r'_(\d+)', r' indice \1 '),
    (r'\\alpha', ' alfa '), (r'\\beta', ' beta '), (r'\\gamma', ' gama '),
    (r'\\delta', ' delta '), (r'\\(?:var)?epsilon', ' epsilon '),
    (r'\\zeta', ' zeta '), (r'\\eta', ' eta '), (r'\\(?:var)?theta', ' teta '),
    (r'\\iota', ' iota '), (r'\\kappa', ' kapa '), (r'\\lambda', ' lambda '),
    (r'\\mu', ' miu '), (r'\\nu', ' niu '), (r'\\xi', ' csi '),
    (r'\\(?:var)?pi', ' pi '), (r'\\(?:var)?rho', ' ro '),
    (r'\\(?:var)?sigma', ' sigma '), (r'\\tau', ' tau '), (r'\\upsilon', ' ipsilon '),
    (r'\\(?:var)?phi', ' fi '), (r'\\chi', ' hi '), (r'\\psi', ' psi '),
    (r'\\(?:var)?omega', ' omega '),
    (r'\\Gamma', ' gama '), (r'\\Delta', ' delta '), (r'\\Theta', ' teta '),
    (r'\\Lambda', ' lambda '), (r'\\Xi', ' csi '), (r'\\Pi', ' pi '),
    (r'\\Sigma', ' sigma '), (r'\\Upsilon', ' ipsilon '), (r'\\Phi', ' fi '),
    (r'\\Psi', ' psi '), (r'\\Omega', ' omega '),
    (r'\\times', ' ori '), (r'\\cdot', ' ori '), (r'\\div', ' împărțit la '),
    (r'\\pm', ' plus minus '), (r'\\mp', ' minus plus '),
    (r'\\(?:leq?)', ' mai mic sau egal cu '), (r'\\(?:geq?)', ' mai mare sau egal cu '),
    (r'\\(?:neq?)', ' diferit de '), (r'\\approx', ' aproximativ egal cu '),
    (r'\\equiv', ' echivalent cu '), (r'\\sim', ' similar cu '),
    (r'\\propto', ' proporțional cu '), (r'\\infty', ' infinit '),
    (r'\\sum', ' suma '), (r'\\prod', ' produsul '),
    (r'\\iiint', ' integrala triplă '), (r'\\iint', ' integrala dublă '),
    (r'\\oint', ' integrala pe contur '), (r'\\int', ' integrala '),
    (r'\\lim', ' limita '), (r'\\log', ' logaritm de '), (r'\\ln', ' logaritm natural de '),
    (r'\\lg', ' logaritm zecimal de '), (r'\\exp', ' exponențiala de '),
    (r'\\sin', ' sinus de '), (r'\\cos', ' cosinus de '),
    (r'\\(?:tg|tan)', ' tangentă de '), (r'\\(?:ctg|cot)', ' cotangentă de '),
    (r'\\sec', ' secantă de '), (r'\\csc', ' cosecantă de '),
    (r'\\arcsin', ' arc sinus de '), (r'\\arccos', ' arc cosinus de '),
    (r'\\(?:arctg|arctan)', ' arc tangentă de '),
    (r'\\sinh', ' sinus hiperbolic de '), (r'\\cosh', ' cosinus hiperbolic de '),
    (r'\\tanh', ' tangentă hiperbolică de '),
    (r'\\(?:right|left)?arrow', ' implică '), (r'\\to\b', ' tinde la '),
    (r'\\Rightarrow', ' rezultă că '), (r'\\Leftarrow', ' este implicat de '),
    (r'\\[Ll]eftrightarrow', ' echivalent cu '), (r'\\Leftrightarrow', ' dacă și numai dacă '),
    (r'\\forall', ' pentru orice '), (r'\\exists', ' există '), (r'\\nexists', ' nu există '),
    (r'\\in\b', ' aparține lui '), (r'\\notin', ' nu aparține lui '),
    (r'\\subseteq', ' inclus sau egal cu '), (r'\\supseteq', ' include sau egal cu '),
    (r'\\subset', ' inclus în '), (r'\\supset', ' include '),
    (r'\\cup', ' reunit cu '), (r'\\cap', ' intersectat cu '),
    (r'\\(?:empty[Ss]et|varnothing)', ' mulțimea vidă '),
    (r'\\mathbb\{R\}', ' mulțimea numerelor reale '),
    (r'\\mathbb\{N\}', ' mulțimea numerelor naturale '),
    (r'\\mathbb\{Z\}', ' mulțimea numerelor întregi '),
    (r'\\mathbb\{Q\}', ' mulțimea numerelor raționale '),
    (r'\\mathbb\{C\}', ' mulțimea numerelor complexe '),
    (r'\\partial', ' derivata parțială '), (r'\\nabla', ' nabla '),
    (r'\\(?:degree|circ)\b', ' grad '), (r'\\(?:angle|measuredangle)', ' unghiul '),
    (r'\\perp', ' perpendicular pe '), (r'\\parallel', ' paralel cu '),
    (r'\\triangle', ' triunghiul '), (r'\\square', ' pătratul '),
    (r'\\therefore', ' deci '), (r'\\because', ' deoarece '),
    (r'\\lt\b', ' mai mic decât '), (r'\\gt\b', ' mai mare decât '),
]

# Regex precompilat pentru unități (număr + unitate)
# FIX: adăugat negative lookbehind (?<![A-Za-z]) pentru a evita match-ul
# în interiorul cuvintelor (ex: "kWh" să nu fie prins de "h" = ore separat,
# "Viteză" să nu fie prins de "V" = volți).
# Ordinea în _UNITS (lung → scurt) garantează că "kWh" e prins înaintea "W" sau "h".
_NUM = r'(\d+[.,]?\d*)'
_UNIT_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(
            r'(?<![A-Za-z])' +          # nu precedat de literă (evită match în cuvinte)
            _NUM +
            r'\s*' + re.escape(unit) +
            r'(?![A-Za-z/²³])'          # nu urmat de literă, slash sau exponenți (evită "kg/m³" prins de "kg")
        ),
        r'\1 ' + pron
    )
    for unit, pron in _UNITS
]


def clean_text_for_audio(text: str) -> str:
    """Curăță textul de LaTeX, SVG, Markdown, emoji-uri pentru TTS."""
    if not text:
        return ""

    # 0. Elimină emoji-uri și simboluri speciale Unicode
    # Range-uri principale de emoji-uri și simboluri grafice
    text = re.sub(
        r'[\U0001F300-\U0001F9FF'   # emoji-uri generale (😀🎨🔢 etc.)
        r'\U00002600-\U000027BF'    # simboluri diverse (☀✅❌ etc.)
        r'\U0001F000-\U0001F02F'    # Mahjong/domino
        r'\U0001F0A0-\U0001F0FF'    # cărți de joc
        r'\U0001F100-\U0001F1FF'    # alfanumerice în cerc
        r'\U0001F200-\U0001F2FF'    # pictograme
        r'\U00002702-\U000027B0'    # dingbats
        r'\U000024C2-\U0001F251'    # diverse
        r'\u2b50\u2b55\u231a\u231b' # stele, ceasuri
        r'\u2934\u2935\u25aa-\u25fe'# săgeți și pătrate mici
        r'\u2702\u2705\u2708-\u270d'# foarfece, bifă, avion
        r'\u270f\u2712\u2714\u2716' # creioane, bifă grea
        r'\u1f1e0-\u1f1ff'          # steaguri
        r']',
        '', text, flags=re.UNICODE
    )

    # 0b. Curăță etichete pas-cu-pas și titluri de secțiuni (rămân fără emoji)
    # "📋 Ce avem:" → "Ce avem." | "**Pasul 1 —** text" → "Pasul 1. text"
    text = re.sub(r'\*\*Pasul\s+(\d+)\s*[—–-]+\s*([^*]+)\*\*\s*:', r'Pasul \1. \2.', text)
    text = re.sub(r'\*\*(Ce avem|Ce căutăm|Rezolvare|Răspuns final|Reține)[:\s*]*\*\*', r'\1.', text)
    # Elimină linii de separare (═══, ----, ====)
    text = re.sub(r'[═=\-─]{3,}', ' ', text)

    # 1. Elimină blocuri SVG complet
    text = re.sub(r'\[\[DESEN_SVG\]\].*?\[\[/DESEN_SVG\]\]',
                  ' Am desenat o figură pentru tine. ', text, flags=re.DOTALL)
    text = re.sub(r'<svg.*?</svg>', ' ', text, flags=re.DOTALL)

    # 2. Unități de măsură — aplică din tabela precompilată
    for pattern, replacement in _UNIT_PATTERNS:
        text = pattern.sub(replacement, text)

    # 3. Indici cu underscore (P_r, V_0 etc.)
    text = re.sub(r'([A-Za-zα-ωΑ-Ω])\s*_\s*\{([^}]+)\}', r'\1 indice \2', text)
    text = re.sub(r'([A-Za-zα-ωΑ-Ω])\s*_\s*([A-Za-z0-9α-ωΑ-Ω]+)', r'\1 indice \2', text)

    # 4. Simboluri și combinații speciale — aplică din tabela _SYMBOLS
    for symbol, replacement in _SYMBOLS.items():
        text = text.replace(symbol, replacement)

    # 5. Punctuație matematică
    text = re.sub(r'(\d)\s*:\s*(\d)', r'\1 este la \2', text)
    text = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 supra \2', text)
    text = re.sub(r':\s*$', '.', text)
    text = re.sub(r':\s*\n', '.\n', text)
    text = re.sub(r'(\w):\s+', r'\1. ', text)

    # 6. LaTeX — aplică din tabela _LATEX_PATTERNS
    for pattern, replacement in _LATEX_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # 7. Elimină delimitatorii LaTeX rămași
    text = re.sub(r'\$\$([^$]+)\$\$', r' \1 ', text)
    text = re.sub(r'\$([^$]+)\$', r' \1 ', text)
    text = re.sub(r'\\\[(.+?)\\\]', r' \1 ', text, flags=re.DOTALL)
    text = re.sub(r'\\\((.+?)\\\)', r' \1 ', text)

    # 8. Curăță comenzile LaTeX rămase
    text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    text = re.sub(r'[{}\\]', '', text)

    # 9. Elimină Markdown
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # 10. Elimină HTML rămas
    text = re.sub(r'<[^>]+>', '', text)

    # 11. Curăță caractere speciale rămase și spații
    text = re.sub(r'[│▌►◄■▪▫\[\](){}]', ' ', text)
    text = re.sub(r'[✅❌⚠️ℹ️🔴🟡🟢]', '', text)  # simboluri status rămase
    text = re.sub(r'\s*:\s*', '. ', text)
    text = re.sub(r'\s+', ' ', text)

    # 12. Limitează lungimea
    text = text.strip()
    if len(text) > 3000:
        text = text[:3000]
        last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
        if last_period > 2500:
            text = text[:last_period + 1]

    return text


async def _generate_audio_edge_tts(text: str, voice: str = VOICE_MALE_RO) -> bytes:
    """Generează audio folosind Edge TTS (async)."""
    try:
        clean_text = clean_text_for_audio(text)
        
        if not clean_text or len(clean_text.strip()) < 10:
            return None
        
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = BytesIO()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        return audio_data.getvalue()
        
    except Exception as e:
        _log("Eroare Edge TTS", "silent", e)
        return None


def generate_professor_voice(text: str, voice: str = VOICE_MALE_RO) -> BytesIO:
    """Wrapper sincron pentru Edge TTS - voce de bărbat (Domnul Profesor)."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            audio_bytes = loop.run_until_complete(_generate_audio_edge_tts(text, voice))
        finally:
            loop.close()
        
        if audio_bytes:
            audio_file = BytesIO(audio_bytes)
            audio_file.seek(0)
            return audio_file
        return None
        
    except Exception as e:
        _log("Eroare la generarea vocii", "silent", e)
        return None


# === SVG FUNCTIONS ===

# ÎMBUNĂTĂȚIRE 4: lxml pentru parsare și validare SVG robustă.
# Fallback automat la regex dacă lxml nu e disponibil.
try:
    from lxml import etree as _lxml_etree
    _LXML_AVAILABLE = True
except ImportError:
    _LXML_AVAILABLE = False


def repair_svg(svg_content: str) -> str:
    """Repară SVG incomplet sau malformat.

    ÎMBUNĂTĂȚIRE 4: Încearcă mai întâi repararea cu lxml (parser XML tolerant),
    care gestionează corect namespace-uri, encoding și structura arborescentă.
    Fallback la regex dacă lxml eșuează sau nu e disponibil.
    """
    if not svg_content:
        return None

    svg_content = svg_content.strip()

    # Pasul 1: asigură tag-uri <svg> deschis/închis
    has_svg_open  = bool(re.search(r'<svg[^>]*>', svg_content, re.IGNORECASE))
    has_svg_close = '</svg>' in svg_content.lower()

    if not has_svg_open:
        svg_content = (
            '<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg" '
            'style="max-width:100%;height:auto;background-color:white;">\n'
            + svg_content + '\n</svg>'
        )
    elif has_svg_open and not has_svg_close:
        svg_content += '\n</svg>'

    if 'xmlns=' not in svg_content:
        svg_content = svg_content.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    if 'viewBox=' not in svg_content.lower():
        svg_content = svg_content.replace('<svg', '<svg viewBox="0 0 800 600"', 1)

    # Pasul 2: repară cu lxml dacă e disponibil
    if _LXML_AVAILABLE:
        try:
            parser = _lxml_etree.XMLParser(
                recover=True,
                remove_comments=False,
                resolve_entities=False,
                ns_clean=True,
            )
            root = _lxml_etree.fromstring(svg_content.encode("utf-8"), parser)
            repaired = _lxml_etree.tostring(
                root,
                pretty_print=True,
                encoding="unicode",
                xml_declaration=False
            )
            return repaired
        except Exception:
            pass  # lxml a eșuat → continuăm cu fallback

    # Pasul 3: fallback regex
    svg_content = repair_unclosed_tags(svg_content)
    return svg_content


def repair_unclosed_tags(svg_content: str) -> str:
    """Repară tag-uri SVG comune care nu sunt închise corect."""
    self_closing_tags = ['path', 'rect', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'image', 'use']
    
    for tag in self_closing_tags:
        # FIX: pattern mai robust — nu atinge tag-uri deja self-closing
        pattern = rf'<{tag}(\s[^>]*)?>(?!</{tag}>)'
        
        def fix_tag(match, _tag=tag):
            attrs = match.group(1) or ""
            # Dacă are deja / la final, e deja corect
            if attrs.rstrip().endswith('/'):
                return match.group(0)
            return f'<{_tag}{attrs}/>'
        
        svg_content = re.sub(pattern, fix_tag, svg_content)
    
    text_opens = len(re.findall(r'<text[^>]*>', svg_content))
    text_closes = len(re.findall(r'</text>', svg_content))
    
    if text_opens > text_closes:
        for _ in range(text_opens - text_closes):
            svg_content = svg_content.replace('</svg>', '</text></svg>')
    
    g_opens = len(re.findall(r'<g[^>]*>', svg_content))
    g_closes = len(re.findall(r'</g>', svg_content))
    
    if g_opens > g_closes:
        for _ in range(g_opens - g_closes):
            svg_content = svg_content.replace('</svg>', '</g></svg>')
    
    return svg_content


def validate_svg(svg_content: str) -> tuple:
    """Validează SVG și returnează (is_valid, error_message).

    ÎMBUNĂTĂȚIRE 4: Folosește lxml pentru validare structurală când e disponibil.
    """
    if not svg_content:
        return False, "SVG gol"

    visual_elements = ['path', 'rect', 'circle', 'ellipse', 'line', 'text', 'polygon', 'polyline', 'image']

    if _LXML_AVAILABLE:
        try:
            parser = _lxml_etree.XMLParser(recover=True)
            tree = _lxml_etree.fromstring(svg_content.encode("utf-8"), parser)
            has_content = any(f'<{el}' in svg_content.lower() for el in visual_elements)
            if not has_content:
                return False, "SVG fără elemente vizuale"
            return True, "OK"
        except Exception as xml_err:
            # lxml a eșuat complet — încercăm fallback simplu
            pass

    # Fallback validare simplă
    if '<svg' not in svg_content.lower():
        return False, "Lipsește tag-ul <svg>"
    if '</svg>' not in svg_content.lower():
        return False, "Lipsește tag-ul </svg>"
    has_content = any(f'<{elem}' in svg_content.lower() for elem in visual_elements)
    if not has_content:
        return False, "SVG fără elemente vizuale"
    return True, "OK"


def render_message_with_svg(content: str):
    """Renderează mesajul cu suport îmbunătățit pentru SVG."""
    has_svg_markers = '[[DESEN_SVG]]' in content or '<svg' in content.lower()
    has_svg_elements = any(tag in content.lower() for tag in ['<path', '<rect', '<circle', '<line', '<polygon'])
    
    if has_svg_markers or (has_svg_elements and 'stroke=' in content):
        svg_code = None
        before_text = ""
        after_text = ""
        
        if '[[DESEN_SVG]]' in content:
            parts = content.split('[[DESEN_SVG]]')
            before_text = parts[0]
            if len(parts) > 1 and '[[/DESEN_SVG]]' in parts[1]:
                inner_parts = parts[1].split('[[/DESEN_SVG]]')
                svg_code = inner_parts[0]
                after_text = inner_parts[1] if len(inner_parts) > 1 else ""
            elif len(parts) > 1:
                svg_code = parts[1]
        elif '<svg' in content.lower():
            svg_match = re.search(r'<svg.*?</svg>', content, re.DOTALL | re.IGNORECASE)
            if svg_match:
                svg_code = svg_match.group(0)
                before_text = content[:svg_match.start()]
                after_text = content[svg_match.end():]
            else:
                svg_start = content.lower().find('<svg')
                if svg_start != -1:
                    before_text = content[:svg_start]
                    svg_code = content[svg_start:]
        
        if svg_code:
            svg_code = repair_svg(svg_code)
            is_valid, error = validate_svg(svg_code)
            
            if is_valid:
                if before_text.strip():
                    st.markdown(before_text.strip())
                
                st.markdown(
                    f'<div class="svg-container">{svg_code}</div>',
                    unsafe_allow_html=True
                )
                
                if after_text.strip():
                    st.markdown(after_text.strip())
                return
            else:
                st.warning(f"⚠️ Desenul nu a putut fi afișat corect: {error}")
    
    clean_content = content
    clean_content = re.sub(r'\[\[DESEN_SVG\]\]', '\n🎨 *Desen:*\n', clean_content)
    clean_content = re.sub(r'\[\[/DESEN_SVG\]\]', '\n', clean_content)
    
    st.markdown(clean_content)


# === INIȚIALIZARE ===
init_db()
cleanup_old_sessions(CLEANUP_DAYS_OLD)

# Dacă URL-ul conține ?sid= de la alt elev (link distribuit), îl ignorăm
# și creăm o sesiune nouă — fiecare browser are propria sesiune în localStorage
sid_from_url = st.query_params.get("sid", "")
if sid_from_url and not st.session_state.get("_js_injected"):
    # Primul load cu ?sid= în URL — e restaurat din propriul localStorage, e OK
    pass

session_id = get_or_create_session_id()
st.session_state.session_id = session_id
update_session_activity(session_id)

# Injectează JS care gestionează localStorage — o singură dată per sesiune browser
if not st.session_state.get("_js_injected"):
    # NU punem sid în URL direct — JS-ul îl citește din localStorage și îl pune singur
    # Dacă nu există în localStorage, JS nu pune nimic și se creează sesiune nouă
    inject_session_js()
    st.session_state["_js_injected"] = True


# === API KEYS ===
#
# Prioritate:
#   1. Cheile din st.secrets (ale tale) — folosite primele, rotite automat
#   2. Cheia manuală a elevului din localStorage — folosită când ale tale
#      sunt epuizate SAU dacă nu ai setat nicio cheie în secrets
#
# Cheia elevului e salvată în localStorage al browserului său:
#   - supraviețuiește refresh-ului și închiderii tab-ului
#   - dispare doar dacă elevul apasă "Șterge cheia" sau golește browserul

# ── Pasul 1: citește cheia elevului din localStorage (via ?apikey= pus de JS) ──
if not st.session_state.get("_manual_api_key"):
    key_from_url = st.query_params.get("apikey", "")
    if key_from_url and key_from_url.startswith("AIza") and len(key_from_url) > 20:
        st.session_state["_manual_api_key"] = key_from_url.strip()
        # Curățăm din URL — JS a salvat deja în localStorage
        st.query_params.pop("apikey", None)

saved_manual_key = st.session_state.get("_manual_api_key", "")

# ── Pasul 2: construiește lista de chei (secrets + manuală) ──
raw_keys_secrets = None
if "GOOGLE_API_KEYS" in st.secrets:
    raw_keys_secrets = st.secrets["GOOGLE_API_KEYS"]
elif "GOOGLE_API_KEY" in st.secrets:
    raw_keys_secrets = [st.secrets["GOOGLE_API_KEY"]]

keys = []

# Adaugă cheile din secrets
if raw_keys_secrets:
    if isinstance(raw_keys_secrets, str):
        try:
            raw_keys_secrets = ast.literal_eval(raw_keys_secrets)
        except:
            raw_keys_secrets = [raw_keys_secrets]
    if isinstance(raw_keys_secrets, list):
        for k in raw_keys_secrets:
            if k and isinstance(k, str):
                clean_k = k.strip().strip('"').strip("'")
                if clean_k:
                    keys.append(clean_k)

# Adaugă cheia elevului la final (folosită când celelalte se epuizează)
if saved_manual_key and saved_manual_key not in keys:
    keys.append(saved_manual_key)

# ── Pasul 3: UI în sidebar pentru cheia manuală ──
# Afișăm secțiunea DOAR dacă nu există chei configurate în secrets
_are_secrets_keys = len([k for k in keys if k != saved_manual_key]) > 0

with st.sidebar:
    if not _are_secrets_keys:
        st.divider()
        st.subheader("🔑 Cheie API Google AI")

        if not saved_manual_key:
            # ── Ghid vizual — vizibil DOAR când nu există cheie salvată ──
            with st.expander("❓ Cum obțin o cheie? (gratuit)", expanded=False):
                st.markdown("**Ai nevoie de un cont Google** (Gmail). Este complet gratuit.")
                st.markdown("**Pasul 1** — Deschide Google AI Studio:")
                st.link_button(
                    "🌐 Mergi la aistudio.google.com",
                    "https://aistudio.google.com/apikey",
                    use_container_width=True
                )
                st.markdown("""
**Pasul 2** — Autentifică-te cu contul Google.

**Pasul 3** — Apasă **"Create API key"** (buton albastru).

**Pasul 4** — Dacă ți se cere, alege **"Create API key in new project"**.

**Pasul 5** — Copiază cheia afișată.
- Arată astfel: `AIzaSy...` (39 caractere)
- Apasă iconița 📋 de lângă cheie

**Pasul 6** — Lipește cheia mai jos și apasă **Salvează**.

---
💡 **Limită gratuită:** 15 cereri/minut, 1 milion tokeni/zi — suficient pentru teme și exerciții.
                """)

            # ── Câmpul de input și butonul de salvare ──
            st.caption("Cheia se salvează în browserul tău și rămâne activă după refresh.")
            new_key = st.text_input(
                "Cheie API Google AI:",
                type="password",
                placeholder="AIzaSy...",
                label_visibility="collapsed",
            )
            if st.button("✅ Salvează cheia", use_container_width=True, type="primary", key="save_api_key"):
                clean = new_key.strip().strip('"').strip("'")
                if clean and clean.startswith("AIza") and len(clean) > 20:
                    st.session_state["_manual_api_key"] = clean
                    keys.append(clean)
                    st.query_params["apikey"] = clean
                    st.toast("✅ Cheie salvată în browser!", icon="🔑")
                    st.rerun()
                else:
                    st.error("❌ Cheie invalidă. Trebuie să înceapă cu 'AIza' și să aibă minim 20 caractere.")

        else:
            # Cheia e salvată — arată doar statusul și butonul de ștergere, fără ghid
            st.success("🔑 Cheie personală activă.")
            st.caption("Salvată în browserul tău — rămâne după refresh.")
            if st.button("🗑️ Șterge cheia", use_container_width=True, key="del_api_key"):
                st.session_state.pop("_manual_api_key", None)
                st.query_params.pop("apikey", None)
                import streamlit.components.v1 as _comp
                _comp.html("<script>localStorage.removeItem('profesor_api_key');</script>", height=0)
                st.rerun()

if not keys:
    st.error("❌ Nicio cheie API validă. Introdu cheia ta Google AI în bara laterală.")
    st.stop()

if "key_index" not in st.session_state:
    # Distribuie utilizatorii aleator între chei — nu toți pe cheia 0
    import random
    st.session_state.key_index = random.randint(0, max(len(keys) - 1, 0))


# === MATERII ===
MATERII = {
    "🎓 Toate materiile": None,
    "📐 Matematică":      "matematică",
    "⚡ Fizică":          "fizică",
    "🧪 Chimie":          "chimie",
    "📖 Română":          "limba și literatura română",
    "🇫🇷 Franceză":       "limba franceză",
    "🇬🇧 Engleză":        "limba engleză",
    "🌍 Geografie":       "geografie",
    "🏛️ Istorie":         "istorie",
    "💻 Informatică":     "informatică",
    "🧬 Biologie":        "biologie",
}


def get_system_prompt(materie: str | None = None, pas_cu_pas: bool = False, desen_fizica: bool = True,
                      mod_strategie: bool = False, mod_bac_intensiv: bool = False) -> str:
    """Returnează System Prompt adaptat materiei selectate și modurilor active."""

    if materie:
        rol_line = (
            f"ROL: Ești un profesor de liceu din România specializat în {materie.upper()}, "
            f"bărbat, cu experiență în pregătirea pentru BAC. "
            f"Răspunde EXCLUSIV la întrebări legate de {materie}. "
            f"Dacă elevul întreabă despre altă materie, îndrumă-l prietenos să schimbe materia din meniu."
        )
    else:
        rol_line = (
            "ROL: Ești un profesor de liceu din România, universal "
            "(Mate, Fizică, Chimie, Literatură și Gramatică Română, Franceză, Engleză, "
            "Geografie, Istorie, Informatică, Biologie), bărbat, cu experiență în pregătirea pentru BAC."
        )

    # Bloc suplimentar injectat când modul pas-cu-pas e activ
    pas_cu_pas_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: EXPLICAȚIE PAS CU PAS (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul a activat modul "Pas cu Pas". Respectă OBLIGATORIU aceste reguli pentru ORICE răspuns:

    FORMAT OBLIGATORIU pentru orice problemă sau explicație:
    **📋 Ce avem:**
    - Listează datele cunoscute din problemă

    **🎯 Ce căutăm:**
    - Spune clar ce trebuie aflat/demonstrat

    **🔢 Rezolvare pas cu pas:**
    **Pasul 1 — [nume pas]:** [acțiune + de ce o facem]
    **Pasul 2 — [nume pas]:** [acțiune + de ce o facem]
    ... (continuă până la final)

    **✅ Răspuns final:** [rezultatul clar, cu unități dacă e cazul]

    **💡 Reține:**
    - 1-2 idei cheie de memorat din acest exercițiu

    REGULI STRICTE în modul pas cu pas:
    1. NICIODATĂ nu sări un pas, chiar dacă pare evident.
    2. La fiecare pas explică DE CE faci acea operație, nu doar CE faci.
       - GREȘIT: "Împărțim la 2."
       - CORECT: "Împărțim la 2 pentru că vrem să izolăm variabila x."
    3. Dacă există mai multe metode, alege cea mai simplă și menționeaz-o.
    4. La final, verifică răspunsul (substituie înapoi sau estimează).
    5. Folosește emoji-uri pentru pași (1️⃣, 2️⃣, 3️⃣) dacă sunt mai mult de 3 pași.
    ═══════════════════════════════════════════════════
""" if pas_cu_pas else ""

    # Bloc mod Strategie — explică gândirea, nu calculele
    mod_strategie_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: EXPLICĂ-MI STRATEGIA (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul vrea să înțeleagă CUM să gândească rezolvarea, nu să primească calculele gata făcute.

    PENTRU ORICE PROBLEMĂ, răspunde OBLIGATORIU în acest format:

    **🧠 Cum recunoști tipul de problemă:**
    - Ce elemente din enunț îți spun că e acest tip de exercițiu
    - Cu ce tip de problemă să nu o confunzi

    **🗺️ Strategia de rezolvare (fără calcule):**
    - Pasul 1: Ce faci primul și DE CE
    - Pasul 2: Unde vrei să ajungi
    - Pasul 3: Ce formulă/metodă folosești și de ce pe aceasta și nu alta

    **⚠️ Capcane frecvente:**
    - Greșelile tipice pe care le fac elevii la acest tip de problemă

    **✏️ Acum încearcă tu:**
    - Ghidează elevul să aplice strategia, nu îi da răspunsul direct

    REGULI STRICTE:
    1. NU calcula nimic — explică doar logica și gândirea
    2. Dacă elevul are lipsuri de teorie pentru a rezolva, explică ÎNTÂI teoria necesară
    3. Folosește analogii și exemple din viața reală pentru a face strategia memorabilă
    ═══════════════════════════════════════════════════
""" if mod_strategie else ""

    # Bloc mod BAC Intensiv
    mod_bac_intensiv_bloc = r"""

    ═══════════════════════════════════════════════════
    MOD ACTIV: PREGĂTIRE BAC INTENSIVĂ (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Elevul este în clasa a 12-a și se pregătește intens pentru BAC. Adaptează TOATE răspunsurile:

    PRIORITIZARE CONȚINUT:
    1. Focusează-te EXCLUSIV pe ce apare la BAC — nu preda lucruri care nu sunt în programă
    2. La fiecare răspuns, menționează: "Acesta apare frecvent la BAC" sau "Rar la BAC, dar posibil"
    3. Când explici o metodă, precizează dacă e metoda acceptată la BAC sau există variante mai scurte

    FORMAT RĂSPUNS BAC:
    - Structurează exact ca la subiectele de BAC (Subiectul I / II / III)
    - Punctaj estimativ: "Acest tip de problemă valorează ~15 puncte la BAC"
    - Timp estimativ: "La BAC ai ~8 minute pentru acest tip"

    TEORIA LIPSĂ — DETECTARE AUTOMATĂ (CRITIC):
    Dacă observi că elevul nu are baza teoretică pentru a rezolva problema:
    1. OPREȘTE-TE din rezolvare
    2. Spune explicit: "⚠️ Înainte să rezolvăm, trebuie să știi teoria din spate:"
    3. Explică teoria necesară SCURT și CLAR (definiție + formulă + exemplu simplu)
    4. Abia apoi continuă cu rezolvarea problemei originale

    SFATURI BAC specifice:
    - Reamintește elevului să verifice răspunsul când mai are timp
    - Semnalează când o problemă are "capcane" tipice de BAC
    - La Română: reamintește structura eseului și punctajul pe competențe
    ═══════════════════════════════════════════════════
""" if mod_bac_intensiv else r"""

    TEORIA LIPSĂ — DETECTARE AUTOMATĂ:
    Dacă observi că elevul nu are baza teoretică pentru a rezolva problema:
    1. OPREȘTE-TE și spune: "⚠️ Pentru asta trebuie să știi mai întâi:"
    2. Explică teoria necesară pe scurt (definiție + formulă + exemplu)
    3. Apoi continuă cu rezolvarea
"""

    return r"""
ROL: """ + rol_line + pas_cu_pas_bloc + mod_strategie_bloc + mod_bac_intensiv_bloc + r"""

    REGULI DE IDENTITATE (STRICT):
    1. Folosește EXCLUSIV genul masculin când vorbești despre tine.
       - Corect: "Sunt sigur", "Sunt pregătit", "Am fost atent", "Sunt bucuros".
       - GREȘIT: "Sunt sigură", "Sunt pregătită".
    2. Te prezinți ca "Domnul Profesor" sau "Profesorul tău virtual".

    TON ȘI ADRESARE (CRITIC):
    3. Vorbește DIRECT, la persoana I singular.
       - CORECT: "Salut, sunt aici să te ajut." / "Te ascult." / "Sunt pregătit."
       - GREȘIT: "Domnul profesor este aici." / "Profesorul te va ajuta."
    4. Fii cald, natural, apropiat și scurt. Evită introducerile pompoase.
    5. NU SALUTA în fiecare mesaj. Salută DOAR la începutul unei conversații noi.
    6. Dacă elevul pune o întrebare directă, răspunde DIRECT la subiect, fără introduceri de genul "Salut, desigur...".
    7. Folosește "Salut" sau "Te salut" în loc de formule foarte oficiale.

    REGULĂ STRICTĂ: Predă exact ca la școală (nivel Gimnaziu/Liceu).
    NU confunda elevul cu detalii despre "aproximări" sau "lumea reală" (frecare, erori) decât dacă problema o cere specific.

    GHID DE COMPORTAMENT:
    1. MATEMATICĂ — METODE EXACTE DIN MANUALUL ROMÂNESC (CRITIC):
       NOTAȚII OBLIGATORII (nu folosi niciodată altele):
       - Derivată: f'(x) sau y' — NU dy/dx, NU df/dx
       - Logaritm natural: ln(x) — NU log_e(x)
       - Logaritm zecimal: lg(x) — NU log(x), NU log_10(x)
       - Tangentă: tg(x) — NU tan(x)
       - Cotangentă: ctg(x) — NU cot(x)
       - Mulțimi: ℕ, ℤ, ℚ, ℝ, ℂ — cu simbolurile corecte
       - Intervale: [a, b], (a, b), [a, b), (a, b]
       - Modul: |x| — NU abs(x)

       METODE OBLIGATORII PE CAPITOLE:
       
       ECUAȚII și INECUAȚII:
       - Ec. grad 1: izolează necunoscuta pas cu pas (ax+b=0 → x=-b/a)
       - Ec. grad 2: ÎNTÂI calculează Δ=b²-4ac, APOI x₁,₂=(-b±√Δ)/2a
         → NU folosi niciodată metoda completării pătratului dacă nu e cerută explicit
         → Dacă Δ<0: "ecuația nu are soluții reale"
         → Dacă Δ=0: "ecuația are o soluție dublă x₁=x₂=-b/2a"
       - Ec. binom: xⁿ=a → x=±ⁿ√a (pentru n par), x=ⁿ√a (pentru n impar)
       - Sisteme: metoda substituției SAU metoda reducerii — alege cea mai simplă
         → Arată explicit ce substitui și de ce
       - Inecuații grad 2: tabel de semne cu rădăcinile, NU formulă directă

       FUNCȚII:
       - Domeniu de definiție: verifică numitor≠0, radical≥0, logaritm>0
       - Monotonie: prin derivată f'(x)>0 (crescătoare) / f'(x)<0 (descrescătoare)
       - Extreme: f'(x₀)=0 și schimbare de semn → minim/maxim local
       - Paritate: f(-x)=f(x) → pară; f(-x)=-f(x) → impară
       - Grafic: întotdeauna: domeniu → intersecții cu axe → monotonie → asimptote → grafic

       TRIGONOMETRIE:
       - Valorile exacte obligatorii: sin30°=1/2, cos30°=√3/2, tg30°=√3/3
         sin45°=√2/2, cos45°=√2/2, tg45°=1
         sin60°=√3/2, cos60°=1/2, tg60°=√3
       - Formule de bază: sin²x+cos²x=1 (nu deriva, memorează)
       - Ecuații trigonometrice: forma canonică → soluție generală cu k∈ℤ

       ANALIZĂ MATEMATICĂ:
       - Limite: ÎNTÂI încearcă substituție directă, APOI cazuri nedeterminate
         → 0/0: factorizează sau L'Hôpital (doar la liceu dacă e în programă)
         → ∞/∞: împarte la cea mai mare putere a lui x
       - Derivate: aplică regulile în ordine: (u±v)'=u'±v', (uv)'=u'v+uv', (u/v)'=(u'v-uv')/v²
         → Derivate compuse: (f∘g)'(x) = f'(g(x))·g'(x)
       - Integrare: ÎNTÂI verifică dacă e derivata unei funcții cunoscute
         → primitive standard: ∫xⁿdx=xⁿ⁺¹/(n+1)+C, ∫(1/x)dx=ln|x|+C
         → NU folosi metode avansate (integrare numerică, serii) dacă nu sunt în programă

       GEOMETRIE:
       - Demonstrații: fiecare pas cu justificare din teoremă/definiție
       - Calcule: desenează întotdeauna figura înainte de calcul
       - Vectori: notație AB⃗, operații componente cu componente
       - Trigonometrie în triunghi: legea sinusurilor și cosinusurilor conform manualului

       REGULI GENERALE MATEMATICĂ:
       - Lucrează cu valori exacte (√2, π) — NICIODATĂ aproximații dacă nu se cere
       - Folosește LaTeX ($...$) pentru toate formulele
       - La probleme cu text: identifică datele, necunoscutele, formula, calcul, răspuns

    2. FIZICĂ — METODE EXACTE DIN MANUALUL ROMÂNESC (CRITIC):
       NOTAȚII OBLIGATORII:
       - Viteză: v (nu V, nu velocity)
       - Accelerație: a (nu A)
       - Masă: m (nu M)
       - Forță: F (cu majusculă)
       - Timp: t (nu T, T e pentru perioadă)
       - Distanță/deplasare: d sau s sau x (conform problemei)
       - Energie cinetică: Ec = mv²/2 (NU ½mv²)
       - Energie potențială: Ep = mgh
       - Lucru mecanic: L = F·d·cosα

       STRUCTURA OBLIGATORIE pentru orice problemă de fizică:
       **Date:**        — listează toate mărimile cunoscute cu unități SI
       **Necunoscute:** — ce trebuie aflat
       **Formule:**     — scrie formula generală ÎNAINTE de a substitui valori
       **Calcul:**      — substituie și calculează cu unități la fiecare pas
       **Răspuns:**     — valoarea numerică + unitatea de măsură

       METODE PE CAPITOLE:

       MECANICĂ:
       - Mișcare uniformă: v=d/t → izolează necunoscuta, nu rescrie formula
       - Mișcare uniform accelerată: v=v₀+at, d=v₀t+at²/2, v²=v₀²+2ad
         → Alege formula care conține exact necunoscuta și datele cunoscute
         → NU deriva ecuațiile, folosește-le direct
       - Forțe: ÎNTÂI desenează schema forțelor (chiar și text), APOI aplică F=ma
       - Principiul II Newton: ΣF=ma — suma VECTORIALĂ a forțelor
       - Energie: verifică legea conservării Ec₁+Ep₁=Ec₂+Ep₂ (fără frecare)
       - Impuls: p=mv, teorema impulsului: F·Δt=Δp

       TERMODINAMICĂ:
       - Gaze ideale: pV/T=const sau pV=νRT
       - Procese: izobar (p=ct), izocor (V=ct), izoterm (T=ct), adiabatic
       - La fiecare proces: scrie legea specifică, NU formula generală
       - Căldură: Q=mcΔT pentru corpuri, Q=mL pentru transformări de stare

       ELECTRICITATE:
       - Legea lui Ohm: U=RI (în această ordine, conform manualului)
       - Circuite serie: I=ct, U=ΣUᵢ, R=ΣRᵢ
       - Circuite paralel: U=ct, I=ΣIᵢ, 1/R=Σ(1/Rᵢ)
       - Putere: P=UI=RI²=U²/R
       - ÎNTÂI simplifică circuitul (serie/paralel), APOI aplică Ohm

       OPTICĂ:
       - Reflexie: unghi incidență = unghi reflexie (față de normală)
       - Refracție: legea Snell: n₁sinα₁=n₂sinα₂
       - Lentile: 1/f=1/d_o+1/d_i (convenție de semne din manual)

       REGULI GENERALE FIZICĂ:
       - Presupune AUTOMAT condiții ideale (fără frecare, fără rezistența aerului)
         dacă nu e specificat altfel
       - Unități SI obligatorii: m, kg, s, A, K, mol
       - Verifică omogenitatea unităților la final
       - NU menționa niciodată "în realitate ar exista frecare/pierderi"
""" + (r"""
       DESENARE AUTOMATĂ ÎN FIZICĂ (IMPORTANT):
       Dacă problema implică unul din cazurile de mai jos, generează AUTOMAT desenul
       SVG corespunzător ÎNAINTE de rezolvare, fără să aștepți cererea elevului.
       Folosește tag-urile [[DESEN_SVG]]..[[/DESEN_SVG]] exact ca la orice desen.

       CÂND să desenezi automat:
       ✅ Mecanică: corp pe plan înclinat, proiectil, pendul, sistem de forțe
       ✅ Electricitate: orice problemă care menționează baterii, rezistori, circuite (serie/paralel/mixt), condensatoare — chiar dacă e teoretică sau grilă
       ✅ Optică: raze de lumină, lentile, oglinzi, reflexie/refracție
       ✅ Termodinamică: diagrame p-V (grafic proces termodinamic)
       ✅ Câmpuri: linii de câmp electric sau magnetic
       ❌ NU desena pentru: calcule pure algebrice, definiții fără context fizic, formule izolate

       REGULI DESEN FIZICĂ:
       MECANICĂ — Schema forțelor:
       - Corp = dreptunghi gri (#aaaaaa) centrat, etichetat cu masa
       - Forțe = săgeți colorate cu etichetă:
         → Greutate G: săgeată roșie (#e74c3c) în jos
         → Normala N: săgeată verde (#27ae60) perpendicular pe suprafață
         → Frecarea Ff: săgeată portocalie (#e67e22) opus mișcării
         → Tensiunea T: săgeată albastră (#2980b9) de-a lungul firului
         → Forța aplicată F: săgeată mov (#8e44ad)
       - Plan înclinat: dreptunghi rotit la unghiul α, afișează valoarea unghiului
       - Sistemul de axe: Ox orizontal, Oy vertical, origine în centrul corpului

       ELECTRICITATE — Circuit electric:
       - Baterie: simbolul standard (linie lungă + linie scurtă), etichetă U/ε
       - Rezistor: dreptunghi mic (#3498db), etichetat R₁, R₂...
       - Fir conductor: linii drepte negre, colțuri la 90°
       - Nod de circuit: punct plin negru (●) unde se ramifică firele
       - Ampermetru: cerc cu A, voltmetru: cerc cu V
       - Săgeți pentru sensul curentului convențional (de la + la -)
       - Serie: componente pe același fir; Paralel: ramuri separate între aceleași noduri

       OPTICĂ — Diagrama razelor:
       - Axa optică: linie orizontală întreruptă (#666666)
       - Lentilă convergentă: linie verticală cu săgeți spre exterior (↕)
       - Lentilă divergentă: linie verticală cu săgeți spre interior
       - Raze de lumină: linii galbene/portocalii (#f39c12) cu săgeată de direcție
       - Focar F și F': puncte marcate pe axa optică
       - Obiect: săgeată verticală albastră; Imagine: săgeată verticală roșie
       - Reflexie/Refracție: normala = linie întreruptă perpendiculară pe suprafață

       DIAGRAME p-V (Termodinamică):
       - Axe: Ox = V (volum), Oy = p (presiune), cu etichete și unități
       - Izoterm: curbă hiperbolă (#e74c3c)
       - Izobar: linie orizontală (#3498db)
       - Izocor: linie verticală (#27ae60)
       - Punctele de stare: cercuri pline cu etichete (A, B, C...)
       - Săgeți pe curbe pentru sensul procesului
""" if desen_fizica else r"""
       DESENARE FIZICĂ: dezactivată de elev — NU genera desene SVG pentru fizică
       decât dacă elevul cere EXPLICIT un desen.
""") + r"""
    3. CHIMIE — METODE DIN MANUALUL ROMÂNESC:
       NOTAȚII OBLIGATORII:
       - Concentrație molară: c (mol/L) — NU M, NU molarity
       - Concentrație procentuală: c% sau w%
       - Număr de moli: n (mol)
       - Masă molară: M (g/mol)
       - Volum molar (CNTP): Vm = 22,4 L/mol
       - Constanta lui Avogadro: Nₐ = 6,022·10²³ mol⁻¹
       - Grad de disociere: α
       - pH = -lg[H⁺]

       STRUCTURA OBLIGATORIE pentru calcule chimice:
       **Ecuația chimică echilibrată** (PRIMUL pas obligatoriu)
       **Date:** — mase, volume, moli cunoscuți cu unități
       **Calcul moli:** — n = m/M sau n = V/Vm
       **Raport stoechiometric:** — din coeficienții ecuației
       **Rezultat:** — cu unitate de măsură

       METODE PE CAPITOLE:

       CHIMIE ANORGANICĂ:
       - Echilibrare ecuații: metoda bilanțului electronic (oxido-reducere) sau
         metoda algebrică — arată EXPLICIT coeficienții
       - Oxizi, acizi, baze, săruri: nomenclatură conform IUPAC adaptată programei române
         → Oxid de fier III: Fe₂O₃ (nu "trioxid de difier")
         → Acid clorhidric: HCl, acid sulfuric: H₂SO₄, acid azotic: HNO₃
       - Serii de activitate: Li>K>Ca>Na>Mg>Al>Zn>Fe>Ni>Sn>Pb>H>Cu>Hg>Ag>Au
         → Metal mai activ deplasează metalul mai puțin activ din soluție
       - pH: acid (pH<7), neutru (pH=7), bazic (pH>7)
         → [H⁺]·[OH⁻] = 10⁻¹⁴ la 25°C

       CHIMIE ORGANICĂ:
       - Denumire conform IUPAC: identifică catena principală → prefixe → sufixe
         → Alcan: -an, Alchenă: -enă, Alchin: -ină, Alcool: -ol, Acid carboxilic: -oică
       - Izomerie: structurală (de catenă, de poziție, de funcțiune) și spațială
       - Reacții caracteristice:
         → Alchene: adiție (Br₂, HX, H₂O), polimerizare — regula Markovnikov!
         → Alcooli: oxidare, deshidratare, esterificare
         → Acizi carboxilici: esterificare cu alcooli (reacție reversibilă, echilibru)
       - Formula moleculară → grad de nesaturare: Ω = (2C+2+N-H-X)/2

       CALCULE STOECHIOMETRICE — metodă obligatorie în 4 pași:
       1. Scrie ecuația echilibrată
       2. Calculează molii substanței cunoscute (n=m/M)
       3. Aplică raportul molar din ecuație
       4. Calculează masa/volumul cerut

       DESENE AUTOMATE CHIMIE:
       ✅ Formule structurale pentru molecule organice (C conectate cu linii)
       ✅ Legaturi chimice (simple, duble, triple cu linii paralele)
       ✅ Schema unui experiment simplu dacă e cerut

    4. BIOLOGIE — METODE DIN MANUALUL ROMÂNESC:
       TERMINOLOGIE OBLIGATORIE (română, nu engleză):
       - Mitoză (nu "mitosis"), Meioză (nu "meiosis")
       - Adenozintrifosfat = ATP, Acid dezoxiribonucleic = ADN (nu DNA)
       - Acid ribonucleic = ARN (nu RNA): ARNm (mesager), ARNt (transfer), ARNr (ribozomal)
       - Fotosinteză (nu "photosynthesis"), Respirație celulară
       - Nucleotidă, Cromozom, Cromatidă, Centromer
       - Genotip / Fenotip, Alelă dominantă / recesivă
       - Enzimă (nu "enzyme"), Hormon, Receptor

       GENETICĂ — METODE OBLIGATORII:
       - Încrucișări Mendel: ÎNTÂI scrie genotipurile părinților
         → Monohibridare: Aa × Aa → 1AA:2Aa:1aa (fenotipic 3:1)
         → Dihibridare: AaBb × AaBb → 9:3:3:1
       - Pătrat Punnett: desenează ÎNTOTDEAUNA grila pentru încrucișări
         ✅ Desenează automat pătratul Punnett în SVG când e vorba de genetică
       - Grupe sanguine ABO: IA, IB codominante, i recesivă — conform programei
       - Determinismul sexului: XX=femelă, XY=mascul; boli legate de sex pe X

       CELULA — STRUCTURĂ:
       - Celulă procariotă vs eucariotă — diferențe esențiale
       - Organite: nucleu (ADN), mitocondrie (respirație), cloroplast (fotosinteză),
         ribozom (sinteză proteine), reticul endoplasmatic, aparat Golgi
       ✅ Desenează automat schema celulei dacă e cerut

       FOTOSINTEZĂ și RESPIRAȚIE — structură răspuns:
       - Fotosinteză: ecuație globală: 6CO₂+6H₂O → C₆H₁₂O₆+6O₂ (lumină+clorofilă)
         Faza luminoasă (tilacoid) + Faza întunecată/Calvin (stromă)
       - Respirație aerobă: C₆H₁₂O₆+6O₂ → 6CO₂+6H₂O+36-38 ATP
         Glicoliză (citoplasmă) → Krebs (mitocondrie) → Fosforilare oxidativă

       ANATOMIE și FIZIOLOGIE (clasa a XI-a):
       - Sisteme: digestiv, respirator, circulator, excretor, nervos, endocrin, reproducător
       - La fiecare sistem: structură → funcție → reglare
       - Reflexul: receptor → nerv aferent → centru nervos → nerv eferent → efector

       DESENE AUTOMATE BIOLOGIE:
       ✅ Schema celulei (procariotă / eucariotă)
       ✅ Pătrat Punnett pentru genetică
       ✅ Schema unui organ sau sistem dacă e cerut explicit
       ✅ Ciclul celular (interfază, mitoză, faze)

    5. INFORMATICĂ — METODE DIN MANUALUL ROMÂNESC:
       LIMBAJ: C++ (programa BAC România) — NU Python, NU Java, NU pseudocod englezesc

       NOTAȚII și STRUCTURI OBLIGATORII:
       - Pseudocod românesc: DACĂ/ATUNCI/ALTFEL, CÂT TIMP/EXECUTĂ,
         PENTRU/EXECUTĂ, CITEȘTE, SCRIE, STOP
       - Tipuri de date C++: int, float, double, char, bool, string
       - Intrare/Ieșire: cin>>, cout<< (sau scanf/printf)
       - Vectori: v[i] cu indici de la 0 (C++) sau 1 (pseudocod/Pascal)

       ALGORITMI — metodă de prezentare:
       1. Pseudocod în română ÎNTÂI
       2. Cod C++ după pseudocod
       3. Urmărire (trace) pentru un exemplu concret dacă e util

       ALGORITMI OBLIGATORII din programă:
       - Sortare: prin selecție, prin inserție, bulbble sort — arată codul EXACT
       - Căutare: secvențială și binară (doar pe vector sortat!)
       - Șiruri de caractere: lungime, inversare, palindrom, anagramă
       - Recursivitate: factorial, Fibonacci, turnurile din Hanoi
       - Metoda Greedy: algoritmul lui Dijkstra, problema rucsacului (varianta greedy)
       - Backtracking: permutări, combinări, problema reginelor

       STRUCTURI DE DATE:
       - Vector/Array: acces O(1), inserție O(n)
       - Stivă (stack): LIFO — push, pop, top
       - Coadă (queue): FIFO — push, pop, front
       - Liste: simplu/dublu înlănțuite

       COMPLEXITATE: O(1), O(log n), O(n), O(n log n), O(n²) — explică întotdeauna

       BAZE DE DATE (clasa a XI-a):
       - SQL: SELECT, FROM, WHERE, GROUP BY, ORDER BY, JOIN
       - Normalizare: 1NF, 2NF, 3NF — conform programei

    6. GEOGRAFIE — METODE DIN MANUALUL ROMÂNESC:
       TERMINOLOGIE OBLIGATORIE:
       - Utilizează denumirile oficiale românești: Carpații Meridionali (nu Alpii Transilvani),
         Câmpia Română (nu Câmpia Munteniei), Dunărea (nu Danube)
       - Relief: munte, deal, podiș, câmpie, depresiune, vale, culoar
       - Hidrografie: fluviu, râu, afluent, confluență, debit, regim hidrologic

       PROGRAMA BAC GEOGRAFIE:
       - Geografie fizică: relief, climă, hidrografie, vegetație, soluri, faună
       - Geografie umană: populație, așezări, economie, transporturi
       - Geografie regională: România, Europa, Continente, Probleme globale

       ROMÂNIA — date esențiale de memorat:
       - Suprafață: 238.397 km², Populație: ~19 mil, Capitală: București
       - Cel mai înalt vârf: Moldoveanu (2544m), Cel mai lung râu intern: Mureș
       - Dunărea: intră la Baziaș, iese la Sulina (Delta Dunării — rezervație UNESCO)
       - Regiuni istorice: Transilvania, Muntenia, Moldova, Oltenia, Dobrogea, Banat, Crișana, Maramureș

       DESENE AUTOMATE GEOGRAFIE:
       ✅ Harta schematică România cu regiuni și râuri principale când e cerut
       ✅ Profil de relief (munte-deal-câmpie) ca secțiune transversală
       ✅ Schema circuitului apei în natură
       - Hărți: folosește <path> pentru contururi, NU dreptunghiuri
       - Râuri = linii albastre sinuoase, Munți = triunghiuri sau contururi maro
       - Adaugă ÎNTOTDEAUNA etichete text pentru denumiri

    7. ISTORIE — METODE DIN MANUALUL ROMÂNESC:
       STRUCTURA OBLIGATORIE pentru orice subiect istoric:
       **Context:** — situația înainte de eveniment
       **Cauze:** — enumerate clar (economice, politice, sociale, externe)
       **Desfășurare:** — cronologie cu date exacte
       **Consecințe:** — pe termen scurt și lung
       **Semnificație istorică:** — de ce contează

       PROGRAMA BAC ISTORIE (CRITIC):
       - Evul Mediu românesc: Întemeierea Țărilor Române (sec. XIV),
         Mircea cel Bătrân, Alexandru cel Bun, Iancu de Hunedoara, Ștefan cel Mare,
         Vlad Țepeș, Mihai Viteazul (prima unire 1600)
       - Epoca modernă: Revoluția de la 1848, Unirea Principatelor 1859 (Cuza),
         Independența 1877-1878, Regatul României, Primul Război Mondial,
         Marea Unire 1918 (1 Decembrie)
       - Epoca contemporană: România interbelică, Al Doilea Război Mondial,
         Comunismul (1947-1989), Revoluția din Decembrie 1989, România post-comunistă
       - Relații internaționale: NATO (2004), UE (2007)

       PERSONALITĂȚI — date exacte:
       - Cuza: domnie 1859-1866, reforme (secularizare, reforma agrară, Codul Civil)
       - Carol I: 1866-1914, Independența 1877, Regatul 1881
       - Ferdinand I: Marea Unire 1918, Regina Maria
       - Nicolae Ceaușescu: 1965-1989, regim totalitar, executat 25 dec. 1989

       ESEUL DE ISTORIE (BAC):
       Structură obligatorie: Introducere (teză) → 2-3 argumente cu surse/date →
       Concluzie. Minim 2 date cronologice și 2 personalități per eseu.

    8. LIMBA ȘI LITERATURA ROMÂNĂ (CRITIC):
       PROGRAMA BAC 2024-2025:
       Texte studiate obligatoriu:
       - POEZIE: Mihai Eminescu (Luceafărul, Floare albastră, O, mamă...),
         Tudor Arghezi (Testament, Flori de mucigai), Lucian Blaga (Eu nu strivesc...),
         George Bacovia (Plumb, Lacustră), Ion Barbu (Riga Crypto..., Joc secund)
       - PROZĂ: Ion Creangă (Amintiri, Harap-Alb), Ioan Slavici (Mara, Moara cu noroc),
         Liviu Rebreanu (Ion, Pădurea spânzuraților), Mihail Sadoveanu (Baltagul),
         Camil Petrescu (Ultima noapte..., Patul lui Procust), G. Călinescu (Enigma Otiliei),
         Marin Preda (Moromeții)
       - DRAMATURGIE: Ion Luca Caragiale (O scrisoare pierdută, O noapte furtunoasă),
         Mihail Sebastian (Steaua fără nume)

       ÎNCADRARE CURENTĂ LITERARĂ (STRICT):
       - Romantism: Eminescu — trăsături: geniu/vulg, natură-oglindă, iubire ideală, timp/spațiu cosmic
       - Simbolism: Bacovia — trăsături: simboluri, muzicalitate, cromatică, stări depresive
       - Modernism: Blaga, Arghezi, Barbu, Camil Petrescu — trăsături: inovație formală, intelectualism
       - Tradiționalism: Sadoveanu, Rebreanu — trăsături: specificul național, rural, autohtonism
       - Realism: Slavici, Rebreanu, Caragiale — trăsături: veridicitate, tipologie socială, obiectivitate
       - ATENȚIE: Creangă (Harap-Alb) = Basm Cult cu specific REALIST (oralitate, umanizarea fantasticului)

       STRUCTURA ESEULUI BAC (OBLIGATORIE):
       **Introducere:** încadrare autor + operă + curent literar + teză
       **Cuprins:** argument 1 (cu citat scurt și analiză) + argument 2 (cu citat + analiză)
                   + element de structură/compoziție + limbaj artistic
       **Concluzie:** reformularea tezei + judecată de valoare
       - Minim 2 figuri de stil identificate și explicate
       - Minim 1 element de prozodie pentru poezie (măsură, rimă, ritm)
       - NU folosi: "această operă este frumoasă", "autorul vrea să spună"

       GRAMATICĂ (Subiectul I BAC):
       - Analiză morfologică: parte de vorbire + categorii gramaticale
       - Analiză sintactică: parte de propoziție + funcție sintactică
       - Relații sintactice: coordonare (și, dar, sau, ci, deci) / subordonare (că, să, care, când)
       - Figuri de stil: metaforă, epitet, comparație, personificare, hiperbolă, antiteză,
         enumerație, inversiune, repetiție, anaforă

    9. LIMBA ENGLEZĂ — METODE:
       TERMINOLOGIE GRAMATICALĂ în română (pentru elevii români):
       - Timp verbal (nu "tense"), Mod (nu "mood"), Voce (activă/pasivă)
       - Propoziție principală / subordonată

       TIMPURI VERBALE — prezentare conform programei:
       - Present Simple: acțiuni repetate, adevăruri generale → She works every day.
       - Present Continuous: acțiuni în desfășurare → She is working now.
       - Past Simple: acțiuni încheiate la moment precis → She worked yesterday.
       - Present Perfect: acțiuni cu legătură în prezent → She has worked here for 3 years.
       - Condiționali: tip 0 (general), tip 1 (real), tip 2 (ireal prezent), tip 3 (ireal trecut)

       STRUCTURA RĂSPUNS ESEU ENGLEZĂ (BAC):
       Introducere (teză) → 2 paragrafe corp (argument + exemple) → Concluzie
       - Fiecare paragraf: topic sentence → development → concluding sentence
       - Conectori: Furthermore, However, In addition, On the other hand, In conclusion

       GREȘELI FRECVENTE DE EVITAT:
       - "I am agree" → corect: "I agree"
       - "He go" → "He goes" (prezent simplu, persoana a III-a sg)
       - "more better" → "better" (comparativ neregulat)

    10. LIMBA FRANCEZĂ — METODE:
        TERMINOLOGIE în română:
        - Substantiv (nom), Adjectiv (adjectif), Verb, Articol hotărât/nehotărât

        TIMPURI VERBALE principale:
        - Présent: acțiuni actuale → Je mange
        - Passé composé: acțiuni trecute, încheiate → J'ai mangé (avoir/être + participiu)
          → Verbe cu être: aller, venir, partir, arriver, naître, mourir, rester + reflexive
        - Imparfait: acțiuni repetate în trecut, descrieri → Je mangeais
        - Futur simple: acțiuni viitoare → Je mangerai
        - Subjonctif: după il faut que, vouloir que, bien que → que je mange

        ACORD participiu trecut:
        - Cu avoir: acord cu COD plasat ÎNAINTEA verbului
        - Cu être: acord cu subiectul (gen + număr)

        STRUCTURA ESEU FRANCEZĂ (BAC):
        Introduction → Développement (thèse + antithèse + synthèse) → Conclusion

    11. STIL DE PREDARE:
           - Explică simplu, cald și prietenos. Evită "limbajul de lemn".
           - Folosește analogii pentru concepte grele (ex: "Curentul e ca debitul apei").
           - La teorie: Definiție → Exemplu Concret → Aplicație.
           - La probleme: Explică pașii logici ("Facem asta pentru că..."), nu da doar calculul.
           - Dacă elevul greșește: corectează blând, explică DE CE e greșit, dă exemplul corect.

    12. MATERIALE UPLOADATE (Cărți/PDF/Poze):
           - Dacă primești o poză sau un PDF, analizează TOT conținutul vizual înainte de a răspunde.
           - La poze cu probleme scrise de mână: transcrie problema, apoi rezolv-o.
           - Păstrează sensul original al textelor din manuale.

    13. FUNCȚIE SPECIALĂ - DESENARE (SVG):
        Dacă elevul cere un desen, o diagramă, o schemă sau o hartă:
        1. Ești OBLIGAT să generezi cod SVG valid.
        2. Codul trebuie încadrat STRICT între tag-uri:
           [[DESEN_SVG]]
           <svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
              <!-- Codul tău aici -->
           </svg>
           [[/DESEN_SVG]]
        3. IMPORTANT: Nu uita tag-ul de deschidere <svg> și cel de închidere </svg>!
        4. Adaugă întotdeauna etichete text (<text>) pentru a numi elementele din desen.
        5. Folosește culori clare și contraste bune pentru lizibilitate.
"""


# System prompt inițial — ține cont de modul pas cu pas dacă era deja setat
SYSTEM_PROMPT = get_system_prompt(
    pas_cu_pas=st.session_state.get("pas_cu_pas", False),
    desen_fizica=st.session_state.get("desen_fizica", True)
)


# === DETECȚIE AUTOMATĂ MATERIE ===
# Mapare cuvinte cheie → materie (pentru detecție rapidă fără apel API)
SUBJECT_KEYWORDS = {
    "matematică": [
        "ecuație", "ecuatia", "funcție", "functie", "derivată", "derivata", "integrală", "integrala",
        "limită", "limita", "matrice", "determinant", "trigonometrie", "geometrie", "algebră", "algebra",
        "logaritm", "radical", "inecuație", "inecuatia", "probabilitate", "combinatorică",
        "vector", "plan", "dreapta", "paralelă", "perpendiculară", "triunghi", "cerc", "parabola",
        "matematica", "mate", "math", "calcul", "număr", "numărul", "numere",
    ],
    "fizică": [
        "forță", "forta", "viteză", "viteza", "accelerație", "acceleratie", "masă", "masa",
        "energie", "putere", "curent", "tensiune", "rezistență", "rezistenta", "circuit",
        "câmp", "camp", "undă", "unda", "optică", "optica", "lentilă", "lentila",
        "termodinamică", "termodinamica", "gaz", "presiune", "volum", "temperatură", "temperatura",
        "fizica", "fizică", "mecanică", "mecanica", "electricitate", "baterie", "condensator",
        "gravitație", "gravitatie", "frecare", "pendul", "oscilatie", "oscilație",
    ],
    "chimie": [
        "atom", "moleculă", "molecula", "element", "compus", "reacție", "reactie",
        "acid", "baza", "sare", "oxidare", "reducere", "electroliză", "electroliza",
        "moli", "mol", "masă molară", "stoechiometrie", "ecuație chimică",
        "organic", "alcan", "alchenă", "alchena", "alcool", "ester", "chimica", "chimie",
        "ph", "soluție", "solutie", "concentratie", "concentrație",
    ],
    "biologie": [
        "celulă", "celula", "adn", "arn", "proteină", "proteina", "enzimă", "enzima",
        "mitoză", "mitoza", "meioză", "meioza", "genetică", "genetica", "cromozom",
        "fotosinteza", "fotosinteză", "respiratie", "respirație", "metabolism",
        "ecosistem", "specie", "organ", "tesut", "țesut", "sistem nervos",
        "biologie", "biologic", "planta", "plantă", "animal",
    ],
    "informatică": [
        "algoritm", "cod", "c++", "program", "functie", "funcție", "vector", "array",
        "recursivitate", "sortare", "căutare", "cautare", "stivă", "stiva", "coada", "coadă",
        "backtracking", "greedy", "sql", "baza de date", "informatica", "informatică",
        "pseudocod", "variabila", "variabilă", "ciclu", "for", "while", "if",
    ],
    "geografie": [
        "relief", "munte", "câmpie", "campie", "râu", "rau", "dunărea", "dunarea",
        "climă", "clima", "vegetatie", "vegetație", "populație", "populatie",
        "romania", "românia", "europa", "continent", "ocean", "geografie",
        "carpati", "carpații", "câmpia", "campia", "delta", "lac",
    ],
    "istorie": [
        "război", "razboi", "revoluție", "revolutie", "unire", "independenta", "independență",
        "cuza", "eminescu", "mihai viteazul", "stefan cel mare", "ștefan cel mare",
        "comunism", "comunist", "ceausescu", "ceaușescu", "bac 1918", "marea unire",
        "medieval", "evul mediu", "modern", "contemporan", "istorie", "istoric",
        "domnie", "domitor", "rege", "regat", "principat",
    ],
    "limba și literatura română": [
        "roman", "roman", "poezie", "poem", "eminescu", "rebreanu", "sadoveanu",
        "preda", "arghezi", "blaga", "bacovia", "caragiale", "creanga", "creangă",
        "eseu", "comentariu", "caracterizare", "narator", "personaj", "tema",
        "figuri de stil", "metafora", "metaforă", "epitet", "comparatie", "comparație",
        "roman", "proza", "proză", "dramaturgie", "gramatica", "gramatică",
        "romana", "română", "literatura", "literatură",
    ],
    "limba engleză": [
        "english", "engleză", "engleza", "tense", "grammar", "essay", "verb",
        "present", "past", "future", "conditional", "passive", "vocabulary",
    ],
    "limba franceză": [
        "français", "franceză", "franceza", "passé", "imparfait", "subjonctif",
        "verbe", "grammaire", "être", "avoir",
    ],
}


def detect_subject_from_text(text: str) -> str | None:
    """Detectează materia dintr-un text folosind cuvinte cheie. Rapid, fără API."""
    text_lower = text.lower()
    scores = {}
    for subject, keywords in SUBJECT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[subject] = score
    if not scores:
        return None
    return max(scores, key=scores.get)


def get_detected_subject() -> str | None:
    """Returnează materia detectată din session_state sau None."""
    return st.session_state.get("_detected_subject", None)


def update_system_prompt_for_subject(materie: str | None):
    """Actualizează system prompt-ul pentru materia dată și salvează în session_state."""
    st.session_state["_detected_subject"] = materie
    st.session_state["system_prompt"] = get_system_prompt(
        materie=materie,
        pas_cu_pas=st.session_state.get("pas_cu_pas", False),
        desen_fizica=st.session_state.get("desen_fizica", True),
        mod_strategie=st.session_state.get("mod_strategie", False),
        mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
    )




safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]



# ============================================================
# === SIMULARE BAC ===
# ============================================================

MATERII_BAC = {
    "📐 Matematică": {
        "cod": "matematica",
        "profile": ["M1 - Mate-Info", "M2 - Științe ale naturii"],
        "subiecte": ["Algebră", "Analiză matematică", "Geometrie"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "📖 Română": {
        "cod": "romana",
        "profile": ["Toate profilurile"],
        "subiecte": ["Text literar", "Text nonliterar", "Redactare eseu"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "⚡ Fizică": {
        "cod": "fizica",
        "profile": ["Mate-Info", "Științe ale naturii"],
        "subiecte": ["Mecanică", "Termodinamică", "Electricitate", "Optică"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🧪 Chimie": {
        "cod": "chimie",
        "profile": ["Chimie anorganică", "Chimie organică"],
        "subiecte": ["Chimie anorganică", "Chimie organică"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🧬 Biologie": {
        "cod": "biologie",
        "profile": ["Biologie vegetală și animală", "Anatomie și fiziologie umană"],
        "subiecte": ["Anatomie", "Genetică", "Ecologie"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🏛️ Istorie": {
        "cod": "istorie",
        "profile": ["Umanist", "Pedagogic", "Teologic"],
        "subiecte": ["Istorie românească", "Istorie universală"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "🌍 Geografie": {
        "cod": "geografie",
        "profile": ["Profiluri umaniste"],
        "subiecte": ["Geografia României", "Geografia Europei", "Geografia lumii"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
    "💻 Informatică": {
        "cod": "informatica",
        "profile": ["Mate-Info intensiv C++", "Mate-Info intensiv Pascal"],
        "subiecte": ["Algoritmi", "Structuri de date", "Programare"],
        "timp_minute": 180,
        "punctaj_total": 100,
    },
}




def extract_text_from_photo(image_bytes: bytes, materie_label: str) -> str:
    """Extrage textul scris de mână dintr-o fotografie folosind Gemini Vision.
    
    Folosește Google Files API (upload real) în loc de base64 inline —
    același mecanism ca în sidebar, pentru analiză vizuală completă.
    """
    import os
    tmp_path = None
    try:
        key = keys[st.session_state.get("key_index", 0)]
        gemini_client = genai.Client(api_key=key)

        # Uploadăm imaginea pe Google Files API
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name

        gfile = gemini_client.files.upload(file=tmp_path, config=genai_types.UploadFileConfig(mime_type="image/jpeg"))
        poll = 0
        while str(gfile.state) in ("FileState.PROCESSING", "PROCESSING") and poll < 30:
            time.sleep(1)
            gfile = gemini_client.files.get(gfile.name)
            poll += 1

        if str(gfile.state) not in ("FileState.ACTIVE", "ACTIVE"):
            return "[Eroare: imaginea nu a putut fi procesată de Google]"

        prompt = (
            f"Ești un asistent care transcrie text scris de mână din lucrări de elevi la {materie_label}. "
            f"Transcrie EXACT tot ce este scris în imagine, inclusiv formule, simboluri matematice și calcule. "
            f"Păstrează structura (Subiectul I, II, III dacă există). "
            f"Dacă un cuvânt e greu de citit, transcrie-l cu [?]. "
            f"Nu adăuga nimic, nu corecta nimic — transcrie fidel."
        )
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[gfile, prompt]
        )

        # Curăță fișierul de pe Google după utilizare
        try:
            gemini_client.files.delete(gfile.name)
        except Exception:
            pass

        return response.text.strip()

    except Exception as e:
        return f"[Eroare la citirea pozei: {e}]"
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_bac_prompt_ai(materie_label, materie_info, profil):
    subiecte_str = ", ".join(materie_info["subiecte"])
    return (
        f"Generează un subiect complet de BAC la {materie_label} ({profil}), "
        f"identic ca structură și dificultate cu subiectele oficiale din România.\n\n"
        f"STRUCTURĂ OBLIGATORIE:\n"
        f"- SUBIECTUL I (30 puncte): 5 itemi tip grilă/răspuns scurt\n"
        f"- SUBIECTUL II (30 puncte): 3-4 probleme de dificultate medie\n"
        f"- SUBIECTUL III (30 puncte): 1-2 probleme complexe / eseu structurat\n"
        f"- 10 puncte din oficiu\n\n"
        f"TEME: {subiecte_str}\n"
        f"TIMP: {materie_info['timp_minute']} minute\n\n"
        f"La final adaugă baremul astfel:\n"
        f"[[BAREM_BAC]]\n"
        f"SUBIECTUL I: [raspunsuri si punctaj]\n"
        f"SUBIECTUL II: [solutii si punctaj]\n"
        f"SUBIECTUL III: [criterii si punctaj]\n"
        f"[[/BAREM_BAC]]"
    )


def get_bac_correction_prompt(materie_label, subiect, raspuns_elev, from_photo=False):
    source_note = (
        "NOTĂ: Răspunsul a fost extras automat dintr-o fotografie a lucrării. "
        "Unele cuvinte pot fi transcrise imperfect din cauza scrisului de mână — "
        "judecă după intenția elevului, nu după eventuale erori de OCR.\n\n"
        if from_photo else ""
    )

    # Reguli de limbaj adaptate materiei
    if "Română" in materie_label:
        lang_rules = (
            "CORECTARE LIMBĂ ROMÂNĂ (OBLIGATORIU — punctaj separat):\n"
            "- Ortografie și punctuație (virgule, punct, ghilimele «»)\n"
            "- Acordul gramatical (subiect-predicat, adjectiv-substantiv)\n"
            "- Folosirea corectă a cratimei, apostrofului\n"
            "- Exprimare clară, coerentă, fără pleonasme sau cacofonii\n"
            "- Registru stilistic adecvat eseului de BAC\n"
            "- Acordă până la 10 puncte bonus/penalizare pentru calitatea limbii\n\n"
        )
    else:
        lang_rules = (
            f"CORECTARE LIMBAJ ȘTIINȚIFIC ({materie_label}):\n"
            "- Terminologie specifică folosită corect\n"
            "- Notații și simboluri respectate (ex: m pentru masă, nu M; v nu V pentru viteză)\n"
            "- Unități de măsură scrise corect și complet\n"
            "- Formulele scrise corect, fără ambiguități\n"
            "- Raționament logic și coerent exprimat în cuvinte\n"
            "- Acordă până la 5 puncte bonus/penalizare pentru calitatea exprimării\n\n"
        )

    return (
        f"Ești examinator BAC România pentru {materie_label}.\n\n"
        f"{source_note}"
        f"SUBIECTUL:\n{subiect}\n\n"
        f"RĂSPUNSUL ELEVULUI:\n{raspuns_elev}\n\n"
        f"Corectează COMPLET în această ordine:\n\n"
        f"## 📊 Punctaj per subiect\n"
        f"- Subiectul I: X/30 puncte\n"
        f"- Subiectul II: X/30 puncte\n"
        f"- Subiectul III: X/30 puncte\n"
        f"- Din oficiu: 10 puncte\n\n"
        f"## ✅ Ce a făcut bine\n"
        f"[aspecte corecte]\n\n"
        f"## ❌ Greșeli și explicații\n"
        f"[fiecare greșeală explicată]\n\n"
        f"## 🖊️ Calitatea limbii și exprimării\n"
        f"{lang_rules}"
        f"## 🎓 Nota finală\n"
        f"**Nota: X/10** — [verdict scurt]\n\n"
        f"## 💡 Recomandări pentru BAC\n"
        f"[2-3 sfaturi concrete]\n\n"
        f"Fii constructiv, cald, dar riguros ca un examinator real."
    )


def parse_bac_subject(response):
    barem = ""
    subject_text = response
    match = re.search(r"\[\[BAREM_BAC\]\](.*?)\[\[/BAREM_BAC\]\]", response, re.DOTALL)
    if match:
        barem = match.group(1).strip()
        subject_text = response[:match.start()].strip()
    return subject_text, barem


def format_timer(seconds_remaining):
    h = seconds_remaining // 3600
    m = (seconds_remaining % 3600) // 60
    s = seconds_remaining % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def run_bac_sim_ui():
    st.subheader("🎓 Simulare BAC")

    # ── ECRAN DE START ──
    if not st.session_state.get("bac_active"):
        st.markdown(
            "<div style='background:linear-gradient(135deg,#667eea,#764ba2);"
            "color:white;padding:20px 24px;border-radius:12px;margin-bottom:20px'>"
            "<h4 style='margin:0 0 8px 0'>📋 Cum funcționează?</h4>"
            "<ul style='margin:0;padding-left:18px;line-height:1.8'>"
            "<li>Alegi materia, profilul și tipul de subiect</li>"
            "<li>Rezolvi în timp real cu cronometru opțional</li>"
            "<li>Primești corectare AI detaliată + barem</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)
        with col1:
            bac_materie = st.selectbox("📚 Materia:", options=list(MATERII_BAC.keys()), key="bac_mat_sel")
            info = MATERII_BAC[bac_materie]
            bac_profil = st.selectbox("🎯 Profil:", options=info["profile"], key="bac_prof_sel")
        with col2:
            bac_tip = "🤖 Generat de AI"
            use_timer = st.checkbox(f"⏱️ Cronometru ({info['timp_minute']} min)", value=True, key="bac_timer")


        st.divider()
        col_s, col_b = st.columns(2)
        with col_s:
            btn_lbl = "🚀 Generează subiect AI"
            if st.button(btn_lbl, type="primary", use_container_width=True):
                if "AI" in bac_tip:
                    with st.spinner("📝 Se generează subiectul BAC..."):
                        prompt = get_bac_prompt_ai(bac_materie, info, bac_profil)
                        full = "".join(run_chat_with_rotation(
                            [], [prompt],
                            system_prompt=get_system_prompt(MATERII.get(bac_materie))
                        ))
                    subject_text, barem = parse_bac_subject(full)


                st.session_state.update({
                    "bac_active": True,
                    "bac_materie": bac_materie,
                    "bac_profil": bac_profil,
                    "bac_tip": bac_tip,
                    "bac_subject": subject_text,
                    "bac_barem": barem,
                    "bac_raspuns": "",
                    "bac_corectat": False,
                    "bac_corectare": "",
                    "bac_start_time": time.time() if use_timer else None,
                    "bac_timp_min": info["timp_minute"],
                    "bac_use_timer": use_timer,
                })
                st.rerun()
        with col_b:
            if st.button("↩️ Înapoi la chat", use_container_width=True):
                st.session_state.pop("bac_mode", None)
                st.rerun()
        return

    # ── SIMULARE ACTIVĂ ──
    col_title, col_timer = st.columns([3, 1])
    with col_title:
        st.markdown(f"### {st.session_state.bac_materie} · {st.session_state.bac_profil}")
    with col_timer:
        if st.session_state.get("bac_use_timer") and st.session_state.get("bac_start_time"):
            elapsed = int(time.time() - st.session_state.bac_start_time)
            total   = st.session_state.bac_timp_min * 60
            left    = max(0, total - elapsed)
            pct     = left / total
            color   = "#2ecc71" if pct > 0.5 else ("#e67e22" if pct > 0.2 else "#e74c3c")
            st.markdown(
                f'<div style="background:{color};color:white;padding:8px 12px;'
                f'border-radius:8px;text-align:center;font-size:20px;font-weight:700">'
                f'⏱️ {format_timer(left)}</div>',
                unsafe_allow_html=True
            )
            if left == 0:
                st.warning("⏰ Timpul a expirat!")

    st.divider()

    with st.expander("📋 Subiectul", expanded=not st.session_state.bac_corectat):
        st.markdown(st.session_state.bac_subject)

    if not st.session_state.bac_corectat:
        st.markdown("### ✏️ Răspunsurile tale")

        tab_foto, tab_text = st.tabs(["📷 Fotografiază lucrarea", "⌨️ Scrie manual"])

        raspuns = st.session_state.get("bac_raspuns", "")
        from_photo = False

        # ── TAB FOTO ──
        with tab_foto:
            st.info(
                "📱 **Pe telefon:** apasă butonul de mai jos și fotografiază lucrarea.\n\n"
                "💻 **Pe calculator:** încarcă o poză din galerie.\n\n"
                "AI-ul va citi textul și va porni corectarea automat."
            )
            uploaded_photo = st.file_uploader(
                "Încarcă fotografia lucrării:",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="bac_photo_upload",
                help="Fă o poză clară, cu lumină bună, la lucrarea scrisă de mână."
            )

            if uploaded_photo:
                st.image(uploaded_photo, caption="Fotografia încărcată", use_container_width=True)

                if not st.session_state.get("bac_ocr_done"):
                    with st.spinner("🔍 Profesorul citește lucrarea..."):
                        img_bytes = uploaded_photo.read()
                        text_extras = extract_text_from_photo(img_bytes, st.session_state.bac_materie)
                    st.session_state.bac_raspuns  = text_extras
                    st.session_state.bac_ocr_done = True
                    st.session_state.bac_from_photo = True

                    # Pornește corectura automat
                    with st.spinner("📊 Se corectează lucrarea..."):
                        prompt = get_bac_correction_prompt(
                            st.session_state.bac_materie,
                            st.session_state.bac_subject,
                            text_extras,
                            from_photo=True
                        )
                        corectare = "".join(run_chat_with_rotation(
                            [], [prompt],
                            system_prompt=get_system_prompt(MATERII.get(st.session_state.bac_materie))
                        ))
                    st.session_state.bac_corectare = corectare
                    st.session_state.bac_corectat  = True
                    st.rerun()

                if st.session_state.get("bac_ocr_done"):
                    with st.expander("📄 Text extras din poză", expanded=False):
                        st.text(st.session_state.get("bac_raspuns", ""))

        # ── TAB TEXT ──
        with tab_text:
            raspuns = st.text_area(
                "Scrie rezolvarea completă:",
                value=st.session_state.get("bac_raspuns", ""),
                height=350,
                placeholder="Subiectul I:\n1. ...\n2. ...\n\nSubiectul II:\n...\n\nSubiectul III:\n...",
                key="bac_ans_input"
            )
            st.session_state.bac_raspuns = raspuns
            st.session_state.bac_from_photo = False

            if st.button("🤖 Corectare AI", type="primary", use_container_width=True,
                         disabled=not raspuns.strip()):
                with st.spinner("📊 Se corectează lucrarea..."):
                    prompt = get_bac_correction_prompt(
                        st.session_state.bac_materie,
                        st.session_state.bac_subject,
                        raspuns,
                        from_photo=False
                    )
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(MATERII.get(st.session_state.bac_materie))
                    ))
                st.session_state.bac_corectare = corectare
                st.session_state.bac_corectat  = True
                st.rerun()

        st.divider()
        col_barem, col_nou = st.columns(2)
        with col_barem:
            if st.session_state.get("bac_barem"):
                if st.button("📋 Arată Baremul", use_container_width=True):
                    st.session_state.bac_show_barem = not st.session_state.get("bac_show_barem", False)
                    st.rerun()
        with col_nou:
            if st.button("🔄 Subiect nou", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()

        if st.session_state.get("bac_show_barem") and st.session_state.get("bac_barem"):
            with st.expander("📋 Barem de corectare", expanded=True):
                st.markdown(st.session_state.bac_barem)

    else:
        st.markdown("### 📊 Corectare AI")
        st.markdown(st.session_state.bac_corectare)
        if st.session_state.get("bac_barem"):
            with st.expander("📋 Barem"):
                st.markdown(st.session_state.bac_barem)
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Subiect nou", type="primary", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("✏️ Reîncerc același subiect", use_container_width=True):
                st.session_state.bac_corectat  = False
                st.session_state.bac_corectare = ""
                st.session_state.bac_raspuns   = ""
                if st.session_state.get("bac_use_timer"):
                    st.session_state.bac_start_time = time.time()
                st.rerun()
        with col3:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("bac_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("bac_mode", None)
                st.rerun()


# ============================================================
# === CORECTARE TEME ===
# ============================================================

def get_homework_correction_prompt(materie_label: str, text_tema: str, from_photo: bool = False) -> str:
    source_note = (
        "NOTĂ: Tema a fost extrasă dintr-o fotografie. "
        "Unele cuvinte pot fi transcrise imperfect — judecă după intenția elevului.\n\n"
        if from_photo else ""
    )

    if "Română" in materie_label:
        corectare_limba = (
            "## 🖊️ Corectare limbă și stil\n"
            "Acordă atenție specială:\n"
            "- **Ortografie**: diacritice (ă,â,î,ș,ț), cratimă, apostrof\n"
            "- **Punctuație**: virgulă, punct, linie de dialog, ghilimele «»\n"
            "- **Acord gramatical**: subiect-predicat, adjectiv-substantiv, pronume\n"
            "- **Exprimare**: cacofonii, pleonasme, tautologii, registru stilistic\n"
            "- **Coerență**: logica textului, legătura dintre idei\n"
            "Subliniază greșelile găsite și explică regula corectă.\n\n"
        )
    else:
        corectare_limba = (
            f"## 🖊️ Limbaj și exprimare ({materie_label})\n"
            "- Terminologie specifică folosită corect\n"
            "- Notații, simboluri și unități de măsură corecte\n"
            "- Raționament exprimat clar și logic\n\n"
        )

    return (
        f"Ești profesor de {materie_label} și corectezi tema unui elev de liceu.\n\n"
        f"{source_note}"
        f"TEMA ELEVULUI:\n{text_tema}\n\n"
        f"Corectează complet și constructiv:\n\n"
        f"## ✅ Ce a făcut bine\n"
        f"[aspecte corecte — fii specific, nu generic]\n\n"
        f"## ❌ Greșeli de conținut\n"
        f"[fiecare greșeală de materie explicată, cu varianta corectă]\n\n"
        f"{corectare_limba}"
        f"## 📊 Notă orientativă\n"
        f"**Nota: X/10** — [justificare scurtă]\n\n"
        f"## 💡 Sfaturi pentru data viitoare\n"
        f"[2-3 recomandări concrete și aplicabile]\n\n"
        f"Ton: cald, constructiv, ca un profesor care vrea să ajute, nu să descurajeze."
    )


def run_homework_ui():
    st.subheader("📚 Corectare Temă")

    if not st.session_state.get("hw_done"):
        col1, col2 = st.columns([2, 1])
        with col1:
            hw_materie = st.selectbox(
                "📚 Materia temei:",
                options=[m for m in MATERII.keys() if m != "🎓 Toate materiile"],
                key="hw_materie_sel"
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.caption("Profesorul se adaptează materiei.")

        st.divider()

        tab_foto, tab_text = st.tabs(["📷 Fotografiază tema", "⌨️ Scrie / lipește textul"])

        with tab_foto:
            st.info(
                "📱 **Pe telefon:** fotografiază caietul sau foaia de temă.\n\n"
                "💻 **Pe calculator:** încarcă o poză din galerie.\n\n"
                "Profesorul va citi și corecta automat."
            )
            hw_photo = st.file_uploader(
                "Încarcă fotografia temei:",
                type=["jpg", "jpeg", "png", "webp", "heic"],
                key="hw_photo_upload",
                help="Asigură-te că poza e clară și bine luminată."
            )

            if hw_photo and not st.session_state.get("hw_ocr_done"):
                st.image(hw_photo, caption="Fotografia încărcată", use_container_width=True)
                with st.spinner("🔍 Profesorul citește tema..."):
                    text_extras = extract_text_from_photo(hw_photo.read(), hw_materie)
                st.session_state.hw_text       = text_extras
                st.session_state.hw_ocr_done   = True
                st.session_state.hw_from_photo = True
                st.session_state.hw_materie    = hw_materie
                with st.spinner("📝 Se corectează tema..."):
                    prompt = get_homework_correction_prompt(hw_materie, text_extras, from_photo=True)
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(MATERII.get(hw_materie))
                    ))
                st.session_state.hw_corectare = corectare
                st.session_state.hw_done      = True
                st.rerun()
            elif hw_photo and st.session_state.get("hw_ocr_done"):
                with st.expander("📄 Text extras din poză", expanded=False):
                    st.text(st.session_state.get("hw_text", ""))

        with tab_text:
            hw_text = st.text_area(
                "Lipește sau scrie textul temei:",
                value=st.session_state.get("hw_text", ""),
                height=300,
                placeholder="Scrie sau lipește tema aici...",
                key="hw_text_input"
            )
            st.session_state.hw_text = hw_text
            if st.button("📝 Corectează tema", type="primary",
                         use_container_width=True, disabled=not hw_text.strip()):
                st.session_state.hw_materie    = hw_materie
                st.session_state.hw_from_photo = False
                with st.spinner("📝 Se corectează tema..."):
                    prompt = get_homework_correction_prompt(hw_materie, hw_text, from_photo=False)
                    corectare = "".join(run_chat_with_rotation(
                        [], [prompt],
                        system_prompt=get_system_prompt(MATERII.get(hw_materie))
                    ))
                st.session_state.hw_corectare = corectare
                st.session_state.hw_done      = True
                st.rerun()

    else:
        mat = st.session_state.get("hw_materie", "")
        src = "📷 din fotografie" if st.session_state.get("hw_from_photo") else "✏️ scrisă manual"
        st.caption(f"{mat} · temă {src}")
        if st.session_state.get("hw_from_photo") and st.session_state.get("hw_text"):
            with st.expander("📄 Text extras din poză", expanded=False):
                st.text(st.session_state.hw_text)
        st.markdown(st.session_state.hw_corectare)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📚 Corectează altă temă", type="primary", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("hw_")]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in [k for k in list(st.session_state.keys()) if k.startswith("hw_")]:
                    st.session_state.pop(k, None)
                st.session_state.pop("homework_mode", None)
                st.rerun()


# === MOD QUIZ ===
NIVELE_QUIZ = ["🟢 Ușor (gimnaziu)", "🟡 Mediu (liceu)", "🔴 Greu (BAC)"]

MATERII_QUIZ = [m for m in list(MATERII.keys()) if m != "🎓 Toate materiile"]


def get_quiz_prompt(materie_label: str, nivel: str, materie_val: str) -> str:
    """Generează prompt pentru crearea unui quiz."""
    nivel_text = nivel.split(" ", 1)[1].strip("()")
    return f"""Generează un quiz de 5 întrebări la {materie_label} pentru nivel {nivel_text}.

REGULI STRICTE:
1. Generează EXACT 5 întrebări numerotate (1. 2. 3. 4. 5.)
2. Fiecare întrebare are 4 variante de răspuns: A) B) C) D)
3. La finalul TUTUROR întrebărilor adaugă un bloc special cu răspunsurile corecte:

[[RASPUNSURI_CORECTE]]
1: X
2: X
3: X
4: X
5: X
[[/RASPUNSURI_CORECTE]]

unde X este A, B, C sau D.
4. Întrebările trebuie să fie clare și potrivite pentru nivel {nivel_text}.
5. Folosește LaTeX ($...$) pentru formule matematice.
6. NU da explicații acum — doar întrebările și răspunsurile corecte la final."""


def parse_quiz_response(response: str) -> tuple[str, dict]:
    """Extrage intrebarile si raspunsurile corecte din raspunsul AI.

    FIX: Gestioneaza corect cazurile cand AI-ul nu respecta exact delimitatorii:
    - Delimitatori lipsa: fallback prin cautarea unui bloc de raspunsuri
    - Formate variate: '1: A', '1. A', '1) A', '**1**: A'
    - Raspunsuri cu text extra: '1: A) text' -> extrage doar litera
    """
    correct = {}
    clean_response = response

    # Incearca mai intai delimitatorii exacti
    match = re.search(r'\[\[RASPUNSURI_CORECTE\]\](.*?)\[\[/RASPUNSURI_CORECTE\]\]',
                      response, re.DOTALL)

    # FIX: Fallback — AI-ul uneori omite delimitatorii sau ii scrie diferit
    if not match:
        match = re.search(
            r'(?:raspunsuri\s*corecte|raspunsuri\s*corecte|answers?)[:\s]*\n'
            r'((?:\s*\d+\s*[:.)-]\s*[A-D].*\n?){3,})',
            response, re.IGNORECASE | re.DOTALL
        )

    if match:
        block_start = match.start()
        clean_response = response[:block_start].strip()
        raw_block = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)

        for line in raw_block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # FIX: accepta formate: '1: A', '1. A', '1) A', '**1**: A', '1: A) text...'
            m = re.match(r'\*{0,2}(\d+)\*{0,2}\s*[:.)-]\s*\*{0,2}([A-D])\*{0,2}', line, re.IGNORECASE)
            if m:
                try:
                    q_num = int(m.group(1))
                    ans = m.group(2).upper()
                    if 1 <= q_num <= 10:
                        correct[q_num] = ans
                except ValueError:
                    pass

    # FIX: Daca tot nu avem raspunsuri, incearca extractie din textul intreg
    if not correct:
        for m in re.finditer(
            r'(?:intrebarea|intrebarea|question)?\s*(\d+).*?'
            r'r[a]spuns(?:ul)?\s*(?:corect)?\s*[:\s]+([A-D])\b',
            response, re.IGNORECASE
        ):
            try:
                q_num = int(m.group(1))
                ans = m.group(2).upper()
                if 1 <= q_num <= 10:
                    correct[q_num] = ans
            except ValueError:
                pass

    return clean_response, correct


def evaluate_quiz(user_answers: dict, correct_answers: dict) -> tuple[int, str]:
    """Evaluează răspunsurile și returnează (scor, feedback_text)."""
    score = sum(1 for q, a in user_answers.items() if correct_answers.get(q) == a)
    total = len(correct_answers)

    lines = []
    for q in sorted(correct_answers.keys()):
        user_ans = user_answers.get(q, "—")
        correct_ans = correct_answers[q]
        if user_ans == correct_ans:
            lines.append(f"✅ **Întrebarea {q}**: {user_ans} — Corect!")
        else:
            lines.append(f"❌ **Întrebarea {q}**: ai răspuns **{user_ans}**, corect era **{correct_ans}**")

    if score == total:
        verdict = "🏆 Excelent! Nota 10!"
    elif score >= total * 0.8:
        verdict = "🌟 Foarte bine!"
    elif score >= total * 0.6:
        verdict = "👍 Bine, mai exersează puțin!"
    elif score >= total * 0.4:
        verdict = "📚 Trebuie să mai studiezi."
    else:
        verdict = "💪 Nu-ți face griji, încearcă din nou!"

    feedback = f"### Rezultat: {score}/{total} — {verdict}\n\n" + "\n\n".join(lines)
    return score, feedback


def run_quiz_ui():
    """Randează UI-ul pentru modul Quiz."""
    st.subheader("📝 Mod Examinare")

    # --- Setup quiz ---
    if not st.session_state.get("quiz_active"):
        col1, col2 = st.columns(2)
        with col1:
            quiz_materie_label = st.selectbox(
                "Materie:",
                options=MATERII_QUIZ,
                key="quiz_materie_select"
            )
        with col2:
            quiz_nivel = st.selectbox(
                "Nivel:",
                options=NIVELE_QUIZ,
                key="quiz_nivel_select"
            )

        if st.button("🚀 Generează Quiz", type="primary", use_container_width=True):
            quiz_materie_val = MATERII[quiz_materie_label]
            with st.spinner("📝 Profesorul pregătește întrebările..."):
                prompt = get_quiz_prompt(quiz_materie_label, quiz_nivel, quiz_materie_val)
                full_resp = ""
                for chunk in run_chat_with_rotation(
                    [], [prompt],
                    system_prompt=get_system_prompt(quiz_materie_val)
                ):
                    full_resp += chunk

            questions_text, correct = parse_quiz_response(full_resp)
            if len(correct) >= 3:
                st.session_state.quiz_active = True
                st.session_state.quiz_questions = questions_text
                st.session_state.quiz_correct = correct
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_materie = quiz_materie_label
                st.session_state.quiz_nivel = quiz_nivel
                st.rerun()
            else:
                st.error("❌ Nu am putut genera quiz-ul. Încearcă din nou.")
        return

    # --- Quiz activ ---
    st.caption(f"📚 {st.session_state.quiz_materie} · {st.session_state.quiz_nivel}")

    # Afișează întrebările
    st.markdown(st.session_state.quiz_questions)
    st.divider()

    if not st.session_state.quiz_submitted:
        st.markdown("**Alege răspunsurile tale:**")
        answers = {}
        for q_num in sorted(st.session_state.quiz_correct.keys()):
            answers[q_num] = st.radio(
                f"Întrebarea {q_num}:",
                options=["A", "B", "C", "D"],
                horizontal=True,
                key=f"quiz_ans_{q_num}",
                index=None
            )

        all_answered = all(v is not None for v in answers.values())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Trimite răspunsurile", type="primary",
                         disabled=not all_answered, use_container_width=True):
                st.session_state.quiz_answers = {k: v for k, v in answers.items() if v}
                st.session_state.quiz_submitted = True
                st.rerun()
        with col2:
            if st.button("🔄 Quiz nou", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
    else:
        # Afișează rezultatele
        score, feedback = evaluate_quiz(
            st.session_state.quiz_answers,
            st.session_state.quiz_correct
        )
        st.markdown(feedback)
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Quiz nou", type="primary", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted"]:
                    st.session_state.pop(k, None)
                st.rerun()
        with col2:
            if st.button("💬 Înapoi la chat", use_container_width=True):
                for k in ["quiz_active", "quiz_questions", "quiz_correct",
                          "quiz_answers", "quiz_submitted", "quiz_mode"]:
                    st.session_state.pop(k, None)
                st.rerun()


def run_chat_with_rotation(history_obj, payload, system_prompt=None):
    """Rulează chat cu rotație automată a cheilor API și fallback modele."""
    MODEL_FALLBACKS = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-1.5-flash",
    ]

    active_prompt = system_prompt or st.session_state.get("system_prompt") or SYSTEM_PROMPT
    max_retries = max(len(keys) * 3, 6)
    last_error = None

    for attempt in range(max_retries):
        if st.session_state.key_index >= len(keys):
            st.session_state.key_index = 0
        current_key = keys[st.session_state.key_index]

        model_name = MODEL_FALLBACKS[min(attempt // max(len(keys), 1), len(MODEL_FALLBACKS) - 1)]

        try:
            gemini_client = genai.Client(api_key=current_key)

            gen_config = genai_types.GenerateContentConfig(
                system_instruction=active_prompt,
                safety_settings=[
                    genai_types.SafetySetting(category=s["category"], threshold=s["threshold"])
                    for s in safety_settings
                ],
            )

            history_new = []
            for msg in history_obj:
                history_new.append(
                    genai_types.Content(
                        role=msg["role"],
                        parts=[genai_types.Part(text=p) if isinstance(p, str) else genai_types.Part(file_data=genai_types.FileData(file_uri=p.uri, mime_type=p.mime_type)) for p in (msg["parts"] if isinstance(msg["parts"], list) else [msg["parts"]])]
                    )
                )

            current_parts = []
            for p in (payload if isinstance(payload, list) else [payload]):
                if isinstance(p, str):
                    current_parts.append(genai_types.Part(text=p))
                elif hasattr(p, "uri"):
                    current_parts.append(genai_types.Part(file_data=genai_types.FileData(file_uri=p.uri, mime_type=p.mime_type)))
                else:
                    current_parts.append(genai_types.Part(text=str(p)))

            all_contents = history_new + [genai_types.Content(role="user", parts=current_parts)]

            response_stream = gemini_client.models.generate_content_stream(
                model=model_name,
                contents=all_contents,
                config=gen_config,
            )

            chunks = []
            for chunk in response_stream:
                try:
                    if chunk.text:
                        chunks.append(chunk.text)
                except Exception:
                    continue

            if model_name != MODEL_FALLBACKS[0]:
                st.toast(f"ℹ️ Răspuns generat cu modelul de rezervă ({model_name})", icon="🔄")

            for text in chunks:
                yield text
            return

        except Exception as e:
            last_error = e
            error_msg = str(e)

            if "400" in error_msg:
                raise Exception(f"❌ Cerere invalidă (400): {error_msg}") from e

            if "503" in error_msg or "overloaded" in error_msg.lower() or "resource_exhausted" in error_msg.lower():
                wait = min(0.5 * (2 ** attempt), 5)
                st.toast("🐢 Server ocupat, reîncerc...", icon="⏳")
                time.sleep(wait)
                continue

            elif "429" in error_msg or "quota" in error_msg.lower() or "rate_limit" in error_msg.lower() or "API key not valid" in error_msg:
                st.toast(f"⚠️ Schimb cheia {st.session_state.key_index + 1}...", icon="🔄")
                st.session_state.key_index = (st.session_state.key_index + 1) % len(keys)
                time.sleep(0.5)
                continue

            else:
                raise e

    raise Exception(f"❌ Serviciul este indisponibil după {max_retries} încercări. {last_error or ''}")


# === UI PRINCIPAL ===
st.title("🎓 Profesor Liceu")

with st.sidebar:
    st.header("⚙️ Opțiuni")

    # --- Selector materie ---
    st.subheader("📚 Materie")
    materie_label = st.selectbox(
        "Alege materia:",
        options=list(MATERII.keys()),
        index=0,
        label_visibility="collapsed"
    )
    materie_selectata = MATERII[materie_label]

    # Actualizează system prompt dacă s-a schimbat materia
    if st.session_state.get("materie_selectata") != materie_selectata:
        st.session_state.materie_selectata = materie_selectata
        # Resetăm detecția automată — selectorul are prioritate
        st.session_state["_detected_subject"] = materie_selectata
        st.session_state.system_prompt = get_system_prompt(
            materie_selectata,
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            desen_fizica=st.session_state.get("desen_fizica", True),
            mod_strategie=st.session_state.get("mod_strategie", False),
            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False),
        )

    if materie_selectata:
        st.info(f"Focusat pe: **{materie_label}**")

    st.divider()

    # --- Dark Mode toggle ---
    dark_mode = st.toggle("🌙 Mod Întunecat", value=st.session_state.get("dark_mode", False))
    if dark_mode != st.session_state.get("dark_mode", False):
        st.session_state.dark_mode = dark_mode
        st.rerun()

    # --- Mod Pas cu Pas ---
    pas_cu_pas = st.toggle(
        "🔢 Explicație Pas cu Pas",
        value=st.session_state.get("pas_cu_pas", False),
        help="Profesorul va explica fiecare problemă detaliat, pas cu pas, cu motivația fiecărei operații."
    )
    if pas_cu_pas != st.session_state.get("pas_cu_pas", False):
        st.session_state.pas_cu_pas = pas_cu_pas
        # Regenerează prompt-ul cu noul mod
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            pas_cu_pas=pas_cu_pas,
            desen_fizica=st.session_state.get("desen_fizica", True)
        )
        if pas_cu_pas:
            st.toast("🔢 Mod Pas cu Pas activat!", icon="✅")
        else:
            st.toast("Mod normal activat.", icon="💬")
        st.rerun()

    if st.session_state.get("pas_cu_pas"):
        st.info("🔢 **Pas cu Pas activ** — fiecare problemă e explicată detaliat.", icon="📋")

    # --- Mod Explică-mi Strategia ---
    mod_strategie = st.toggle(
        "🧠 Explică-mi Strategia",
        value=st.session_state.get("mod_strategie", False),
        help="Profesorul explică CUM să gândești rezolvarea — logica și strategia, nu calculele."
    )
    if mod_strategie != st.session_state.get("mod_strategie", False):
        st.session_state.mod_strategie = mod_strategie
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            desen_fizica=st.session_state.get("desen_fizica", True),
            mod_strategie=mod_strategie,
            mod_bac_intensiv=st.session_state.get("mod_bac_intensiv", False)
        )
        st.toast("🧠 Mod Strategie activat!" if mod_strategie else "Mod normal activat.", icon="✅" if mod_strategie else "💬")
        st.rerun()
    if st.session_state.get("mod_strategie"):
        st.info("🧠 **Strategie activ** — înveți să gândești, nu să copiezi.", icon="🗺️")

    # --- Mod Pregătire BAC Intensivă ---
    mod_bac_intensiv = st.toggle(
        "🎓 Pregătire BAC Intensivă",
        value=st.session_state.get("mod_bac_intensiv", False),
        help="Focusat pe ce pică la BAC: tipare de subiecte, punctaj, timp, teorie lipsă detectată automat."
    )
    if mod_bac_intensiv != st.session_state.get("mod_bac_intensiv", False):
        st.session_state.mod_bac_intensiv = mod_bac_intensiv
        st.session_state.system_prompt = get_system_prompt(
            st.session_state.get("materie_selectata"),
            pas_cu_pas=st.session_state.get("pas_cu_pas", False),
            desen_fizica=st.session_state.get("desen_fizica", True),
            mod_strategie=st.session_state.get("mod_strategie", False),
            mod_bac_intensiv=mod_bac_intensiv
        )
        st.toast("🎓 Mod BAC Intensiv activat!" if mod_bac_intensiv else "Mod normal activat.", icon="✅" if mod_bac_intensiv else "💬")
        st.rerun()
    if st.session_state.get("mod_bac_intensiv"):
        st.info("🎓 **BAC Intensiv activ** — focusat pe ce pică la examen.", icon="📝")

    # --- Desen automat Fizică ---
    if st.session_state.get("materie_selectata") == "fizică" or not st.session_state.get("materie_selectata"):
        desen_fizica = st.toggle(
            "🎨 Desen automat Fizică",
            value=st.session_state.get("desen_fizica", True),
            help="Profesorul desenează automat schema forțelor, circuite, raze optice etc. când rezolvă probleme de fizică."
        )
        if desen_fizica != st.session_state.get("desen_fizica", True):
            st.session_state.desen_fizica = desen_fizica
            st.session_state.system_prompt = get_system_prompt(
                st.session_state.get("materie_selectata"),
                pas_cu_pas=st.session_state.get("pas_cu_pas", False),
                desen_fizica=desen_fizica
            )
            if desen_fizica:
                st.toast("🎨 Desen automat activat!", icon="✅")
            else:
                st.toast("Desen automat dezactivat.", icon="🚫")
            st.rerun()

        if not st.session_state.get("desen_fizica", True):
            st.caption("🚫 Desenele automate sunt dezactivate.")

    st.divider()

    # --- Status Supabase ---
    if not st.session_state.get("_sb_online", True):
        st.markdown(
            '<div style="background:#e67e22;color:white;padding:8px 12px;'
            'border-radius:8px;font-size:13px;text-align:center;margin-bottom:8px">'
            '📴 Mod offline — datele sunt salvate local</div>',
            unsafe_allow_html=True
        )
    else:
        pending = len(st.session_state.get("_offline_queue", []))
        if pending:
            st.caption(f"☁️ {pending} mesaje în așteptare pentru sincronizare")


    st.divider()

    if st.button("🗑️ Șterge Istoricul", type="primary"):
        clear_history_db(st.session_state.session_id)
        st.session_state.messages = []
        st.rerun()

    enable_audio = st.checkbox("🔊 Voce", value=False)

    if enable_audio:
        voice_option = st.radio(
            "🎙️ Alege vocea:",
            options=["👨 Domnul Profesor (Emil)", "👩 Doamna Profesoară (Alina)"],
            index=0
        )
        selected_voice = VOICE_MALE_RO if "Emil" in voice_option else VOICE_FEMALE_RO
    else:
        selected_voice = VOICE_MALE_RO

    st.divider()

    st.header("📁 Materiale")

    # Tipuri de fișiere acceptate — imagini + documente
    uploaded_file = st.file_uploader(
        "Încarcă imagine, PDF sau document",
        type=["jpg", "jpeg", "png", "webp", "gif", "pdf"],
        help="Imaginile sunt analizate vizual de AI (culori, forme, text, obiecte). PDF-urile sunt citite integral."
    )
    media_content = None  # obiectul Google File trimis la AI

    # ── Uploadăm fișierul pe Google Files API (o singură dată per fișier) ──
    if uploaded_file:
        import os

        file_key   = f"_gfile_{uploaded_file.name}_{uploaded_file.size}"
        cached_gf  = st.session_state.get(file_key)

        # Dacă fișierul e deja încărcat și valid pe serverele Google, îl refolosim
        if cached_gf:
            try:
                gemini_client = genai.Client(api_key=keys[st.session_state.key_index])
                refreshed = gemini_client.files.get(cached_gf.name)
                if str(refreshed.state) in ("FileState.ACTIVE", "ACTIVE", "FileState.PROCESSING", "PROCESSING"):
                    media_content = refreshed
            except Exception:
                # Fișierul a expirat pe Google (TTL 48h) — îl re-uploadăm
                st.session_state.pop(file_key, None)
                cached_gf = None

        if not cached_gf:
            file_type = uploaded_file.type
            is_image  = file_type.startswith("image/")
            is_pdf    = "pdf" in file_type

            # Determină sufixul și mime_type corect
            suffix_map = {
                "image/jpeg": ".jpg", "image/jpg": ".jpg",
                "image/png": ".png",  "image/webp": ".webp",
                "image/gif": ".gif",  "application/pdf": ".pdf",
            }
            suffix    = suffix_map.get(file_type, ".bin")
            mime_type = file_type

            spinner_text = (
                "🖼️ Profesorul analizează imaginea..." if is_image
                else "📚 Se trimite documentul la AI..."
            )

            try:
                tmp_path = None
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    gemini_client = genai.Client(api_key=keys[st.session_state.key_index])

                    with st.spinner(spinner_text):
                        gfile = gemini_client.files.upload(file=tmp_path, config=genai_types.UploadFileConfig(mime_type=mime_type))
                        # Așteptăm procesarea (mai rapid pentru imagini, mai lent pentru PDF-uri mari)
                        poll = 0
                        while str(gfile.state) in ("FileState.PROCESSING", "PROCESSING") and poll < 60:
                            time.sleep(1)
                            gfile = gemini_client.files.get(gfile.name)
                            poll += 1

                    if gfile.state.name == "ACTIVE":
                        media_content = gfile
                        st.session_state[file_key] = gfile  # cache pentru reruns
                    else:
                        st.error(f"❌ Fișierul nu a putut fi procesat (stare: {gfile.state.name})")

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)

            except Exception as e:
                st.error(f"❌ Eroare la încărcarea fișierului: {e}")

        # ── Preview în sidebar ──
        if media_content:
            file_type = uploaded_file.type
            is_image  = file_type.startswith("image/")

            if is_image:
                st.image(uploaded_file, caption=f"🖼️ {uploaded_file.name}", use_container_width=True)
                st.success("✅ Imaginea e pe serverele Google — AI-ul o vede complet (culori, forme, text, obiecte).")
            else:
                st.success(f"✅ **{uploaded_file.name}** încărcat ({uploaded_file.size // 1024} KB)")
                st.caption("📄 AI-ul poate citi și analiza tot conținutul documentului.")

            # Buton de ștergere — curăță și de pe Google
            if st.button("🗑️ Elimină fișierul", use_container_width=True, key="remove_media"):
                file_key = f"_gfile_{uploaded_file.name}_{uploaded_file.size}"
                gf = st.session_state.pop(file_key, None)
                if gf:
                    try:
                        gemini_client = genai.Client(api_key=keys[st.session_state.key_index])
                        gemini_client.files.delete(gf.name)
                    except Exception:
                        pass  # dacă a expirat deja, ignorăm
                media_content = None
                st.rerun()

    st.divider()

    # --- Mod Quiz + BAC ---
    st.divider()
    st.subheader("📝 Examinare & BAC")

    def _clear_all_modes():
        for k in list(st.session_state.keys()):
            if k.startswith("bac_") or k.startswith("hw_"):
                st.session_state.pop(k, None)
        for k in ["quiz_active", "quiz_questions", "quiz_correct", "quiz_answers", "quiz_submitted"]:
            st.session_state.pop(k, None)

    col_q, col_b = st.columns(2)
    with col_q:
        if st.button("🎯 Quiz rapid", use_container_width=True,
                     type="primary" if st.session_state.get("quiz_mode") else "secondary"):
            entering = not st.session_state.get("quiz_mode", False)
            _clear_all_modes()
            st.session_state.quiz_mode = entering
            st.session_state.pop("bac_mode", None)
            st.session_state.pop("homework_mode", None)
            st.rerun()
    with col_b:
        if st.button("🎓 Simulare BAC", use_container_width=True,
                     type="primary" if st.session_state.get("bac_mode") else "secondary"):
            entering = not st.session_state.get("bac_mode", False)
            _clear_all_modes()
            st.session_state.bac_mode = entering
            st.session_state.pop("quiz_mode", None)
            st.session_state.pop("homework_mode", None)
            st.rerun()

    if st.button("📚 Corectează Temă", use_container_width=True,
                 type="primary" if st.session_state.get("homework_mode") else "secondary"):
        entering = not st.session_state.get("homework_mode", False)
        _clear_all_modes()
        st.session_state.homework_mode = entering
        st.session_state.pop("quiz_mode", None)
        st.session_state.pop("bac_mode", None)
        st.rerun()

    st.divider()

    # --- Istoric conversații ---
    st.subheader("🕐 Conversații anterioare")
    if st.button("🔄 Conversație nouă", use_container_width=True):
        new_sid = generate_unique_session_id()
        register_session(new_sid)
        switch_session(new_sid)
        st.rerun()

    sessions = get_session_list(limit=15)
    current_sid = st.session_state.session_id
    for s in sessions:
        is_current = s["session_id"] == current_sid
        label = f"{'▶ ' if is_current else ''}{s['preview']}"
        caption = f"{format_time_ago(s['last_active'])} · {s['msg_count']} mesaje"
        with st.container():
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    label,
                    key=f"sess_{s['session_id']}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary",
                    help=caption,
                ):
                    if not is_current:
                        switch_session(s["session_id"])
                        st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{s['session_id']}", help="Șterge"):
                    clear_history_db(s["session_id"])
                    if is_current:
                        st.session_state.messages = []
                    st.rerun()

    st.divider()

    if st.checkbox("🔧 Debug Info", value=False):
        msg_count = len(st.session_state.get("messages", []))
        st.caption(f"📊 Mesaje în memorie: {msg_count}/{MAX_MESSAGES_IN_MEMORY}")
        st.caption(f"🔑 Cheie API activă: {st.session_state.key_index + 1}/{len(keys)}")
        st.caption(f"🆔 Sesiune: {st.session_state.session_id[:16]}...")


# === MAIN UI — TEME / BAC / QUIZ / CHAT ===
if st.session_state.get("homework_mode"):
    run_homework_ui()
    st.stop()

if st.session_state.get("bac_mode"):
    run_bac_sim_ui()
    st.stop()

if st.session_state.get("quiz_mode"):
    run_quiz_ui()
    st.stop()

# === ÎNCĂRCARE MESAJE (CHAT MODE) ===
if "messages" not in st.session_state or not st.session_state.messages:
    st.session_state.messages = load_history_from_db(st.session_state.session_id)

# Banner mod Pas cu Pas
if st.session_state.get("pas_cu_pas"):
    st.markdown(
        '<div style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;'
        'padding:10px 16px;border-radius:10px;margin-bottom:12px;'
        'display:flex;align-items:center;gap:10px;font-size:14px;">'
        '🔢 <strong>Mod Pas cu Pas activ</strong> — '
        'Profesorul îți va explica fiecare problemă detaliat, cu motivația fiecărui pas.'
        '</div>',
        unsafe_allow_html=True
    )

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            render_message_with_svg(msg["content"])
        else:
            st.markdown(msg["content"])

    # Butoanele apar DOAR sub ultimul mesaj al profesorului
    if (msg["role"] == "assistant" and
            i == len(st.session_state.messages) - 1):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Nu am înțeles", key="qa_reexplain", use_container_width=True, help="Explică altfel, cu o altă analogie"):
                st.session_state["_quick_action"] = "reexplain"
                st.rerun()
        with col2:
            if st.button("✏️ Exercițiu similar", key="qa_similar", use_container_width=True, help="Generează un exercițiu similar pentru practică"):
                st.session_state["_quick_action"] = "similar"
                st.rerun()
        with col3:
            if st.button("🧠 Explică strategia", key="qa_strategy", use_container_width=True, help="Cum să gândești acest tip de problemă"):
                st.session_state["_quick_action"] = "strategy"
                st.rerun()


# ── Handler pentru butoanele de acțiuni rapide ──
TYPING_HTML = """
<div class="typing-indicator">
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span>Domnul Profesor scrie...</span>
</div>
"""

if st.session_state.get("_quick_action"):
    action = st.session_state.pop("_quick_action")
    ref = st.session_state.pop("_quick_action_ref", "")

    action_prompts = {
        "reexplain": "Nu am înțeles explicația anterioară. Te rog să explici altfel — folosește o altă analogie, o altă abordare sau un exemplu diferit din viața reală.",
        "similar":   "Generează un exercițiu similar cu cel de mai sus, cu date diferite, de dificultate puțin mai mare. Rezolvă-l complet după ce îl enunți.",
        "strategy":  "Explică-mi STRATEGIA pentru acest tip de problemă — cum recunosc că e acest tip, ce pași urmez în minte, ce capcane să evit. Fără calcule, doar gândirea."
    }
    injected = action_prompts.get(action, "")
    if injected:
        with st.chat_message("user"):
            st.markdown(injected)
        st.session_state.messages.append({"role": "user", "content": injected})
        save_message_with_limits(st.session_state.session_id, "user", injected)

        context_messages = get_context_for_ai(st.session_state.messages)
        history_obj = []
        for msg in context_messages:
            role_gemini = "model" if msg["role"] == "assistant" else "user"
            history_obj.append({"role": role_gemini, "parts": [msg["content"]]})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
            try:
                for text_chunk in run_chat_with_rotation(history_obj, [injected]):
                    full_response += text_chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.empty()
                render_message_with_svg(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                save_message_with_limits(st.session_state.session_id, "assistant", full_response)
            except Exception as e:
                st.error(f"❌ Eroare: {e}")
    st.stop()

# ── Handler întrebare sugerată — ÎNAINTE de afișarea butoanelor ──
if st.session_state.get("_suggested_question"):
    user_input = st.session_state.pop("_suggested_question")
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție automată materie ──
    _selector_materie = MATERII.get(st.session_state.get("materie_selectata", "🎓 Toate materiile"))
    if _selector_materie is None:
        _detected = detect_subject_from_text(user_input)
        _prev_detected = st.session_state.get("_detected_subject")
        if _detected and _detected != _prev_detected:
            update_system_prompt_for_subject(_detected)
    else:
        if st.session_state.get("_detected_subject") != _selector_materie:
            update_system_prompt_for_subject(_selector_materie)

    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)
        try:
            for text_chunk in run_chat_with_rotation(history_obj, [user_input]):
                full_response += text_chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.empty()
            render_message_with_svg(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)
        except Exception as e:
            st.error(f"❌ Eroare: {e}")
    st.rerun()

# ── Întrebări sugerate per materie — afișate doar când chat-ul e gol ──
INTREBARI_SUGERATE = {
    None: [
        "Explică-mi cum se rezolvă ecuațiile de gradul 2",
        "Ce este fotosinteza și cum funcționează?",
        "Cum se scrie un eseu la BAC?",
        "Explică legea lui Ohm cu un exemplu",
    ],
    "matematică": [
        "Cum rezolv o ecuație de gradul 2?",
        "Explică-mi derivatele — ce sunt și cum se calculează",
        "Cum calculez aria și volumul unui corp geometric?",
        "Ce este limita unui șir și cum o calculez?",
    ],
    "fizică": [
        "Explică legile lui Newton cu exemple",
        "Cum rezolv o problemă cu plan înclinat?",
        "Ce este legea lui Ohm și cum aplic în circuit?",
        "Explică reflexia și refracția luminii",
    ],
    "chimie": [
        "Cum echilibrez o ecuație chimică?",
        "Explică-mi legăturile chimice (ionică, covalentă)",
        "Cum calculez concentrația molară?",
        "Ce este regula lui Markovnikov?",
    ],
    "limba și literatura română": [
        "Cum structurez un eseu de BAC la Română?",
        "Explică-mi curentele literare principale",
        "Cum analizez o poezie — figuri de stil, prozodie",
        "Care sunt operele obligatorii la BAC Română?",
    ],
    "biologie": [
        "Explică-mi mitoza vs meioza",
        "Cum funcționează fotosinteza și respirația celulară?",
        "Ce este ADN-ul și cum funcționează codul genetic?",
        "Explică-mi legile lui Mendel cu pătrat Punnett",
    ],
    "informatică": [
        "Explică algoritmul de sortare prin selecție în C++",
        "Ce este recursivitatea? Exemplu cu factorial",
        "Cum funcționează căutarea binară?",
        "Explică-mi backtracking-ul cu un exemplu simplu",
    ],
    "geografie": [
        "Care sunt unitățile de relief ale României?",
        "Explică-mi clima României — regiuni și factori",
        "Care sunt râurile principale din România?",
        "Explică formarea Munților Carpați",
    ],
    "istorie": [
        "Explică Marea Unire din 1918 — cauze și consecințe",
        "Care au fost reformele lui Alexandru Ioan Cuza?",
        "Explică-mi perioada comunistă în România",
        "Ce s-a întâmplat la Revoluția din 1989?",
    ],
    "limba franceză": [
        "Explică-mi Passé Composé vs Imparfait",
        "Cum se acordă participiul trecut cu avoir și être?",
        "Explică Subjonctivul — când și cum se folosește",
        "Cum structurez un eseu în franceză?",
    ],
    "limba engleză": [
        "Explică Present Perfect vs Past Simple",
        "Cum funcționează propozițiile condiționale (tip 1, 2, 3)?",
        "Explică vocea pasivă în engleză",
        "Cum scriu un eseu argumentativ în engleză?",
    ],
}

if not st.session_state.get("messages"):
    materie_curenta = st.session_state.get("materie_selectata")
    intrebari = INTREBARI_SUGERATE.get(materie_curenta, INTREBARI_SUGERATE[None])
    st.markdown("##### 💡 Cu ce începem azi?")
    cols = st.columns(2)
    for i, intrebare in enumerate(intrebari):
        with cols[i % 2]:
            if st.button(intrebare, key=f"sugg_{i}", use_container_width=True):
                st.session_state["_suggested_question"] = intrebare
                st.rerun()

# === CHAT INPUT ===
if user_input := st.chat_input("Întreabă profesorul..."):

    # --- Debounce: blochează mesaje duplicate trimise rapid ---
    now_ts = time.time()
    last_msg = st.session_state.get("_last_user_msg", "")
    last_ts  = st.session_state.get("_last_msg_ts", 0)
    DEBOUNCE_SECONDS = 2.5

    if user_input.strip() == last_msg.strip() and (now_ts - last_ts) < DEBOUNCE_SECONDS:
        st.toast("⏳ Mesaj duplicat ignorat.", icon="🔁")
        st.stop()

    st.session_state["_last_user_msg"] = user_input
    st.session_state["_last_msg_ts"]  = now_ts

    # FIX BUG 1: Afișează și salvează mesajul utilizatorului ÎNAINTE de răspunsul AI
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție automată materie ──
    _selector_materie = MATERII.get(st.session_state.get("materie_selectata", "🎓 Toate materiile"))
    if _selector_materie is None:
        # Selectorul e pe "Toate" — detectăm automat din mesajul curent
        _detected = detect_subject_from_text(user_input)
        _prev_detected = st.session_state.get("_detected_subject")
        if _detected and _detected != _prev_detected:
            update_system_prompt_for_subject(_detected)
            st.toast(f"📚 Materie detectată: {_detected.capitalize()}", icon="🎯")
    else:
        # Selectorul are o materie specifică — o folosim pe aceea
        if st.session_state.get("_detected_subject") != _selector_materie:
            update_system_prompt_for_subject(_selector_materie)

    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})
    
    final_payload = []
    if media_content:
        # Prompt contextual bazat pe tipul fișierului încărcat
        fname = uploaded_file.name if uploaded_file else ""
        ftype = (uploaded_file.type if uploaded_file else "") or ""
        if ftype.startswith("image/"):
            final_payload.append(
                "Elevul ți-a trimis o imagine. Analizează-o vizual complet: "
                "descrie ce vezi (obiecte, persoane, text, culori, forme, diagrame, exerciții scrise de mână) "
                "și răspunde la întrebarea elevului ținând cont de tot conținutul vizual."
            )
        else:
            final_payload.append(
                f"Elevul ți-a trimis documentul '{fname}'. "
                "Citește și analizează tot conținutul înainte de a răspunde."
            )
        final_payload.append(media_content)
    final_payload.append(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Typing indicator înainte să înceapă streaming-ul
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        try:
            stream_generator = run_chat_with_rotation(history_obj, final_payload)
            first_chunk = True

            for text_chunk in stream_generator:
                full_response += text_chunk
                if first_chunk:
                    first_chunk = False  # typing indicator dispare la primul chunk

                if "<svg" in full_response or ("<path" in full_response and "stroke=" in full_response):
                    message_placeholder.markdown(
                        full_response.split("<path")[0] + "\n\n*🎨 Domnul Profesor desenează...*\n\n▌"
                    )
                else:
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            render_message_with_svg(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)
            
            if enable_audio:
                with st.spinner("🎙️ Domnul Profesor vorbește..."):
                    audio_file = generate_professor_voice(full_response, selected_voice)
                    
                    if audio_file:
                        st.audio(audio_file, format='audio/mp3')
                    else:
                        st.caption("🔇 Nu am putut genera vocea pentru acest răspuns.")
                        
        except Exception as e:
            st.error(f"❌ Eroare: {e}")

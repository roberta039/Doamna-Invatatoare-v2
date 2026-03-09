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


st.set_page_config(page_title="Doamna Învățătoare", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

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
    "🌟 Toate materiile":                              None,
    "📖 Limba și literatura română":                   "limba și literatura română",
    "🔢 Matematică":                                   "matematică",
    "🌿 Matematică și explorarea mediului (cls 1-2)":  "matematică și explorarea mediului",
    "🧪 Științe ale naturii (cls 3-4)":                "științe ale naturii",
    "🇬🇧 Limba modernă (Engleză)":                     "limba engleză",
    "🎭 Educație civică (cls 3-4)":                    "educație civică",
    "💛 Dezvoltare personală":                         "dezvoltare personală",
    "🎨 Arte vizuale și lucru manual":                 "arte vizuale și lucru manual",
    "🎵 Muzică și mișcare":                            "muzică și mișcare",
    "🏃 Educație fizică":                              "educație fizică",
    "🙏 Religie":                                      "religie",
}


def get_system_prompt(materie: str | None = None, pas_cu_pas: bool = False, desen_fizica: bool = True,
                      mod_strategie: bool = False, mod_bac_intensiv: bool = False) -> str:
    """Returnează System Prompt adaptat pentru Doamna Învățătoare (clasele 1-4)."""

    if materie:
        rol_line = (
            f"ROL: Ești Doamna Învățătoare, o învățătoare drăguță și răbdătoare din România, "
            f"specializată în {materie.upper()} pentru clasele I-IV. "
            f"Răspunde EXCLUSIV la întrebări legate de {materie} la nivel de clasele 1-4. "
            f"Dacă copilul întreabă despre altă materie, îndrumă-l prietenos să schimbe materia."
        )
    else:
        rol_line = (
            "ROL: Ești Doamna Învățătoare, o învățătoare drăguță, răbdătoare și veselă din România, "
            "care predă toate materiile pentru clasele I-IV: Matematică, Română, Științe, Geografie, "
            "Engleză, Educație civică, Arte, Muzică, Educație fizică și Dezvoltare personală."
        )

    pas_cu_pas_bloc = """

    ═══════════════════════════════════════════════════
    MOD ACTIV: EXPLICAȚIE PAS CU PAS (PRIORITATE MAXIMĂ)
    ═══════════════════════════════════════════════════
    Copilul a cerut explicație pas cu pas. Respectă OBLIGATORIU:

    FORMAT OBLIGATORIU:
    **📋 Ce știm:**
    - Datele din problemă, scrise simplu

    **🎯 Ce vrem să aflăm:**
    - Ce trebuie calculat sau răspuns

    **👣 Rezolvare pas cu pas:**
    **Pasul 1 — [ce facem]:** [explicație simplă + de ce]
    **Pasul 2 — [ce facem]:** [explicație simplă + de ce]

    **✅ Răspunsul:** [rezultatul clar]

    **💡 Să ne amintim:**
    - O idee importantă din exercițiu
    ═══════════════════════════════════════════════════
""" if pas_cu_pas else ""

    return """
ROL: """ + rol_line + pas_cu_pas_bloc + """

    TON ȘI ADRESARE (CRITIC — respectă întotdeauna):
    1. Vorbește SIMPLU, cald și vesel — ca o învățătoare adevărată cu copiii mici.
    2. Folosește cuvinte pe care un copil de 6-10 ani le înțelege.
    3. Folosește emoji-uri vesele des: ⭐, 🌟, 😊, 🎉, 👏, 🌈, 🐣, 🦋, ✏️, 📚
    4. Laudă copilul când răspunde corect: "Bravo!", "Super!", "Foarte bine!", "Ești grozav/grozavă!"
    5. Când greșește, fii blândă: "Aproape! Hai să încercăm împreună..." sau "Nu-i nimic, greșim ca să învățăm!"
    6. NU SALUTA în fiecare mesaj. Salută DOAR la prima întrebare.
    7. Răspunsuri SCURTE și CLARE — copiii nu citesc texte lungi.
    8. Folosește exemple din viața de zi cu zi a copiilor (jucării, animale, fructe, școală).
    9. Te prezinți ca "Doamna Învățătoare" sau "Doamna".
    10. Vorbești la feminin: "sunt bucuroasă", "sunt gata", "am pregătit".

    REGULĂ STRICTĂ: Explică TOTUL la nivelul clasei I-IV.
    - Matematică: adunare, scădere, înmulțire, împărțire, fracții simple, probleme cu text simplu
    - NU folosi termeni complicați fără să îi explici imediat cu un exemplu simplu
    - Dacă ceva e prea greu pentru vârsta lor, spune: "Asta vei învăța mai târziu! Acum să facem..."

    GHID PE MATERII:

    1. MATEMATICĂ (clasele 1-4):
       - Adunare și scădere: explică cu obiecte concrete ("Dacă ai 3 mere și mai primești 2...")
       - Înmulțire: tabela înmulțirii pas cu pas, cu povești ("3 grupuri de câte 4 copii")
       - Împărțire: "Împărțim 12 bomboane la 4 prieteni — câte primește fiecare?"
       - Fracții simple: ½, ¼ — "Tai un măr în 2 părți egale, fiecare parte e o jumătate"
       - Probleme cu text: ÎNTÂI citim împreună, APOI scriem ce știm și ce căutăm
       - Geometrie: forme simple (pătrat, dreptunghi, triunghi, cerc) cu exemple din clasă
       - Unități de măsură: cm, m, kg, g, l, ml, ore, minute — cu exemple practice
       - Scrie calculele clar: 3 + 5 = 8, NU formule complicate

    2. LIMBA ȘI LITERATURA ROMÂNĂ (clasele 1-4):
       - Citire: ajută copilul să citească silabisind dacă e nevoie: "MA-MĂ", "CA-SA"
       - Scriere: dictare, copiere, propoziții simple
       - Gramatică simplă: substantiv ("numele lucrurilor"), verb ("ce face"), adjectiv ("cum e")
       - Literele alfabetului: A, B, C... cu exemple de cuvinte pentru fiecare
       - Propoziții: subiect + predicat explicat simplu ("Cine face ceva?")
       - Despărțirea în silabe: bate din palme pentru fiecare silabă
       - Semne de punctuație: punct, virgulă, semnul întrebării, semnul exclamării
       - Povești și lecturi: Capra cu trei iezi, Punguța cu doi bani, Fata babei și fata moșneagului

    3. ȘTIINȚE ALE NATURII (clasele 3-4):
       - Plante: rădăcină, tulpină, frunze, floare, fruct — cu desene simple
       - Animale: domestice vs sălbatice, hrană, înmulțire
       - Corpul uman: organe principale explicat simplu (inima pompează sânge ca o pompă)
       - Anotimpuri: primăvară, vară, toamnă, iarnă — ce se întâmplă în natură
       - Apa: stări (lichidă, solidă, gazoasă), circuitul apei în natură simplu
       - Mediul înconjurător: protejarea naturii, reciclare

    4. GEOGRAFIE (clasele 3-4):
       - România: țara noastră, capitala București, vecinii
       - Forme de relief: munte, deal, câmpie — cu exemple și desene simple
       - Râuri principale: Dunărea, Mureșul, Oltul
       - Harta: punctele cardinale (Nord, Sud, Est, Vest) — "Soarele răsare la Est"
       - Continente și oceane: cele 7 continente, cele mai importante oceane

    5. LIMBA ENGLEZĂ (clasele 1-4):
       - Vocabular de bază: culori (red, blue, green), numere (one, two, three), animale, familie
       - Salutări: Hello, Good morning, How are you? — cu pronunție fonetică în română
       - Propoziții simple: "This is a cat", "I have a dog"
       - Alfabetul englez: A, B, C... cu pronunție
       - Cântece și rime simple în engleză

    6. EDUCAȚIE CIVICĂ (clasele 3-4):
       - Reguli de comportament: politețe, respect, ajutor reciproc
       - Familie și comunitate: roluri, drepturi și responsabilități
       - Reguli de circulație pentru copii: cum traversăm strada

    7. ARTE VIZUALE ȘI LUCRU MANUAL:
       - Culori primare și secundare: roșu + galben = portocaliu
       - Tehnici simple de desen și pictură
       - Activități practice: pliere, lipire, decupare

    8. MUZICĂ ȘI MIȘCARE:
       - Note muzicale: Do, Re, Mi, Fa, Sol, La, Si
       - Cântece pentru copii din programa școlară
       - Ritm și mișcare

    9. EDUCAȚIE FIZICĂ:
       - Exerciții simple, jocuri de mișcare
       - Reguli sportive de bază

    10. DEZVOLTARE PERSONALĂ:
        - Emoții: bucurie, tristețe, frică, furie — cum le recunoaștem și gestionăm
        - Prietenie, cooperare, empatie
        - Igiena personală și rutine zilnice

    11. RELIGIE:
        - Rugăciuni simple, sărbători creștine (Crăciun, Paște)
        - Valori morale: bunătate, cinste, respect

    DESENARE AUTOMATĂ (IMPORTANT):
    Dacă problema sau întrebarea implică ceva vizual, desenează AUTOMAT pentru a ajuta copilul:
    ✅ Probleme de matematică: desenează obiectele din problemă (mere, bile, copii)
    ✅ Forme geometrice: pătrat, triunghi, cerc colorat
    ✅ Plante și animale: scheme simple și colorate
    ✅ Hărți simple: România, puncte cardinale
    ✅ Corpul uman: schema simplă cu organe principale
    ✅ Fracții: împarte forme în părți egale
    Folosește tag-urile [[DESEN_SVG]]..[[/DESEN_SVG]] și culori vesele, vii!

    REGULI DESEN PENTRU COPII:
    - Culori vesele și saturate: roșu, galben, verde, albastru, portocaliu
    - Forme simple și mari — ușor de înțeles
    - Etichete text mari și clare pe desen
    - Dacă desenezi animale sau obiecte: stilizat, simplu, drăguț
    - Fundalul: alb sau foarte deschis
    - Dimensiune viewBox: 0 0 600 400

    FUNCȚIE SPECIALĂ - DESENARE (SVG):
    Dacă copilul cere un desen, o schemă sau o figură:
    1. Generează cod SVG valid și colorat.
    2. Codul trebuie încadrat STRICT între tag-uri:
       [[DESEN_SVG]]
       <svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
          <!-- Codul tău aici -->
       </svg>
       [[/DESEN_SVG]]
    3. Folosește culori vesele și forme simple, potrivite pentru copii.
    4. Adaugă întotdeauna etichete text mari și clare.
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
        "adunare", "scădere", "înmulțire", "inmultire", "împărțire", "impartire",
        "plus", "minus", "egal", "număr", "numere", "cifre", "cifra", "zeci", "sute",
        "problemă", "problema", "calcul", "rezultat", "fracție", "fractie", "jumătate",
        "sfert", "triunghi", "pătrat", "patrat", "dreptunghi", "cerc", "formă", "forma",
        "cm", "metri", "kg", "gram", "litru", "ore", "minute", "matematica", "mate",
        "tabela", "înmulțirii", "înmultirii", "exercitiu", "exercițiu",
    ],
    "limba și literatura română": [
        "silabe", "silabă", "silaba", "literă", "litera", "cuvânt", "cuvant", "propoziție",
        "propozitie", "substantiv", "verb", "adjectiv", "punct", "virgulă", "virgula",
        "dictare", "compunere", "poveste", "lectură", "lectura", "alfabet", "vocală",
        "vocala", "consoană", "consoana", "română", "romana", "citire", "scriere",
        "despărțire", "despartire", "cratimă", "cratima", "rimă", "rima",
    ],
    "științe ale naturii": [
        "plantă", "planta", "animal", "rădăcină", "radacina", "tulpină", "tulpina",
        "frunze", "floare", "fruct", "insecte", "mamifere", "păsări", "pasari",
        "anotimpuri", "primăvară", "primavara", "vară", "vara", "toamnă", "toamna",
        "iarnă", "iarna", "apă", "apa", "aer", "sol", "corpul", "inimă", "inima",
        "plămâni", "plamani", "stiinte", "știinte", "natură", "natura", "mediu",
    ],
    "geografie": [
        "românia", "romania", "bucurești", "bucuresti", "munte", "deal", "câmpie",
        "campie", "râu", "rau", "dunărea", "dunarea", "hartă", "harta", "nord", "sud",
        "est", "vest", "continent", "ocean", "europa", "geografie", "vecinii",
    ],
    "limba engleză": [
        "english", "engleză", "engleza", "hello", "good morning", "colours", "colors",
        "numbers", "animals", "family", "school", "red", "blue", "green", "one", "two",
        "three", "cat", "dog", "house",
    ],
    "educație civică": [
        "politețe", "politete", "respect", "reguli", "comunitate", "familie",
        "drepturi", "responsabilități", "responsabilitati", "civica", "civică",
        "circulatie", "circulație", "strada", "semafoare",
    ],
    "arte vizuale și lucru manual": [
        "desen", "pictură", "pictura", "culori", "roșu", "rosu", "galben", "albastru",
        "verde", "portocaliu", "violet", "lipire", "decupare", "pliere", "origami",
        "arte", "vizuale", "manual", "creion", "pensulă", "pensula",
    ],
    "muzică și mișcare": [
        "muzică", "muzica", "notă", "nota", "do", "re", "mi", "fa", "sol", "la", "si",
        "cântec", "cantec", "ritm", "melodie", "instrumente", "mișcare", "miscare",
    ],
    "educație fizică": [
        "sport", "mișcare", "miscare", "exerciții", "exercitii", "alergare", "sărituri",
        "sarituri", "joc", "echipă", "echipa", "fizica", "fizică", "gimnastică", "gimnastica",
    ],
    "dezvoltare personală": [
        "emoții", "emotii", "bucurie", "tristețe", "tristete", "frică", "frica",
        "furie", "prietenie", "prieteni", "ajutor", "empatie", "igienă", "igiena",
        "rutine", "personala", "personală",
    ],
    "religie": [
        "rugăciune", "rugaciune", "crăciun", "craciun", "paște", "paste", "dumnezeu",
        "biserică", "biserica", "sfânt", "sfant", "religie", "bunătate", "bunatate",
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
# === FUNCȚII SPECIALE (dezactivate pentru clasele 1-4) ===
# Simulare BAC, Teme, Quiz — nu sunt relevante pentru învățământ primar

# === UI PRINCIPAL ===
st.title("👩‍🏫 Doamna Învățătoare")

with st.sidebar:
    st.header("⚙️ Setări")

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
        st.info(f"📚 Materia: **{materie_label}**")

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
        help="Doamna va explica fiecare exercițiu pas cu pas, simplu și clar."
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
    mod_strategie = False  # Dezactivat pentru clasele 1-4

    # --- Desene automate ---
    mod_bac_intensiv = False  # Dezactivat pentru clasele 1-4
    desen_fizica = True  # Mereu activ pentru copii — desene colorate ajută înțelegerea

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
            options=["👩 Doamna Învățătoare (Maria)", "👧 Asistentă (Alina)"],
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
        st.caption(f"🔑 Cheie activă: {st.session_state.key_index + 1}/{len(keys)}")
        st.caption(f"🆔 Sesiune: {st.session_state.session_id[:16]}...")


# === CHAT MODE ===

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
            if st.button("💡 Cum să gândesc?", key="qa_strategy", use_container_width=True, help="Cum să gândești acest tip de problemă"):
                st.session_state["_quick_action"] = "strategy"
                st.rerun()


# ── Handler pentru butoanele de acțiuni rapide ──
TYPING_HTML = """
<div class="typing-indicator">
    <div class="typing-dots"><span></span><span></span><span></span></div>
    <span>Doamna Învățătoare scrie... ✏️</span>
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
        st.rerun()

# ── Handler întrebare sugerată — ÎNAINTE de afișarea butoanelor ──
if st.session_state.get("_suggested_question"):
    user_input = st.session_state.pop("_suggested_question")
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție automată materie ──
    _selector_materie = MATERII.get(st.session_state.get("materie_selectata", "🌟 Toate materiile"))
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
        "Ajută-mă cu o problemă de matematică! ➕",
        "Vreau să scriu o compunere! ✏️",
        "Explică-mi ceva despre animale! 🐾",
        "Cum se desparte un cuvânt în silabe? 📚",
    ],
    "matematică": [
        "Cum rezolv o problemă cu adunare? ➕",
        "Ajută-mă cu tabela înmulțirii! ✖️",
        "Ce sunt fracțiile? 🍕",
        "Cum calculez perimetrul unui pătrat? 📐",
    ],
    "limba și literatura română": [
        "Ajută-mă să despart cuvinte în silabe! 🔤",
        "Ce este un substantiv? 📝",
        "Cum scriu o compunere? ✍️",
        "Spune-mi o poveste! 📖",
    ],
    "științe ale naturii": [
        "Cum cresc plantele? 🌱",
        "Care sunt părțile corpului uman? 🫀",
        "Ce animale trăiesc în pădure? 🦊",
        "Ce se întâmplă iarna în natură? ❄️",
    ],
    "geografie": [
        "Unde e România pe hartă? 🗺️",
        "Ce este un munte? ⛰️",
        "Care sunt punctele cardinale? 🧭",
        "Ce râuri mari avem în România? 🏞️",
    ],
    "limba engleză": [
        "Cum spun culorile în engleză? 🎨",
        "Cum număr până la 10 în engleză? 🔢",
        "Cum mă prezint în engleză? 👋",
        "Cum spun animalele în engleză? 🐱",
    ],
}

# ── Input chat principal ──
if user_input := st.chat_input("Întreabă Doamna Învățătoare... ✏️"):

    now_ts = time.time()
    last_msg = st.session_state.get("_last_user_msg", "")
    last_ts  = st.session_state.get("_last_msg_ts", 0)
    DEBOUNCE_SECONDS = 2.5

    if user_input.strip() == last_msg.strip() and (now_ts - last_ts) < DEBOUNCE_SECONDS:
        st.toast("⏳ Mesaj duplicat ignorat.", icon="🔁")
        st.stop()

    st.session_state["_last_user_msg"] = user_input
    st.session_state["_last_msg_ts"]  = now_ts

    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})
    save_message_with_limits(st.session_state.session_id, "user", user_input)

    # ── Detecție automată materie ──
    _toate_key = "🌟 Toate materiile" if "app3.py" == "app3.py" else "🌺 Toate materiile"
    _selector_materie = MATERII.get(st.session_state.get("materie_selectata", _toate_key))
    if _selector_materie is None:
        _detected = detect_subject_from_text(user_input)
        _prev_detected = st.session_state.get("_detected_subject")
        if _detected and _detected != _prev_detected:
            update_system_prompt_for_subject(_detected)
            st.toast(f"📚 Materie detectată: {_detected.capitalize()}", icon="🎯")
    else:
        if st.session_state.get("_detected_subject") != _selector_materie:
            update_system_prompt_for_subject(_selector_materie)

    context_messages = get_context_for_ai(st.session_state.messages)
    history_obj = []
    for msg in context_messages:
        role_gemini = "model" if msg["role"] == "assistant" else "user"
        history_obj.append({"role": role_gemini, "parts": [msg["content"]]})

    final_payload = []
    if media_content:
        fname_up = uploaded_file.name if uploaded_file else ""
        ftype = (uploaded_file.type if uploaded_file else "") or ""
        if ftype.startswith("image/"):
            final_payload.append(
                "Elevul ți-a trimis o imagine. Analizează-o vizual complet și răspunde la întrebarea elevului."
            )
        else:
            final_payload.append(
                f"Elevul ți-a trimis documentul '{fname_up}'. Citește și analizează tot conținutul înainte de a răspunde."
            )
        final_payload.append(media_content)
    final_payload.append(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        message_placeholder.markdown(TYPING_HTML, unsafe_allow_html=True)

        try:
            stream_generator = run_chat_with_rotation(history_obj, final_payload)
            first_chunk = True

            for text_chunk in stream_generator:
                full_response += text_chunk
                if first_chunk:
                    first_chunk = False
                if "<svg" in full_response or ("<path" in full_response and "stroke=" in full_response):
                    message_placeholder.markdown(
                        full_response.split("<path")[0] + "\n\n*🎨 Doamna desenează...*\n\n▌",
                        unsafe_allow_html=True
                    )
                else:
                    message_placeholder.markdown(full_response + "▌")

            message_placeholder.empty()
            render_message_with_svg(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_message_with_limits(st.session_state.session_id, "assistant", full_response)

        except Exception as e:
            message_placeholder.empty()
            err = str(e)
            st.error(f"❌ {err}")

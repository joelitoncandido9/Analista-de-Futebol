"""Configuracao centralizada do Football AI."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# Diretorio base
HOME = Path(os.getenv("FOOTBALL_AI_HOME", "/home/palpites/football_ai"))

# API-Football (3 keys rotacionadas)
API_FOOTBALL_KEYS = [
    os.getenv("API_FOOTBALL_KEY1"),
    os.getenv("API_FOOTBALL_KEY2"),
    os.getenv("API_FOOTBALL_KEY3"),
]
API_FOOTBALL_KEYS = [k for k in API_FOOTBALL_KEYS if k]
API_FOOTBALL_URL = "https://v3.football.api-sports.io"
API_FOOTBALL_RATE = 10  # req/min por key

# The-Odds-API (6-7 keys rotacionadas)
THE_ODDS_API_KEY = os.getenv("THE_ODDS_API_KEY")  # keep for backward compat
THE_ODDS_API_KEYS = [k for k in [
    os.getenv("THE_ODDS_API_KEY"),
    os.getenv("ODDS_API_KEY"),
    os.getenv("ODDS_KEY1"),
    os.getenv("ODDS_KEY2"),
    os.getenv("ODDS_KEY3"),
    os.getenv("ODDS_KEY4"),
    os.getenv("ODDS_KEY5"),
] if k]
THE_ODDS_API_URL = "https://api.the-odds-api.com/v4"

# OpenRouter (acesso a Claude e outros modelos)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4")

# Claude / Anthropic (fallback caso OpenRouter nao configurado)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# OpenAI (embeddings)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# WhatsApp (Evolution API)
WA_NUMBER = os.getenv("WA_NUMBER", "558393066653")
EVO_API_KEY = os.getenv("EVO_API_KEY")
EVO_BASE_URL = os.getenv("EVO_BASE_URL", "http://localhost:8080")
EVO_INSTANCE = os.getenv("EVO_INSTANCE", "Corretor")

# Caminhos de dados
DATA_DIR = HOME / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = DATA_DIR / "exports"
LOGS_DIR = HOME / "logs"
MODELS_DIR = HOME / "models" / "saved"

# Banco SQLite
DB_PATH = HOME / "database" / "football_ai.db"

# Banco legado (Analista v1)
ANALISTA_V1_DB = Path(os.getenv("ANALISTA_V1_DB", "/home/palpites/analista/dados/analista.db"))
FOOTBALL_DATA_DB = Path(os.getenv("FOOTBALL_DATA_DB", "/home/palpites/analista/dados/football_data.db"))

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Criar diretorios se nao existirem
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, EXPORTS_DIR, LOGS_DIR, MODELS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

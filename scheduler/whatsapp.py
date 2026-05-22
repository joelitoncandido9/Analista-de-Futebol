"""Envio de mensagens WhatsApp via Evolution API.

Usa a instância Evolution API rodando em Docker para enviar
alertas e relatorios diretamente para o WhatsApp do usuario.
"""
import json
import subprocess

from loguru import logger

from config.settings import WA_NUMBER, EVO_API_KEY, EVO_BASE_URL, EVO_INSTANCE

EVO_RECIPIENT = WA_NUMBER or "558393066653"


def _get_container() -> str | None:
    """Descobre o nome do container Evolution API em execucao."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=evolution_api", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        containers = [c.strip() for c in result.stdout.split("\n") if c.strip()]
        # Preferir container que esta "Up"
        for name in containers:
            status = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", name],
                capture_output=True, text=True, timeout=5,
            )
            if status.stdout.strip() == "running":
                return name
        return containers[0] if containers else None
    except Exception as e:
        logger.warning(f"[WhatsApp] Erro ao buscar container: {e}")
        return None


def _api_call(endpoint: str, data: dict) -> dict | None:
    """Faz chamada a Evolution API dentro do container Docker."""
    container = _get_container()
    if not container:
        logger.warning("[WhatsApp] Container Evolution API nao encontrado")
        return None

    payload = json.dumps(data)
    escaped_payload = json.dumps(payload)
    cmd = [
        "docker", "exec", container,
        "node", "-e",
        f"""
        const http = require('http');
        const data = {escaped_payload};
        const req = http.request({{
            hostname:'localhost', port:8080,
            path:'{endpoint}',
            method:'POST',
            headers:{{
                'Content-Type':'application/json',
                'apiKey':'{EVO_API_KEY}',
                'Content-Length':Buffer.byteLength(data)
            }}
        }}, (res) => {{
            let body = '';
            res.on('data', d => body += d);
            res.on('end', () => console.log(JSON.stringify({{status:res.statusCode,body}})));
        }});
        req.on('error', e => console.log(JSON.stringify({{error:e.message}})));
        req.write(data);
        req.end();
        """
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0 and result.stdout.strip():
            resp = json.loads(result.stdout.strip())
            if resp.get("status") == 201 or resp.get("status") == 200:
                return resp.get("body")
            else:
                logger.warning(f"[WhatsApp] Erro API: {resp}")
                return None
        else:
            logger.warning(f"[WhatsApp] Erro docker: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        logger.warning("[WhatsApp] Timeout ao chamar Evolution API")
        return None
    except Exception as e:
        logger.warning(f"[WhatsApp] Erro: {e}")
        return None


def send_text(message: str, number: str | None = None) -> bool:
    """Envia mensagem de texto via WhatsApp.

    Args:
        message: Texto da mensagem.
        number: Telefone destino (padrao: numero do usuario).

    Returns:
        True se enviou com sucesso.
    """
    recipient = number or EVO_RECIPIENT
    payload = {"number": recipient, "text": message}

    result = _api_call(f"/message/sendText/{EVO_INSTANCE}", payload)
    if result is not None:
        logger.info(f"[WhatsApp] Mensagem enviada para {recipient}")
        return True

    logger.warning(f"[WhatsApp] Falha ao enviar para {recipient}")
    return False


def send_pre_match(home: str, away: str, league: str,
                    corners_pred: dict | None = None,
                    result_pred: dict | None = None) -> bool:
    """Envia relatorio pre-jogo via WhatsApp."""
    lines = [
        f"*{home} x {away}*",
        f"Liga: {league}",
        "",
    ]
    if corners_pred:
        lines.append(f"Escanteios previstos: {corners_pred.get('predicted_total_corners', '?')}")
        for line, prob in list(corners_pred.get("probabilities", {}).items())[:4]:
            lines.append(f"  {line}: {prob:.1%}")
    if result_pred:
        lines.append("")
        lines.append(f"Resultado provavel: {result_pred.get('most_likely_score', '?')}")
        lines.append(f"Casa: {result_pred.get('prob_home', 0):.1%} | "
                      f"Emp: {result_pred.get('prob_draw', 0):.1%} | "
                      f"Fora: {result_pred.get('prob_away', 0):.1%}")
    return send_text("\n".join(lines))


def send_daily_summary(stats: dict) -> bool:
    """Envia resumo diario."""
    lines = ["*Resumo Football AI*", "Partidas no banco:"]
    for league, count in stats.get("matches_per_league", {}).items():
        lines.append(f"  {league}: {count}")
    return send_text("\n".join(lines))

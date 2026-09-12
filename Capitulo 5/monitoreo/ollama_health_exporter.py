#!/usr/bin/env python3
"""
ollama_health_exporter.py
----------------------------------------------------------------------------
Exportador ligero de métricas Prometheus para un servidor Ollama.

Por qué existe este script: Ollama no expone un endpoint nativo /metrics
compatible con Prometheus (a septiembre de 2026 sigue siendo una petición
abierta de la comunidad, ver github.com/ollama/ollama/issues/3144). Este
script consulta periódicamente la API propia de Ollama (/api/ps y
/api/version) y traduce esa información a métricas Prometheus estándar,
sin dependencias externas más allá de la librería estándar de Python.

No requiere privilegios de administrador ni modificar la instalación de
Ollama: es un proceso independiente que solo hace peticiones HTTP de
lectura contra la API local.

Uso:
    python3 ollama_health_exporter.py [--ollama-url http://localhost:11434]
                                       [--port 9273] [--interval 15]

Métricas expuestas en http://localhost:9273/metrics :
    ollama_up                        1 si Ollama respondió, 0 si no
    ollama_loaded_models_total       número de modelos actualmente cargados
    ollama_model_size_bytes{model=}  tamaño en RAM/VRAM de cada modelo cargado
    ollama_scrape_duration_seconds   tiempo que tardó la última consulta

Ver monitoreo/docker-compose-monitoring.yml para desplegarlo junto a
Prometheus y Grafana, y monitoreo/prometheus.yml para el scrape config.
----------------------------------------------------------------------------
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

STATE = {
    "up": 0,
    "loaded_models": [],
    "scrape_duration": 0.0,
}
STATE_LOCK = Lock()


def fetch_ollama_ps(ollama_url: str, timeout: float = 5.0):
    """Consulta /api/ps y devuelve la lista de modelos cargados, o levanta
    una excepción si Ollama no responde."""
    req = urllib.request.Request(f"{ollama_url.rstrip('/')}/api/ps")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("models", [])


def poll_loop(ollama_url: str, interval: int):
    while True:
        start = time.monotonic()
        try:
            models = fetch_ollama_ps(ollama_url)
            with STATE_LOCK:
                STATE["up"] = 1
                STATE["loaded_models"] = models
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            with STATE_LOCK:
                STATE["up"] = 0
                STATE["loaded_models"] = []
            print(f"[ollama_health_exporter] No se pudo consultar {ollama_url}: {exc}")
        finally:
            with STATE_LOCK:
                STATE["scrape_duration"] = time.monotonic() - start
        time.sleep(interval)


def render_metrics() -> str:
    with STATE_LOCK:
        up = STATE["up"]
        models = list(STATE["loaded_models"])
        duration = STATE["scrape_duration"]

    lines = [
        "# HELP ollama_up 1 si el servidor Ollama respondió a la última consulta, 0 si no.",
        "# TYPE ollama_up gauge",
        f"ollama_up {up}",
        "# HELP ollama_loaded_models_total Número de modelos actualmente cargados en memoria/VRAM.",
        "# TYPE ollama_loaded_models_total gauge",
        f"ollama_loaded_models_total {len(models)}",
        "# HELP ollama_model_size_bytes Tamaño reportado por Ollama de cada modelo cargado.",
        "# TYPE ollama_model_size_bytes gauge",
    ]
    for model in models:
        name = model.get("name", "desconocido").replace('"', "'")
        size = model.get("size", 0)
        lines.append(f'ollama_model_size_bytes{{model="{name}"}} {size}')
    lines += [
        "# HELP ollama_scrape_duration_seconds Duración de la última consulta a la API de Ollama.",
        "# TYPE ollama_scrape_duration_seconds gauge",
        f"ollama_scrape_duration_seconds {duration:.4f}",
    ]
    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802 (nombre requerido por BaseHTTPRequestHandler)
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return
        body = render_metrics().encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # silencia el log por defecto
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ollama-url", default="http://localhost:11434",
                         help="URL base del servidor Ollama a monitorear")
    parser.add_argument("--port", type=int, default=9273,
                         help="Puerto donde este exportador expone /metrics")
    parser.add_argument("--interval", type=int, default=15,
                         help="Segundos entre consultas a la API de Ollama")
    args = parser.parse_args()

    import threading
    poller = threading.Thread(
        target=poll_loop, args=(args.ollama_url, args.interval), daemon=True
    )
    poller.start()

    server = HTTPServer(("0.0.0.0", args.port), MetricsHandler)
    print(f"[ollama_health_exporter] Sirviendo métricas en :{args.port}/metrics "
          f"(monitoreando {args.ollama_url} cada {args.interval}s)")
    server.serve_forever()


if __name__ == "__main__":
    main()

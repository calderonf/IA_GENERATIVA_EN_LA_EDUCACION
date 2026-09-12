#!/usr/bin/env bash
# setup_openwebui_docker.sh
# ---------------------------------------------------------------------------
# Despliega OpenWebUI en Docker, conectado a una instancia de Ollama que ya
# esté corriendo (local o en otro servidor de la red institucional).
#
# Requisitos: Docker y Docker Compose instalados.
#
# Uso:
#   chmod +x setup_openwebui_docker.sh
#   ./setup_openwebui_docker.sh [URL_DE_OLLAMA]
#
# Si no se indica URL_DE_OLLAMA, se asume que Ollama corre en la misma
# máquina (http://host.docker.internal:11434 en Docker Desktop, o la IP del
# host en Linux nativo).
# ---------------------------------------------------------------------------
set -euo pipefail

OLLAMA_URL="${1:-http://host.docker.internal:11434}"
COMPOSE_FILE="$(dirname "$0")/../plantillas/docker-compose-openwebui.yml"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker no está instalado. Instálalo antes de continuar: https://docs.docker.com/get-docker/" >&2
  exit 1
fi

if [[ ! -f "${COMPOSE_FILE}" ]]; then
  echo "No se encontró la plantilla ${COMPOSE_FILE}." >&2
  echo "Ejecuta este script desde una copia completa de la carpeta 'Capitulo 5' del repositorio." >&2
  exit 1
fi

export OLLAMA_BASE_URL="${OLLAMA_URL}"

echo "==> Desplegando OpenWebUI (Ollama en: ${OLLAMA_URL})"
if docker compose version >/dev/null 2>&1; then
  docker compose -f "${COMPOSE_FILE}" up -d
else
  docker-compose -f "${COMPOSE_FILE}" up -d
fi

echo
echo "Listo. Abre http://localhost:3000 en el navegador para crear la cuenta"
echo "de administrador (la primera cuenta creada obtiene privilegios de admin)."
echo
echo "Si OpenWebUI no logra conectar con Ollama:"
echo "  * En Linux nativo (sin Docker Desktop), host.docker.internal puede no"
echo "    resolver; usa la IP real del host, por ejemplo:"
echo "      ./setup_openwebui_docker.sh http://192.168.1.50:11434"
echo "  * Revisa guia_solucion_problemas.md en esta carpeta."

#!/usr/bin/env bash
# configure_multiuser_ollama.sh
# ---------------------------------------------------------------------------
# Reconfigura una instalación EXISTENTE de Ollama (Linux, systemd) para
# habilitar acceso multiusuario, sin reinstalar nada. Útil cuando Ollama ya
# se instaló en modo "solo esta máquina" y ahora se quiere convertir en un
# servidor centralizado para un laboratorio o departamento.
#
# Uso:
#   chmod +x configure_multiuser_ollama.sh
#   sudo ./configure_multiuser_ollama.sh [--proxy http://proxy.universidad.edu:8080]
# ---------------------------------------------------------------------------
set -euo pipefail

MODELS_DIR="${OLLAMA_MODELS_DIR:-/shared/models}"
BIND_ADDR="${OLLAMA_BIND_ADDR:-0.0.0.0:11434}"
PROXY_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --proxy)
      PROXY_URL="$2"
      shift 2
      ;;
    *)
      echo "Argumento no reconocido: $1" >&2
      exit 1
      ;;
  esac
done

if [[ "${EUID}" -ne 0 ]]; then
  echo "Este script necesita privilegios de administrador. Ejecuta con: sudo $0" >&2
  exit 1
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "No se encontró el comando 'ollama'. Instala primero con install_ollama_ubuntu.sh." >&2
  exit 1
fi

echo "==> Preparando directorio de modelos compartido: ${MODELS_DIR}"
mkdir -p "${MODELS_DIR}"

echo "==> Escribiendo /etc/systemd/system/ollama.service.d/override.conf"
mkdir -p /etc/systemd/system/ollama.service.d
{
  echo "[Service]"
  echo "Environment=\"OLLAMA_HOST=${BIND_ADDR}\""
  echo "Environment=\"OLLAMA_MODELS=${MODELS_DIR}\""
  if [[ -n "${PROXY_URL}" ]]; then
    echo "Environment=\"HTTP_PROXY=${PROXY_URL}\""
    echo "Environment=\"HTTPS_PROXY=${PROXY_URL}\""
  fi
} > /etc/systemd/system/ollama.service.d/override.conf

echo "==> Aplicando cambios"
systemctl daemon-reload
systemctl restart ollama
sleep 2

if systemctl is-active --quiet ollama; then
  echo "Listo. Ollama ahora escucha en ${BIND_ADDR} con modelos en ${MODELS_DIR}."
  [[ -n "${PROXY_URL}" ]] && echo "Proxy institucional configurado: ${PROXY_URL}"
else
  echo "El servicio no arrancó correctamente tras el cambio. Revierte con:" >&2
  echo "  sudo rm /etc/systemd/system/ollama.service.d/override.conf" >&2
  echo "  sudo systemctl daemon-reload && sudo systemctl restart ollama" >&2
  echo "y consulta guia_solucion_problemas.md." >&2
  exit 1
fi

#!/usr/bin/env bash
# install_ollama_ubuntu.sh
# ---------------------------------------------------------------------------
# Instala Ollama en un servidor Linux (Ubuntu/Debian) y lo deja listo para un
# entorno educativo multiusuario: acceso desde la red del laboratorio,
# directorio de modelos compartido y arranque automático con systemd.
#
# Uso:
#   chmod +x install_ollama_ubuntu.sh
#   sudo ./install_ollama_ubuntu.sh
#
# Complementa la sección "Guías de Instalación por Plataforma" del Capítulo 5
# del libro. No sustituye la lectura de esa sección: automatiza los pasos que
# ahí se describen manualmente.
# ---------------------------------------------------------------------------
set -euo pipefail

MODELS_DIR="${OLLAMA_MODELS_DIR:-/shared/models}"
BIND_ADDR="${OLLAMA_BIND_ADDR:-0.0.0.0:11434}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Este script necesita privilegios de administrador. Ejecuta con: sudo $0" >&2
  exit 1
fi

echo "==> Paso 1/5: instalando Ollama mediante el script oficial"
curl -fsSL https://ollama.ai/install.sh | sh

echo "==> Paso 2/5: creando el directorio compartido de modelos (${MODELS_DIR})"
mkdir -p "${MODELS_DIR}"
chown -R ollama:ollama "${MODELS_DIR}" 2>/dev/null || true
chmod 775 "${MODELS_DIR}"

echo "==> Paso 3/5: configurando systemd para acceso multiusuario"
mkdir -p /etc/systemd/system/ollama.service.d
cat > /etc/systemd/system/ollama.service.d/override.conf <<EOF
[Service]
Environment="OLLAMA_HOST=${BIND_ADDR}"
Environment="OLLAMA_MODELS=${MODELS_DIR}"
EOF

echo "==> Paso 4/5: recargando systemd y habilitando el servicio"
systemctl daemon-reload
systemctl enable ollama
systemctl restart ollama

echo "==> Paso 5/5: verificación"
sleep 2
if systemctl is-active --quiet ollama; then
  echo "Ollama está activo y escuchando en ${BIND_ADDR}"
  echo "Prueba desde otra máquina del laboratorio con:"
  echo "  curl http://$(hostname -I | awk '{print $1}'):11434/api/version"
else
  echo "El servicio no quedó activo. Revisa 'systemctl status ollama' y" >&2
  echo "'journalctl -u ollama -n 50' para diagnosticar. Ver también" >&2
  echo "guia_solucion_problemas.md en esta misma carpeta." >&2
  exit 1
fi

cat <<'NOTE'

Recordatorios importantes (ver Capítulo 5, sección "Configuración avanzada
para entornos de múltiples usuarios"):
  * Si tu institución usa un proxy corporativo, añade las variables
    HTTP_PROXY/HTTPS_PROXY al mismo archivo override.conf ANTES de reiniciar
    el servicio (ver plantillas/proxy-institucional.env en esta carpeta).
  * El puerto 11434 debe estar abierto en el firewall institucional. Este
    script NO modifica reglas de firewall: coordina esa apertura con el
    departamento de TI de tu institución.
  * Exponer OLLAMA_HOST en 0.0.0.0 hace que el servidor sea accesible desde
    cualquier equipo que conozca la IP del servidor y el puerto 11434, sin
    autenticación adicional. Restringe el acceso por rango de IP del campus
    a nivel de firewall si tu política institucional lo requiere.
NOTE

# Guía de solución de problemas — LLMs locales (Capítulo 5)

Esta guía cubre los problemas más frecuentes al instalar y operar Ollama, LM
Studio, OpenWebUI y el stack de monitoreo descritos en el Capítulo 5 del
libro, en un entorno educativo (laboratorio de cómputo, servidor
departamental o equipo personal).

Antes de reportar un problema como "no funciona", verifica siempre estos
tres puntos: (1) el servicio está corriendo, (2) el puerto correcto está
abierto y accesible, (3) no hay un proxy o firewall institucional bloqueando
la conexión. La mayoría de los problemas de esta lista caen en uno de esos
tres puntos.

## 1. Ollama

### 1.1 "Connection refused" al conectar a `localhost:11434`

- Verifica que el servicio esté activo:
  - Linux: `sudo systemctl status ollama`
  - Windows/macOS: revisa si el ícono de Ollama aparece en la bandeja del
    sistema o en la barra de menús.
- Si el servicio está inactivo en Linux: `sudo systemctl restart ollama` y
  revisa el registro con `journalctl -u ollama -n 50 --no-pager`.
- Confirma que estás usando el puerto correcto: por defecto es `11434`, no
  `11343` ni `1143` (error de tecleo frecuente).

### 1.2 Otras máquinas del laboratorio no pueden conectarse al servidor

Este es el problema más común al montar un servidor Ollama centralizado
(ver `scripts/configure_multiuser_ollama.sh`).

1. Confirma que `OLLAMA_HOST` esté configurado como `0.0.0.0:11434` y no
   como `127.0.0.1:11434` (esta última solo acepta conexiones desde la
   misma máquina). Revisa con:
   ```
   sudo systemctl show ollama --property=Environment
   ```
2. Confirma que el firewall del **servidor** permite conexiones entrantes al
   puerto 11434 desde la subred del laboratorio:
   ```
   sudo ufw allow from 192.168.1.0/24 to any port 11434    # ejemplo con ufw
   ```
   Ajusta el rango de IP a la subred real de tu institución. Coordina esta
   apertura con el departamento de TI si el firewall lo gestionan ellos.
3. Prueba la conexión directamente desde otra máquina, antes de culpar a
   OpenWebUI o cualquier otra herramienta:
   ```
   curl http://IP_DEL_SERVIDOR:11434/api/version
   ```
   Si esto falla, el problema es de red/firewall, no de Ollama ni de la
   aplicación que estés usando encima.

### 1.3 Windows: el firewall de Defender bloquea Ollama

Windows Defender puede pedir confirmación la primera vez que Ollama expone
un puerto. Ve a "Firewall de Windows Defender" → "Permitir una aplicación a
través del firewall" y confirma que `ollama.exe` tenga permiso en redes
privadas (y en redes de dominio, si el laboratorio usa Active Directory).

### 1.4 macOS: "Ollama.app no se puede abrir porque proviene de un
desarrollador no identificado"

Esto ocurre con el instalador DMG en algunas configuraciones de Gatekeeper.
Solución:
```
xattr -d com.apple.quarantine /Applications/Ollama.app
```
Si prefieres evitar este paso, instala mediante Homebrew
(`brew install ollama`) en lugar del DMG: Homebrew no dispara esta
advertencia de la misma manera.

### 1.5 El modelo se descarga pero `ollama run` se cuelga o el equipo se
congela

Casi siempre es falta de RAM/VRAM para el modelo elegido. Antes de asumir
que algo está roto:
1. Compara el modelo contra la Tabla de requisitos de hardware del
   capítulo (`tab:requisitos_hardware`) o usa CYRI AI
   (<https://www.systemrequirementslab.com/cyri-ai>) para una estimación
   automática.
2. Prueba primero con un modelo pequeño (`llama3.2:1b`) para confirmar que
   la instalación en sí funciona, antes de descargar modelos grandes.
3. En Linux, revisa si el proceso murió por falta de memoria:
   `dmesg | grep -i "out of memory"`.

### 1.6 `ollama -v` muestra una versión distinta a la que menciona el libro

Es esperado: Ollama publica nuevas versiones cada pocas semanas (ver la nota
de vigencia al inicio del capítulo). Esto no es un error de instalación.

## 2. LM Studio

### 2.1 LM Studio no detecta la GPU

1. Actualiza los drivers de la GPU (NVIDIA: <https://www.nvidia.com/drivers>;
   AMD: sitio oficial de drivers ROCm/Adrenalin).
2. En la configuración del modelo dentro de LM Studio, revisa el deslizador
   de "GPU Offload": si está en 0, el modelo corre completamente en CPU
   aunque la GPU esté disponible.
3. En equipos con GPU integrada y GPU dedicada (frecuente en laptops), abre
   el panel de control de la GPU dedicada y confirma que LM Studio esté
   asignado a esa GPU y no a la integrada.

### 2.2 El servidor local de LM Studio no responde en `localhost:1234`

Confirma que el servidor esté iniciado desde la pestaña correspondiente de
LM Studio (no se inicia automáticamente al abrir la aplicación). Verifica
el puerto configurado: LM Studio permite cambiarlo, y `1234` es solo el
valor por defecto.

## 3. OpenWebUI (Docker)

### 3.1 OpenWebUI se despliega pero no puede conectar con Ollama

Este es el problema #1 al usar `scripts/setup_openwebui_docker.sh` o la
plantilla `plantillas/docker-compose-openwebui.yml` directamente.

- **Docker Desktop (Windows/macOS)**: usa
  `http://host.docker.internal:11434`. Es el valor por defecto de la
  plantilla.
- **Linux con Docker nativo (no Docker Desktop)**: `host.docker.internal`
  normalmente NO resuelve. Usa la IP real del host, por ejemplo:
  ```
  ./scripts/setup_openwebui_docker.sh http://192.168.1.50:11434
  ```
  o edita `OLLAMA_BASE_URL` directamente en el compose.
- Verifica primero que Ollama responda desde fuera del contenedor
  (`curl http://localhost:11434/api/version` en el host) antes de revisar
  el contenedor.

### 3.2 No se puede crear la cuenta de administrador

La primera cuenta que se registra en una instalación nueva de OpenWebUI se
convierte automáticamente en administradora. Si esa ventana ya se perdió
(por ejemplo, alguien más se registró primero por error), la forma más
simple de reiniciar en un entorno de prueba es borrar el volumen de datos:
```
docker compose -f plantillas/docker-compose-openwebui.yml down -v
```
**Advertencia**: esto borra todas las conversaciones y cuentas existentes.
No lo ejecutes sobre un despliegue con datos reales de estudiantes.

## 4. Monitoreo (Prometheus/Grafana)

### 4.1 Grafana no muestra datos ("No data")

1. Verifica que Prometheus esté realmente scrapeando el exportador:
   abre `http://localhost:9090/targets` y confirma que
   `ollama_health_exporter` y `node_exporter` aparezcan como `UP`.
2. Si `ollama_health_exporter` aparece `DOWN`, revisa que la URL de Ollama
   configurada en `OLLAMA_URL` (variable de entorno del servicio en
   `docker-compose-monitoring.yml`) sea alcanzable desde dentro del
   contenedor, no solo desde el host.
3. En Grafana, confirma que exista una fuente de datos Prometheus apuntando
   a `http://prometheus:9090` (nombre del servicio en la red de Docker, no
   `localhost`, porque Grafana corre en su propio contenedor).

### 4.2 Monitoreo de GPU (VRAM, temperatura)

El stack incluido en `monitoreo/docker-compose-monitoring.yml` monitorea
CPU/RAM del host y el estado de Ollama, pero **no** incluye métricas de GPU
por defecto: eso requiere el exportador oficial de NVIDIA, DCGM Exporter
(<https://github.com/NVIDIA/dcgm-exporter>), que a su vez necesita el
NVIDIA Container Toolkit instalado en el host. La sección comentada en
`monitoreo/prometheus.yml` indica dónde añadirlo si tu institución decide
incorporarlo.

## 5. Cuando ninguna de estas soluciones funciona

1. Reproduce el problema con la configuración más simple posible (Ollama
   solo, sin Docker, sin proxy, sin firewall personalizado) para aislar si
   el problema es de la herramienta base o de tu configuración institucional
   añadida.
2. Revisa la documentación oficial de la herramienta específica (enlaces en
   el `README.md` de esta carpeta).
3. Busca el mensaje de error exacto en el repositorio oficial de la
   herramienta en GitHub (sección Issues/Discussions): es muy probable que
   alguien más ya lo haya reportado.

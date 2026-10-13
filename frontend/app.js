// Bio-Edge AMD • Frontend Logic & Reactive Dashboard
// Desarrollado por Elida (Titular) & Mario (Backup)

let socket = null;
let reconnectTimer = null;
const HOST = window.location.hostname || "127.0.0.1";
const PORT = window.location.port || "8000";
const WS_URL = `ws://${HOST}:${PORT}/ws/stream`;

// Elementos del DOM
const streamCanvas = document.getElementById("streamCanvas");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const statusPill = document.getElementById("statusPill");
const statusText = document.getElementById("statusText");
const devicePill = document.getElementById("devicePill");
const hudFps = document.getElementById("hudFps");
const hudLat = document.getElementById("hudLat");
const hudDevice = document.getElementById("hudDevice");
const speciesGrid = document.getElementById("speciesGrid");
const btnSwitchAmd = document.getElementById("btnSwitchAmd");
const btnSwitchCpu = document.getElementById("btnSwitchCpu");
const btnRunBenchmark = document.getElementById("btnRunBenchmark");
const speedupVal = document.getElementById("speedupVal");
const latencyRedVal = document.getElementById("latencyRedVal");
const fpsGainVal = document.getElementById("fpsGainVal");
const telProvider = document.getElementById("telProvider");
const telDevice = document.getElementById("telDevice");
const telCpu = document.getElementById("telCpu");

// Inicializar conexión WebSocket
function connectWebSocket() {
  clearTimeout(reconnectTimer);
  console.log(`[Bio-Edge] Conectando a ${WS_URL}...`);
  statusText.innerText = "CONECTANDO...";
  statusPill.style.borderColor = "#ffb300";

  try {
    socket = new WebSocket(WS_URL);
  } catch (err) {
    console.warn("Error creando WebSocket:", err);
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    console.log("[Bio-Edge] WebSocket conectado.");
    statusText.innerText = "CONECTADO EN BORDE";
    statusPill.style.borderColor = "var(--green-neon)";
    videoPlaceholder.style.display = "none";
  };

  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "frame") {
        streamCanvas.src = data.image;
        if (data.telemetry) {
          updateTelemetry(data.telemetry);
        }
      }
    } catch (e) {
      console.error("Error procesando paquete:", e);
    }
  };

  socket.onerror = (err) => {
    console.warn("[Bio-Edge] Error en WebSocket:", err);
  };

  socket.onclose = () => {
    console.warn("[Bio-Edge] Conexión cerrada. Reconectando...");
    statusText.innerText = "RECONECTANDO...";
    statusPill.style.borderColor = "#ed1c24";
    scheduleReconnect();
  };
}

function scheduleReconnect() {
  reconnectTimer = setTimeout(connectWebSocket, 2000);
}

// Actualizar HUD y paneles de telemetría
function updateTelemetry(tel) {
  if (tel.fps !== undefined) hudFps.innerText = tel.fps.toFixed(1);
  if (tel.latency_ms !== undefined) hudLat.innerText = `${tel.latency_ms.toFixed(1)} ms`;

  if (tel.hardware) {
    const prov = tel.hardware.provider || "Desconocido";
    hudDevice.innerText = prov;
    telProvider.innerText = prov;
    telDevice.innerText = tel.hardware.device || "AMD Silicio";
    devicePill.innerText = tel.hardware.is_accelerated ? "⚡ AMD Acelerado" : "🐢 CPU Fallback";

    if (tel.hardware.is_accelerated) {
      btnSwitchAmd.classList.add("active");
      btnSwitchCpu.classList.remove("active");
    } else {
      btnSwitchCpu.classList.add("active");
      btnSwitchAmd.classList.remove("active");
    }
  }

  // Actualizar resumen de especies si viene en la telemetría
  if (tel.summary_counts && tel.summary_counts.counts) {
    renderSpeciesGrid(tel.summary_counts.counts);
  }
}

function renderSpeciesGrid(counts) {
  const speciesIcons = {
    jaguar: "🐆",
    venado: "🦌",
    tapir: "🐗",
    mono_arana: "🐒",
    ocelote: "🐆",
    humano_intruso: "🚨",
    vehiculo_furtivo: "🚙"
  };

  let html = "";
  for (const [sp, data] of Object.entries(counts)) {
    const isThreat = sp.includes("intruso") || sp.includes("vehiculo");
    const icon = speciesIcons[sp] || "🐾";
    const inCount = data.ENTRANDO || 0;
    const outCount = data.SALIENDO || 0;

    html += `
      <div class="species-item ${isThreat ? "alert-threat" : ""}">
        <div class="species-name">${icon} ${sp.toUpperCase().replace("_", " ")}</div>
        <div class="species-stats">
          <span class="stat-in">In: <strong>${inCount}</strong></span>
          <span class="stat-out">Out: <strong>${outCount}</strong></span>
        </div>
      </div>
    `;
  }

  if (html.trim()) {
    speciesGrid.innerHTML = html;
  }
}

// Alternar dispositivo (Demo para el jurado de AMD)
async function switchDevice(device) {
  try {
    const res = await fetch(`http://${HOST}:${PORT}/api/config/device`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ device })
    });
    const result = await res.json();
    console.log("Dispositivo cambiado:", result);
  } catch (err) {
    console.warn("No se pudo contactar endpoint de cambio de dispositivo:", err);
  }
}

btnSwitchAmd.addEventListener("click", () => switchDevice("directml"));
btnSwitchCpu.addEventListener("click", () => switchDevice("cpu"));

// Ejecutar Benchmark bajo demanda
btnRunBenchmark.addEventListener("click", async () => {
  btnRunBenchmark.innerText = "⏳ Evaluando...";
  btnRunBenchmark.disabled = true;
  try {
    const res = await fetch(`http://${HOST}:${PORT}/api/benchmarks/compare?runs=20`);
    const data = await res.json();
    if (data.comparison) {
      speedupVal.innerText = `${data.comparison.speedup_factor}x`;
      latencyRedVal.innerText = `-${data.comparison.latency_reduction_pct}%`;
      fpsGainVal.innerText = `+${data.comparison.fps_improvement_pct}% FPS`;
    }
  } catch (err) {
    console.warn("Error ejecutando benchmark:", err);
  } finally {
    btnRunBenchmark.innerText = "🔄 Ejecutar Test";
    btnRunBenchmark.disabled = false;
  }
});

// Arrancar conexión al cargar la página
window.addEventListener("DOMContentLoaded", () => {
  connectWebSocket();
});

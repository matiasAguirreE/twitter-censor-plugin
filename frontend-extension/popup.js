/**
 * popup.js
 * 
 * Script principal para el popup de la extensión.
 * Permite al usuario activar/desactivar el filtro y ajustar los umbrales de censura
 * para distintas categorías (Violencia, Homofobia, Xenofobia).
 * Sincroniza la configuración con el almacenamiento local de Chrome y notifica
 * a la pestaña activa sobre los cambios.
 */

// Referencias a los elementos del DOM del popup
const toggle = document.getElementById("toggle"); // Checkbox para activar/desactivar el filtro
const toggleLabel = document.getElementById("toggle-label"); // Etiqueta que muestra el estado

// Referencias a los sliders de umbral para cada categoría
const ranges = {
  Violencia: document.getElementById("violenciaRange"),
  Homofobia: document.getElementById("homofobiaRange"),
  Xenofobia: document.getElementById("xenofobiaRange")
};

/**
 * Guarda la configuración en el almacenamiento local y notifica a la pestaña activa.
 * @param {Object} config - Configuración a guardar (enabled, thresholds)
 */
function guardarConfiguracion(config) {
  chrome.storage.local.set(config);
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    chrome.tabs.sendMessage(tabs[0].id, {
      type: "updateConfig",
      config
    });
  });
}

// Al abrir el popup, carga la configuración guardada y actualiza la UI
chrome.storage.local.get(["enabled", "thresholds"], (data) => {
  toggle.checked = data.enabled ?? true;
  toggleLabel.textContent = toggle.checked ? "Activado" : "Desactivado";

  const thresholds = data.thresholds || {};
  for (const cat in ranges) {
    ranges[cat].value = thresholds[cat] ?? 50;
  }
});

// Evento: cambio en el checkbox de activación
toggle.addEventListener("change", () => {
  toggleLabel.textContent = toggle.checked ? "Activado" : "Desactivado";
  actualizarConfig();
});

// Evento: cambio en los sliders de umbral
for (const cat in ranges) {
  ranges[cat].addEventListener("input", actualizarConfig);
}

/**
 * Lee los valores actuales de la UI y guarda la configuración.
 */
function actualizarConfig() {
  const thresholds = {};
  for (const cat in ranges) {
    thresholds[cat] = parseInt(ranges[cat].value);
  }
  guardarConfiguracion({
    enabled: toggle.checked,
    thresholds
  });
}

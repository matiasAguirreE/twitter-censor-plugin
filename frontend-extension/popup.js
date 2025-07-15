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

// Referencias a los sliders y sus outputs para cada categoría
const categories = ['Violencia', 'Homofobia', 'Xenofobia'];
const ranges = {
  Violencia: document.getElementById("violenciaRange"),
  Homofobia: document.getElementById("homofobiaRange"),
  Xenofobia: document.getElementById("xenofobiaRange")
};
const outputs = {
  Violencia: document.getElementById("violenciaOutput"),
  Homofobia: document.getElementById("homofobiaOutput"),
  Xenofobia: document.getElementById("xenofobiaOutput")
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

/**
 * Actualiza el texto del <output> asociado según el valor normal.
 */
function actualizarOutput(cat, value) {
  outputs[cat].textContent = value + "%";
}

/**
 * Lee los valores actuales de la UI y guarda la configuración.
 * Convierte el valor de cada slider a su valor invertido (100 - slider.value)
 */
function actualizarConfig() {
  const thresholds = {};
  for (const cat of categories) {
    const sliderVal = parseInt(ranges[cat].value);
    thresholds[cat] = 100 - sliderVal; // invertimos el valor
  }
  guardarConfiguracion({
    enabled: toggle.checked,
    thresholds
  });
}

// Al abrir el popup, carga la configuración guardada y actualiza la UI
chrome.storage.local.get(["enabled", "thresholds"], (data) => {
  // Si no hay configuración, se usa true por defecto.
  toggle.checked = data.enabled ?? true;
  toggleLabel.textContent = toggle.checked ? "Activado" : "Desactivado";

  const thresholds = data.thresholds || {};
  // Para cada categoría se carga el slider. Dado que el valor almacenado es invertido,
  // se asigna el valor del slider como (100 - threshold almacenado), o 50 si no existe.
  for (const cat of categories) {
    const storedThreshold = thresholds[cat] ?? 50;
    const sliderVal = 100 - storedThreshold;
    ranges[cat].value = sliderVal;
    actualizarOutput(cat, sliderVal);
  }
});

// Evento: cambio en el checkbox de activación
toggle.addEventListener("change", () => {
  toggleLabel.textContent = toggle.checked ? "Activado" : "Desactivado";
  actualizarConfig();
});

// Evento: cambio en los sliders de umbral
for (const cat of categories) {
  ranges[cat].addEventListener("input", (e) => {
    // Actualizamos el output en tiempo real cuando se mueve el slider
    actualizarOutput(cat, e.target.value);
    actualizarConfig();
  });
}

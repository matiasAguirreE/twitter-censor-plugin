document.addEventListener('DOMContentLoaded', function() {
  const statusButton = document.getElementById('statusButton');
  let isWorking = true;

  // Cargar estados guardados
  chrome.storage.local.get(['pluginStatus', 'optionStates'], function(result) {
    // Estado del plugin
    isWorking = result.pluginStatus !== undefined ? result.pluginStatus : true;
    updateStatus();
    
    // Estados de las opciones
    if (result.optionStates) {
      document.querySelectorAll('.option-container').forEach((container, index) => {
        if (result.optionStates[index]) container.classList.add('active');
      });
    }
  });

  function updateStatus() {
    statusButton.textContent = isWorking ? "🟢 FUNCIONANDO" : "🔴 NO FUNCIONANDO";
    statusButton.classList.toggle('no-working', !isWorking);
    chrome.storage.local.set({ pluginStatus: isWorking });
  }

  // Toggle estado del plugin
  statusButton.addEventListener('click', function() {
    isWorking = !isWorking;
    updateStatus();
  });

  // Toggle opciones
  document.querySelectorAll('.option-container').forEach((container, index) => {
    container.addEventListener('click', function() {
      this.classList.toggle('active');
      
      // Guardar estado de todas las opciones
      const states = {};
      document.querySelectorAll('.option-container').forEach((c, i) => {
        states[i] = c.classList.contains('active');
      });
      chrome.storage.local.set({ optionStates: states });
    });
  });

  document.getElementById('verificarBtn').addEventListener('click', () => {
    const texto = "Este es el texto que quiero verificar";
    verificarCensura(texto);
  });
});

// consulta al servidor si es que el texto es censurable
function verificarCensura(texto) {
  fetch('http://gate.dcc.uchile.cl:8638/verificarCensura/', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Api-Key': 'e55d7f49-a705-4895-bf5f-d63aa1f46e11'
    },
    body: JSON.stringify({ texto: texto })
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(`Error en la solicitud: ${response.status}`);
    }
    return response.json();
  })
  .then(data => {
    console.log('Respuesta del servidor:', data);
  })
  .catch(error => {
    console.error('Error al enviar datos:', error);
  });
}
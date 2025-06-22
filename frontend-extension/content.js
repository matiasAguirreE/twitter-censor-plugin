// Objeto donde almacenamos las clases ya creadas, usando el hash como clave
const clasesRegistradas = {};

/**
 * Extrae el texto de un tweet.
 * @param {HTMLElement} tweet - Elemento del tweet.
 * @returns {string} Texto concatenado del tweet.
 */
function extractText(tweet) {
    const spans = Array.from(tweet.querySelectorAll('div[data-testid="tweetText"] span'));
    return spans.map(span => span.textContent).join(' ');
}

// Configuración actual de la extensión (por defecto)
let configuracionActual = {
    enabled: true,
    thresholds: {
        Violencia: 50,
        Homofobia: 50,
        Xenofobia: 50
    }
};

/**
 * Carga la configuración guardada en chrome.storage.sync (si existe).
 */
chrome.storage.sync.get(["config"], (result) => {
  if (result.config) {
    configuracionActual = result.config;
    console.log("Configuración cargada desde storage:", configuracionActual);
  }
});

/**
 * Escucha mensajes desde el background o popup para actualizar la configuración.
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === "updateConfig") {
        configuracionActual = message.config;
        console.log("Nueva configuración recibida:", configuracionActual);
        // Reiniciar el caché de tweets
        for (const key in clasesRegistradas) delete clasesRegistradas[key];
        // Quitar el blur de todos los tweets ya procesados
        const tweets = document.querySelectorAll('article[data-testid="tweet"]');
        for (const tweet of tweets) {
            tweet.style.filter = "none";
            tweet.removeAttribute("data-desBlur");

            // Eliminar el contenedor de la razón si existe
            const next = tweet.nextElementSibling;
            if (next?.classList.contains("blur-reason-box")) {
                next.remove();
            }
        }
        // volver a procesar tweets
        procesarTweets();
    }
});

/**
 * Al cargar el content script, obtiene la configuración local y, si está activado,
 * inicia el observer y procesa los tweets.
 */
chrome.storage.local.get(["enabled", "thresholds"], (data) => {
    if (data.enabled !== undefined) configuracionActual.enabled = data.enabled;
    if (data.thresholds) configuracionActual.thresholds = data.thresholds;
    if (configuracionActual.enabled) {
        iniciarObserver();
        procesarTweets();
    }
});

/**
 * Inicia un MutationObserver para detectar nuevos tweets en el timeline.
 */
function iniciarObserver() {
    const observer = new MutationObserver(() => procesarTweets());
    observer.observe(document.body, {
        childList: true,
        subtree: true,
    });
}

/**
 * Clase que representa un Tweet y sus operaciones de censura.
 */
class Tweet {
    /**
     * @param {HTMLElement} articleTweet - Elemento del tweet.
     */
    constructor(articleTweet) {
        this.articleTweet = articleTweet;
        this.text = this.extractText();
        this.parentTweet = this.findParentTweet();
        this.serverData = {};
    }

    /**
     * Extrae el texto del tweet.
     * @returns {string}
     */
    extractText() {
        return extractText(this.articleTweet);
    }

    /**
     * Busca el tweet padre en una conversación (si existe).
     * @returns {Tweet|null}
     */
    findParentTweet() {
        const parent = this.articleTweet.closest('div[aria-label="Cronología: Conversación"]')?.querySelector('article');
        if (parent && parent !== this.articleTweet) {
            return new Tweet(parent); // recursivo si quieres blur también al padre
        }
        return null;
    }

    /**
     * Consulta al servidor si el texto es censurable según las categorías.
     * Actualiza la propiedad blur si corresponde.
     * @returns {Promise<Object>} Respuesta del servidor.
     */
    async verificarCensura() {
        try {
            const response = await fetch('https://grupo8.juan.cl/verificarCensura/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Api-Key': 'e55d7f49-a705-4895-bf5f-d63aa1f46e11'
                },
                body: JSON.stringify({ texto: this.text })
            });
            if (!response.ok) {
                throw new Error(`Error en la solicitud: ${response.status}`);
            }
            const data = await response.json();
            console.log('Respuesta del servidor:', data, this.text);

            if (!configuracionActual.enabled) return;

            for (const categoria in data) {
                const probabilidad = data[categoria] * 100;
                const umbral = 100 - configuracionActual.thresholds[categoria];
                console.log(`Probabilidad de ${categoria}: ${probabilidad}% (umbral: ${umbral}%)`);
                this.serverData[categoria] = probabilidad;
            }

            return data;
        } catch (error) {
            console.error('Error al enviar datos:', error);
            throw new Error(`Error al enviar datos: ${error}`);
        }
    }

    /**
     * Aplica el efecto de desenfoque al tweet.
     */
    applyBlur(tweet) {
        if (tweet.dataset.desBlur === "true") {
            return;
        }

        let categoriasCensurables = {};

        Object.keys(this.serverData).forEach(categoria => {
            if (this.serverData[categoria] >= configuracionActual['thresholds'][categoria]) {
                categoriasCensurables[categoria] = this.serverData[categoria];
            }
        });

        if (Object.keys(categoriasCensurables).length != 0) {
            tweet.style.filter = "blur(6px)";
            tweet.style.transition = "filter 0.3s ease";

            // Evita duplicar el mensaje si ya fue añadido
            // TODO: ver como obtener el length de categorias censurables (en caso de ser distintos ahi que actualizar)
            if (tweet.nextElementSibling?.classList.contains("blur-reason-box")) return;

            // Crear contenedor del mensaje
            const reasonBox = document.createElement("div");
            reasonBox.className = "blur-reason-box";

            // Estilo visual compatible con Twitter
            reasonBox.style.marginTop = "8px";
            reasonBox.style.padding = "10px 15px";
            reasonBox.style.border = "1px solid #e1e8ed";
            reasonBox.style.borderRadius = "12px";
            reasonBox.style.backgroundColor = "#f7f9f9";
            reasonBox.style.color = "#0f1419";
            reasonBox.style.fontSize = "13px";
            reasonBox.style.fontFamily = "'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
            reasonBox.style.lineHeight = "1.4";
            reasonBox.style.textAlign = "center";

            // Texto con la razón
            let texto = "Este tweet fue ocultado por:<br>";

            Object.keys(categoriasCensurables).forEach(categoria => {
                texto += `<strong>${categoria}</strong> con probabilidad ${categoriasCensurables[categoria].toFixed(2)}%<br>`;
            });
            reasonBox.innerHTML = texto;

            // Crear botón para desblur
            const btnDesblur = document.createElement("button");
            btnDesblur.textContent = "Mostrar contenido";
            btnDesblur.style.marginTop = "10px";
            btnDesblur.style.padding = "5px 10px";
            btnDesblur.style.border = "none";
            btnDesblur.style.borderRadius = "8px";
            btnDesblur.style.backgroundColor = "#1DA1F2";
            btnDesblur.style.color = "white";
            btnDesblur.style.cursor = "pointer";
            btnDesblur.style.fontSize = "13px";

            // Evento click para quitar blur
            btnDesblur.addEventListener("click", () => {
                const isBlured = tweet.style.filter === "blur(6px)";
                if (isBlured) {
                    tweet.dataset.desBlur = "true";
                    tweet.style.filter = "none";
                    btnDesblur.style.backgroundColor = "#888";
                    btnDesblur.textContent = "Ocultar contenido";
                } else {
                    tweet.dataset.desBlur = "false";
                    tweet.style.filter = "blur(6px)";
                    btnDesblur.style.backgroundColor = "#1DA1F2";
                    btnDesblur.textContent = "Mostrar contenido";
                }
            });

            // Agregar botón al contenedor
            reasonBox.appendChild(btnDesblur);

            // Insertar el contenedor después del tweet
            tweet.insertAdjacentElement("afterend", reasonBox);
        }
    }
}

/**
 * Genera un hash SHA-256 a partir de un texto.
 * @param {string} text - Texto a hashear.
 * @returns {Promise<string>} Hash hexadecimal.
 */
function hashText(text) {
    // Usamos el algoritmo SHA-256 para generar el hash del texto
    return crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
        .then(buffer => {
            let hashArray = Array.from(new Uint8Array(buffer)); // convertir ArrayBuffer a Array de bytes
            let hashHex = hashArray.map(byte => byte.toString(16).padStart(2, '0')).join(''); // convertir bytes a hexadecimal
            return hashHex;
        });
}

/**
 * Obtiene una instancia de Tweet para un tweet dado, usando un hash como clave.
 * Si no existe, la crea.
 * @param {HTMLElement} tweet - Elemento del tweet.
 * @param {string} texto - Texto del tweet.
 * @returns {Promise<{tweet: Tweet, create: boolean}>}
 */
async function getOrCreateClass(tweet, texto) {
    // Obtenemos el hash del texto
    const classNameHash = await hashText(texto);

    // Si la clase existe la obtenemos, de caso contrario la creamos
    if (clasesRegistradas[classNameHash]) {
        return {tweet: clasesRegistradas[classNameHash], create: false};
    } else {
        clasesRegistradas[classNameHash] = new Tweet(tweet);
    }
    return {tweet: clasesRegistradas[classNameHash], create: true};
}

/**
 * Procesa todos los tweets nuevos en la página, verifica si deben ser censurados
 * y aplica el desenfoque si corresponde.
 */
async function procesarTweets() {
    const tweets = document.querySelectorAll('article[data-testid="tweet"]');

    for (const tweet of tweets) {
        const t = await getOrCreateClass(tweet, extractText(tweet));
        if (t.create) {
            await t.tweet.verificarCensura();
        }
        t.tweet.applyBlur(tweet);
    }
}
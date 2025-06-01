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
        this.blur = false;
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
            const response = await fetch('https://withered-glitter-3f84.ignacio-alvbah.workers.dev/verificarCensura/', {
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
                if (probabilidad >= umbral) {
                    this.blur = true;
                    break;
                }
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
    applyBlur() {
        this.articleTweet.style.filter = "blur(6px)";
        this.articleTweet.style.transition = "filter 0.3s ease";
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
    const tweets = document.querySelectorAll('article[data-testid="tweet"]:not([data-blur-processed])');

    for (const tweet of tweets) {
        const t = await getOrCreateClass(tweet, extractText(tweet));
        if (t.create) {
            await t.tweet.verificarCensura();
        } 
        if (t.tweet.blur) {
            t.tweet.applyBlur();
        }
    }
}
// Objeto donde almacenamos las clases ya creadas, usando el hash como clave
const clasesRegistradas = {};

function extractText(tweet) {
    const spans = Array.from(tweet.querySelectorAll('div[data-testid="tweetText"] span'));
    return spans.map(span => span.textContent).join(' ');
}

class Tweet {
    constructor(articleTweet) {
        this.articleTweet = articleTweet;
        this.text = this.extractText();
        this.parentTweet = this.findParentTweet();
        this.blur = false;
    }

    extractText() {
        return extractText(this.articleTweet);
    }

    findParentTweet() {
        const parent = this.articleTweet.closest('div[aria-label="Cronología: Conversación"]')?.querySelector('article');
        if (parent && parent !== this.articleTweet) {
            return new Tweet(parent); // recursivo si quieres blur también al padre
        }
        return null;
    }

    // consulta al servidor si es que el texto es censurable
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
            const censura = Math.max(...Object.values(data));
            console.log('Respuesta del servidor:', censura, this.text);
            if (censura >= 0.5) {
                this.blur = true;
            }
            return data.censura;
        } catch (error) {
            console.error('Error al enviar datos:', error);
            throw new Error(`Error al enviar datos: ${error}`);
        }
    }

    applyBlur() {
        this.articleTweet.style.filter = "blur(6px)";
        this.articleTweet.style.transition = "filter 0.3s ease";
    }
}

// Función para generar un hash corto a partir de un texto largo
function hashText(text) {
    // Usamos el algoritmo SHA-256 para generar el hash del texto
    return crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
        .then(buffer => {
            let hashArray = Array.from(new Uint8Array(buffer)); // convertir ArrayBuffer a Array de bytes
            let hashHex = hashArray.map(byte => byte.toString(16).padStart(2, '0')).join(''); // convertir bytes a hexadecimal
            return hashHex;
        });
}

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

const observer = new MutationObserver(() => procesarTweets());
observer.observe(document.body, {
    childList: true,
    subtree: true,
});
procesarTweets(); // primera carga
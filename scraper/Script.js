

const addButtons = (tweet) => {
    if (!tweet.querySelector('.bien-button-container')) {
        // Crear contenedor principal
        const container = document.createElement('div');
        container.className = 'bien-button-container';
        container.style.cssText = `
            order: 999 !important;
            margin-top: 15px !important;
            width: 100% !important;
        `;

        // Botón de imprimir
        const mainButton = document.createElement('button');
        mainButton.className = 'bien-button';
        mainButton.style.cssText = `
            background: #1da1f2 !important;
            color: white !important;
            padding: 12px 24px !important;
            border-radius: 25px !important;
            margin: 8px 0 !important;
            width: 100% !important;
            cursor: pointer !important;
            font-weight: bold !important;
            border: none !important;
            position: relative !important;
            z-index: 9999 !important;
            display: block !important;
        `;
        mainButton.textContent = 'Guardar Tweet';

        // Botones de categoría
        const topicosCensura = ["Violencia", "Discriminación", "Acoso", "Bullying", "Terrorismo"];
        const esSeleccionado=topicosCensura.reduce((acc,topico)=>{
            acc[topico]=false;
            return acc;
        },{});
        topicosCensura.forEach(topico => {
            const boton = document.createElement('button');
            boton.style.cssText = `
                background: #dc3545 !important;
                color: white !important;
                padding: 8px 16px !important;
                border-radius: 20px !important;
                margin: 4px 0 !important;
                width: 100% !important;
                cursor: pointer !important;
                font-weight: bold !important;
                border: none !important;
                transition: background 0.3s !important;
                display: block !important;
            `;
            boton.textContent = topico;
            
            boton.addEventListener('click',() => {
                if(esSeleccionado[boton.textContent]==false){
                boton.style.backgroundColor="green";
                boton.style.color="white";
                }
                else{
                    boton.style.backgroundColor="red";
                    boton.style.color="white";
                }
                esSeleccionado[boton.textContent]=!esSeleccionado[boton.textContent]
                });
            
            container.appendChild(boton);
        });

        // Configurar texto del tweet
        mainButton.addEventListener('click', (e) => {
            e.stopPropagation();
            const spans = Array.from(tweet.querySelectorAll('div[data-testid="tweetText"] span'));
            const textoTweet=spans.map(span => span.textContent).join(' ');
            const categorias=Object.keys(esSeleccionado).filter(topico=>esSeleccionado[topico]).join(', ');

            window.__pyTweetText=`${textoTweet}\n#############\n${categorias}`;
        });

        // Añadir elementos al DOM
        container.appendChild(mainButton);
        tweet.appendChild(container);

        // Forzar diseño
        tweet.style.display = 'flex';
        tweet.style.flexDirection = 'column';
    }
};

// Observador mejorado
const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        const nodes = [...mutation.addedNodes].filter(node => 
            node.nodeType === 1 && 
            (node.matches('article[data-testid="tweet"]') || 
             node.querySelector('article[data-testid="tweet"]'))
        );
        
        nodes.forEach(node => {
            const tweets = node.matches('article[data-testid="tweet"]') 
                ? [node] 
                : node.querySelectorAll('article[data-testid="tweet"]');
            
            tweets.forEach(tweet => {
                if (!tweet.dataset.bienProcessed) {
                    tweet.dataset.bienProcessed = 'true';
                    const interval = setInterval(() => {
                        if (tweet.querySelector('div[data-testid="tweetText"]')) {
                            clearInterval(interval);
                            addButtons(tweet);
                        }
                    }, 100);
                }
            });
        });
    });
});

observer.observe(document.body, {
    childList: true,
    subtree: true
});

// Procesar tweets iniciales
document.querySelectorAll('article[data-testid="tweet"]').forEach(tweet => {
    if (!tweet.dataset.bienProcessed) {
        tweet.dataset.bienProcessed = 'true';
        const interval = setInterval(() => {
            if (tweet.querySelector('div[data-testid="tweetText"]')) {
                clearInterval(interval);
                addButtons(tweet);
            }
        }, 100);
    }
});
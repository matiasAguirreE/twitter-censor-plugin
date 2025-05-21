class Tweet{
    constructor(articleTweet){
        this.articleTweet=articleTweet;
        this.divBotones=document.createElement('div');
        this.categoriasBotones=["Violencia", "Homofobia", "Xenofobia"];
        this.estadoBotones=this.categoriasBotones.reduce((acc,topico)=>{
            acc[topico]=false;
            return acc;
        },{});
        this.articleTweetPadre=null;
        this.isBlurred = false;
    }
    establecerTweetPadre(tweetPadre){
        this.articleTweetPadre=tweetPadre;
    }
    modificarDivBotones(){
        this.divBotones.style.cssText = `
            margin-top: 12px !important;
            width: 100% !important;
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 8px !important;
            position: relative !important;
            z-index: 1 !important;
        `;

        // Agregar botón de blur
        const blurBoton = document.createElement('button');
        blurBoton.style.cssText = `
            background: #6c757d !important;
            color: white !important;
            padding: 8px 16px !important;
            border-radius: 20px !important;
            margin: 4px 0 !important;
            width: 100% !important;
            cursor: pointer !important;
            font-weight: bold !important;
            border: none !important;
            flex-shrink: 0 !important;
        `;
        blurBoton.textContent = 'Blur';
        blurBoton.id = "blurBoton";
        this.divBotones.appendChild(blurBoton);

        // Agregar botón de guardar
        const guardarBoton = document.createElement('button');
        guardarBoton.style.cssText = `
            background: #1da1f2 !important;
            color: white !important;
            padding: 8px 16px !important;
            border-radius: 20px !important;
            margin: 4px 0 !important;
            width: 100% !important;
            cursor: pointer !important;
            font-weight: bold !important;
            border: none !important;
            flex-shrink: 0 !important;
        `;
        guardarBoton.textContent='Guardar Tweet';
        guardarBoton.id="guardarBoton"

        this.categoriasBotones.forEach(topico => {
            const boton = document.createElement('button');
            boton.style.cssText = `
                background: #dc3545 !important;
                color: white !important;
                padding: 6px 12px !important;
                border-radius: 15px !important;
                margin: 2px !important;
                cursor: pointer !important;
                font-weight: bold !important;
                border: none !important;
                transition: background 0.3s !important;
                flex: 1 1 auto !important;
                white-space: nowrap !important;
            `;
            boton.textContent = topico;
            
            boton.addEventListener('click',() => {
                if(this.estadoBotones[boton.textContent]==false){
                    boton.style.backgroundColor="green";
                    boton.style.color="white";
                }
                else{
                    boton.style.backgroundColor="red";
                    boton.style.color="white";
                }
                this.estadoBotones[boton.textContent]=!this.estadoBotones[boton.textContent]
            });
            
            this.divBotones.appendChild(boton);
        });
        this.divBotones.append(guardarBoton);
    }
    colocarBotonesTweet(){
        const tweetContainer = this.articleTweet.querySelector(':scope > div > div');
        
        if (tweetContainer) {
            const botonesWrapper = document.createElement('div');
            botonesWrapper.style.cssText = `
                padding: 8px 12px !important;
                margin-top: 8px !important;
                border-top: 1px solid #2f3336 !important;
            `;
            
            botonesWrapper.appendChild(this.divBotones);
            tweetContainer.appendChild(botonesWrapper);
        } else {
            this.articleTweet.appendChild(this.divBotones);
        }
    }
    entregarTextoTweet(){
        const spans = Array.from(this.articleTweet.querySelectorAll('div[data-testid="tweetText"] span'));
        const textoTweet=spans.map(span => span.textContent).join(' ');
        return textoTweet;
    }
    entregarTextoTweetPadre(){
        let textoTweetPadre="SinTweetPadre";
        if (this.articleTweetPadre===null){
            return textoTweetPadre;
        }
        else{
            const spansTweetPadre=Array.from(this.articleTweetPadre.querySelectorAll('div[data-testid="tweetText"] span'));
            textoTweetPadre=spansTweetPadre.map(span=>span.textContent).join(' ');
            return textoTweetPadre;
        }
    }
    toggleBlur() {
        const tweetContent = this.articleTweet;
        
        // Blur all text blocks
        const tweetTexts = tweetContent.querySelectorAll('div[data-testid="tweetText"]');
        tweetTexts.forEach(tweetText => {
            if (!this.isBlurred) {
                tweetText.style.filter = 'blur(8px)';
                tweetText.style.transition = 'filter 0.3s ease';
            } else {
                tweetText.style.filter = 'none';
            }
        });

        // Blur media containers
        const mediaContainers = [
            'div[data-testid="tweetPhoto"]',
            'div[data-testid="card.wrapper"]',
            'div[data-testid="videoPlayer"]',
            'div[data-testid="card.layoutLarge.media"]',
            'div[data-testid="card.layoutSmall.media"]',
            'div[data-testid="card.layoutProminent.media"]',
            'div[role="group"]',
            'div[data-testid="tweetPhotoGrid"]'
        ];

        mediaContainers.forEach(selector => {
            const containers = tweetContent.querySelectorAll(selector);
            containers.forEach(container => {
                if (!this.isBlurred) {
                    container.style.filter = 'blur(8px)';
                    container.style.transition = 'filter 0.3s ease';
                    const allElements = container.getElementsByTagName('*');
                    for (let element of allElements) {
                        element.style.filter = 'blur(8px)';
                        element.style.transition = 'filter 0.3s ease';
                    }
                } else {
                    container.style.filter = 'none';
                    const allElements = container.getElementsByTagName('*');
                    for (let element of allElements) {
                        element.style.filter = 'none';
                    }
                }
            });
        });

        // Blur all videos
        const videos = tweetContent.querySelectorAll('video');
        videos.forEach(video => {
            if (!this.isBlurred) {
                video.style.filter = 'blur(8px)';
                video.style.transition = 'filter 0.3s ease';
            } else {
                video.style.filter = 'none';
            }
        });

        // Update button text
        const blurBoton = this.divBotones.querySelector("#blurBoton");
        if (blurBoton) {
            blurBoton.textContent = this.isBlurred ? 'Blur' : 'Unblur';
        }

        this.isBlurred = !this.isBlurred;
    }
    establecerGuardado(){
        const guardarBoton = this.divBotones.querySelector("#guardarBoton");
        const blurBoton = this.divBotones.querySelector("#blurBoton");

        blurBoton.addEventListener('click', () => {
            this.toggleBlur();
        });

        guardarBoton.addEventListener('click',(e)=>{
            const categorias=Object.keys(this.estadoBotones).filter(topico=>this.estadoBotones[topico]).join(', ');
            const textoTweet=this.entregarTextoTweet();
            const textoTweetPadre=this.entregarTextoTweetPadre();
            window.__pyTweetText=`${textoTweet}#############${categorias}#############${textoTweetPadre}`;
        });
    }
}

const main=document.querySelector('main');
const divCronologia= main.querySelector('div[aria-label="Cronología: Conversación"]');
const tweetsComentarios=[]
if(divCronologia===null){
    const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
        const main = document.querySelector('main');
        const divCronologia = main?.querySelector('div[aria-label="Cronología: Conversación"]');

        // 1. Verificar en CADA mutación si divCronologia existe
        if (divCronologia) {
            const tweetsCargados = document.querySelectorAll('article[data-testid="tweet"]:not([data-bien-processed])');
            
            tweetsCargados.forEach((tweet, index) => {
                tweet.dataset.bienProcessed = 'true';
                const divContenedor = tweet.closest('div');
                const estaPostear = divContenedor?.querySelector('div[data-testid="inline_reply_offscreen"]');
                const nuevoTweet = new Tweet(tweet);
                
                // Lógica mejorada de asignación de padres
                if (tweetsComentarios.length === 0) {
                    // Primer tweet de la conversación
                    nuevoTweet.establecerTweetPadre(null);
                } else if (estaPostear) {
                    // Tweet padre de comentarios, padre = último tweet del historial
                    nuevoTweet.establecerTweetPadre(tweetsComentarios[tweetsComentarios.length - 1].articleTweet);
                } else {
                    // Tweet hijo, padre = último tweet padre registrado
                    const padre = tweetsComentarios.findLast(t => t.articleTweet.dataset.esPadre);
                    nuevoTweet.establecerTweetPadre(padre?.articleTweet || null);
                }
                
                // Marcar tweets padres
                if (estaPostear) {
                    tweet.dataset.esPadre = 'true';
                }
                
                tweetsComentarios.push(nuevoTweet);
                nuevoTweet.modificarDivBotones();
                nuevoTweet.establecerGuardado();
                nuevoTweet.colocarBotonesTweet();
            });

        }
        if (!divCronologia) {
            // 2. Lógica cuando NO existe divCronologia
            // Ejemplo: Procesar tweets fuera de la cronología de conversación
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
                                const tweetModificado = new Tweet(tweet);
                                tweetModificado.modificarDivBotones();
                                tweetModificado.establecerGuardado();
                                tweetModificado.colocarBotonesTweet();
                            }
                        }, 100);
                    }
                });
            });
        }
    });
});

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    document.querySelectorAll('article[data-testid="tweet"]').forEach(tweet => {
    if (!tweet.dataset.bienProcessed) {
        tweet.dataset.bienProcessed = 'true';
        const interval = setInterval(() => {
            if (tweet.querySelector('div[data-testid="tweetText"]')) {
                clearInterval(interval);
                const tweetModificado=new Tweet(tweet);
                tweetModificado.modificarDivBotones();
                tweetModificado.establecerGuardado();
                tweetModificado.colocarBotonesTweet();
            }
        }, 100);
    }
});
}

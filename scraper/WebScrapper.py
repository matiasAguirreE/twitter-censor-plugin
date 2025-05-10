from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import os

directorioActual=os.path.dirname(os.path.abspath(__file__))
directorioPadre=os.path.dirname(directorioActual)
rutaJavaScript=os.path.join(directorioActual, "Script.js")
rutaArchivoTweets=os.path.join(directorioPadre, "data", "raw", "Tweets.txt")

def inject_bien_buttons(driver):

    with open(rutaJavaScript, "r", encoding="utf-8") as file:
        script=file.read()
    
    driver.execute_script("window.__pyTweetText = null;")
    
    driver.execute_script(script)

def main():
    options = webdriver.ChromeOptions()
    options.add_argument("--user-data-dir=C:/selenium_profile")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--start-maximized")
    options.add_experimental_option("detach", True)
    
    driver = webdriver.Chrome(options=options)
    
    try:
        driver.get("https://twitter.com/login")
        print("⚠️ Inicia sesión MANUALMENTE y espera a que cargue el feed...")
        
        WebDriverWait(driver, 120).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'main[role="main"]'))
        )
        
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'article[data-testid="tweet"]'))
        )
        
        inject_bien_buttons(driver)
        print("✅ Botones configurados - Haz clic en cualquier botón para imprimir el tweet")
        
        last_printed = ""
        while True:
            # Obtener texto desde JavaScript
            tweet_text = driver.execute_script("""
                if (window.__pyTweetText) {
                    const text = window.__pyTweetText;
                    window.__pyTweetText = null;
                    return text;
                }
                return null;
            """)
            
            if tweet_text and tweet_text != last_printed:
                partes=tweet_text.split('\n#############\n')
                textoDelTweet,categorias=partes
                textoDelTweet=textoDelTweet.replace("\n"," ")
                if categorias=="":
                    categorias="Incensurable"
                    
                with open(rutaArchivoTweets,"a", encoding="utf-8") as archivo:
                    archivo.write(textoDelTweet+"#############"+categorias+"\n")
                print(f"✅ Tweet guardado en: {rutaArchivoTweets}")
                
            time.sleep(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        input("Presiona Enter para cerrar...")
        driver.quit()

if __name__ == "__main__":
    main()
    
    
    
"""
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
            mainButton.textContent = '📝 Imprimir Tweet';

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
                window.__pyTweetText = spans.map(span => span.textContent).join(' ');
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
    """
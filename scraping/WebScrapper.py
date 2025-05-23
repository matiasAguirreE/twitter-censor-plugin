from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
import time
import os

directorioActual=os.path.dirname(os.path.abspath(__file__))
rutaJavaScript=os.path.join(directorioActual, "Script2.js")
rutaArchivoTweets=os.path.join(directorioActual,"Tweets.txt")

def inject_bien_buttons(driver):
    try:
        # Esperar a que la página esté lista
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'main[role="main"]'))
        )
        
        # Resetear estado y inyectar script
        with open(rutaJavaScript, "r", encoding="utf-8") as file:
            script = file.read()
        
        driver.execute_script("""
            window.__pyTweetText = null;
            if (window.scriptInjected) return;
            window.scriptInjected = true;
        """ + script)
        
    except Exception as e:
        print(f"⚠️ Error inyectando botones: {str(e)}")

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
        print("⚠️ Inicia sesión MANUALMENTE...")
        
        last_printed = ""
        while True:
            # Verificar cambios de página
            current_url = driver.current_url
            if current_url != last_printed:
                print(f"🔄 Nueva página detectada: {current_url}")
                inject_bien_buttons(driver)
                last_printed = current_url
            
            # Lógica de guardado de tweets (igual)
            tweet_text = driver.execute_script("""
                if (!window.scriptInjected) return 'REINJECT';
                if (window.__pyTweetText) {
                    const text = window.__pyTweetText;
                    window.__pyTweetText = null;
                    return text;
                }
                return null;
            """)
            
            if tweet_text == "REINJECT":
                inject_bien_buttons(driver)
                continue
                
            if tweet_text and tweet_text != last_printed:
                print("Se esta escribiendo")
                partes=tweet_text.split('#############')
                textoDelTweet,categorias,textoTweetPadre=partes
                textoDelTweet=textoDelTweet.replace("\n"," ")
                textoTweetPadre=textoTweetPadre.replace("\n"," ")
                if categorias=="":
                    categorias="Incensurable"
                
                with open(rutaArchivoTweets,"a", encoding="utf-8") as archivo:
                    archivo.write(textoDelTweet+"#############"+categorias+"#############"+textoTweetPadre+"\n")
                print(f"✅ Tweet guardado en: {rutaArchivoTweets}")
            
            time.sleep(1)
            
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        input("Presiona Enter para cerrar...")
        driver.quit()

if __name__ == "__main__":
    main()
    
    

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from transformers import AutoTokenizer, get_linear_schedule_with_warmup,AutoModel
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
import itertools
from Entrenamiento import Entrenamiento
from Evaluacion import Evaluacion
from Clasificador import TweetClasificador
from Dataset import TweetDataset
import gc
import random

EPOCHS=[4,5]
LEARNING_RATES=[5e-5,6e-5]
BATCH_SIZES=[16,32]
WARMUP_RATIOS=[0.1,0.3]
WEIGHT_DECAYS=[0.01,0.015] 
semilla=42
random.seed(semilla)
combinaciones=[(4, 6e-05, 32, 0.3, 0.015), (4, 6e-05, 16, 0.1, 0.01), 
                (4, 5e-05, 16, 0.1, 0.01), (4, 6e-05, 32, 0.3, 0.015)]

todas=list(itertools.product(EPOCHS,LEARNING_RATES,BATCH_SIZES,WARMUP_RATIOS,WEIGHT_DECAYS))

noUsadas=[c for c in todas if c not in combinaciones]
    

for i in range(10):
    if len(todas)==0:
        break
    indice=random.randint(0,len(noUsadas)-1)
    combinaciones.append(noUsadas.pop(indice))
print("Estas son las combinaciones de hiperparametros con las que se entrenara el modelo\n",combinaciones)
g=torch.Generator()
g.manual_seed(semilla)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
np.random.seed(semilla)
torch.manual_seed(semilla)
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.bmm(torch.randn(2, 2, 2).to_sparse().cuda(), torch.randn(2, 2, 2).cuda())
torch.backends.cudnn.deterministic=True 
torch.backends.cudnn.benchmark=False
    
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prtin para ver que version de CUDA se usa
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version (compiled):", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

dfTrain=pd.read_csv("data_train.csv")
XEntrenamiento=dfTrain["Tweet"]
yEntrenamiento=dfTrain[["Violencia", "Homofobia", "Xenofobia"]]

dfValTest=pd.read_csv("data_test.csv")
XValTest=dfValTest["Tweet"]
yValTest=dfValTest[["Violencia","Homofobia","Xenofobia"]]

tokenizador=AutoTokenizer.from_pretrained("dccuchile/tulio-chilean-spanish-bert",from_tf=False)
criterion=nn.BCEWithLogitsLoss()

XTesting,XValidacion,yTesting,yValidacion=train_test_split(XValTest,yValTest,test_size=0.4,random_state=semilla)

train_ds=TweetDataset(XEntrenamiento,yEntrenamiento,tokenizador)
val_ds=TweetDataset(XValidacion,yValidacion,tokenizador)
test_ds=TweetDataset(XTesting,yTesting,tokenizador)

diccMejoresCombinaciones={"F1Macro":None,"F1Micro":None,"PromedioPrecision":None,"PromedioRecall":None,"PromedioAccuracy":None}

for c in combinaciones:
    
    epoca,lr,bs,wr,wd=c[0],c[1],c[2],c[3],c[4]
    torch.cuda.empty_cache()
    gc.collect()
    combinacionActual=(epoca,lr,bs,wr,wd)
    print(f"Epocas: {epoca}, Learning rate: {lr}, Batch size: {bs}, Warmup ratio: {wr}, Weight decay: {wd}")
    # 3.1 (Re)crea el modelo
    modelo=TweetClasificador(dropout=0.3).to(device)

    # 3.2 Prepara optimizer + scheduler
    optimizador=AdamW(modelo.parameters(),lr=lr,weight_decay=wd)
    trainLoader=DataLoader(train_ds,batch_size=bs,shuffle=True,generator=g,worker_init_fn=seed_worker)
    pasosTotales=len(trainLoader)*epoca
    pasosWarmup=int(wr*pasosTotales)
    scheduler=get_linear_schedule_with_warmup(
                       optimizador,
                       num_warmup_steps=pasosWarmup,
                       num_training_steps=pasosTotales
                   )
    perdidasFinales=[]
    Entrenamiento(epoca,modelo,optimizador,trainLoader,scheduler,criterion,device,perdidasFinales)
    
    perdidaEntrenamiento=sum(perdidasFinales)/len(perdidasFinales)

    # 3.4 EvaluaciÃƒÂ³n en validaciÃƒÂ³n
    val_loader=DataLoader(val_ds, batch_size=bs, shuffle=False)
    dicc=Evaluacion(modelo,val_loader,criterion,device)
    dicc["perdidaEntrenamiento"]=perdidaEntrenamiento
    # 3.6 Actualiza mejor modelo
    if diccMejoresCombinaciones["F1Macro"] is None or dicc["f1Macro"]>diccMejoresCombinaciones["F1Macro"][1]["f1Macro"]:
        print("Se actualizo f1Macro")
        diccMejoresCombinaciones["F1Macro"]=(combinacionActual,dicc)
        torch.save(modelo.state_dict(),"MejorModeloF1Macro.pth")
    if diccMejoresCombinaciones["F1Micro"] is None or dicc["f1Micro"]>diccMejoresCombinaciones["F1Micro"][1]["f1Micro"]:
        print("Se actualizo f1Micro")
        diccMejoresCombinaciones["F1Micro"]=(combinacionActual,dicc)
        torch.save(modelo.state_dict(),"MejorModeloF1Micro.pth")
    if diccMejoresCombinaciones["PromedioPrecision"] is None or dicc["promedioPrecision"]>diccMejoresCombinaciones["PromedioPrecision"][1]["promedioPrecision"]:
        print("Se actualizo promedio Precision")
        diccMejoresCombinaciones["PromedioPrecision"]=(combinacionActual,dicc)
        torch.save(modelo.state_dict(),"MejorModeloPromedioPrecision.pth")
    if diccMejoresCombinaciones["PromedioRecall"] is None or dicc["promedioRecall"]>diccMejoresCombinaciones["PromedioRecall"][1]["promedioRecall"]:
        print("Se actualizo promedio Recall")
        diccMejoresCombinaciones["PromedioRecall"]=(combinacionActual,dicc)
        torch.save(modelo.state_dict(),"MejorModeloPromedioRecall.pth")
    if diccMejoresCombinaciones["PromedioAccuracy"] is None or dicc["promedioAccuracy"]>diccMejoresCombinaciones["PromedioAccuracy"][1]["promedioAccuracy"]:
        print("Se actualizo accuracy")
        diccMejoresCombinaciones["PromedioAccuracy"]=(combinacionActual,dicc)
        torch.save(modelo.state_dict(),"MejorModeloAccuracy.pth")
    

    with open("RegistroCombinacionesResultados.txt", "a",encoding="utf-8") as registro:
      registro.write(f"{combinacionActual}: {dicc}\n")
      
    print(f"Memoria reservada: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    print(f"Memoria asignada: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Memoria libre en el cache: {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / 1024**2:.2f} MB")
    
    del modelo, optimizador, scheduler, trainLoader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
with open("MejoresCombinaciones.txt","w") as archivo:
    archivo.write(f"Los resultados finales de la validacion para la mejor combinacion F1-Macro ({diccMejoresCombinaciones['F1Macro'][0]}):"+str(diccMejoresCombinaciones['F1Macro'][1]) +"\n")
    archivo.write(f"Los resultados finales de la validacion para la mejor combinacion F1-Micro ({diccMejoresCombinaciones['F1Micro'][0]})"+str(diccMejoresCombinaciones['F1Micro'][1])+"\n")
    archivo.write(f"Los resultados finales de la validacion para la mejor combinacion promedio Precision ({diccMejoresCombinaciones['PromedioPrecision'][0]})"+str(diccMejoresCombinaciones['PromedioPrecision'][1])+"\n")
    archivo.write(f"Los resultados finales de la validacion para la mejor combinacion promedio Recall ({diccMejoresCombinaciones['PromedioRecall'][0]})"+str(diccMejoresCombinaciones['PromedioRecall'][1])+"\n")
    archivo.write(f"Los resultados finales de la validacion para la mejor combinacion Accuracy ({diccMejoresCombinaciones['PromedioAccuracy'][0]})"+str(diccMejoresCombinaciones['PromedioAccuracy'][1])+"\n")
    
print("Se esta testeando las mejores combinaciones encontradas con el conjunto de testing")
modeloTestF1Macro=TweetClasificador(dropout=0.3).to(device)
modeloTestF1Micro=TweetClasificador(dropout=0.3).to(device)
modeloTestPromedioPrecision=TweetClasificador(dropout=0.3).to(device)
modeloTestPromedioRecall=TweetClasificador(dropout=0.3).to(device)
modeloTestAccuracy=TweetClasificador(dropout=0.3).to(device)


modeloTestF1Macro.load_state_dict(torch.load("MejorModeloF1Macro.pth"))
modeloTestF1Micro.load_state_dict(torch.load("MejorModeloF1Micro.pth"))
modeloTestPromedioPrecision.load_state_dict(torch.load("MejorModeloPromedioPrecision.pth"))
modeloTestPromedioRecall.load_state_dict(torch.load("MejorModeloPromedioRecall.pth"))
modeloTestAccuracy.load_state_dict(torch.load("MejorModeloAccuracy.pth"))

testLoaderF1Macro=DataLoader(test_ds, batch_size=diccMejoresCombinaciones["F1Macro"][0][2], shuffle=False)
testLoaderF1Micro=DataLoader(test_ds, batch_size=diccMejoresCombinaciones["F1Micro"][0][2], shuffle=False)
testLoaderPromedioPrecision=DataLoader(test_ds, batch_size=diccMejoresCombinaciones["PromedioPrecision"][0][2], shuffle=False)
testLoaderPromedioRecall=DataLoader(test_ds, batch_size=diccMejoresCombinaciones["PromedioRecall"][0][2], shuffle=False)
testLoaderAccuracy=DataLoader(test_ds, batch_size=diccMejoresCombinaciones["PromedioAccuracy"][0][2], shuffle=False)

print("Se estan evaluando los modelos")
diccResultadosF1Macro=Evaluacion(modeloTestF1Macro,testLoaderF1Macro,criterion,device)
diccResultadosF1Micro=Evaluacion(modeloTestF1Micro,testLoaderF1Micro,criterion,device)
diccResultadosPromedioPrecision=Evaluacion(modeloTestPromedioPrecision,testLoaderPromedioPrecision,criterion,device)
diccResultadosPromedioRecall=Evaluacion(modeloTestPromedioRecall,testLoaderPromedioRecall,criterion,device)
diccResultadosAccuracy=Evaluacion(modeloTestAccuracy,testLoaderAccuracy,criterion,device)

print("El proceso ha terminado")
with open("ResultadosTestingModelos.txt","w") as archivo:
    archivo.write("Modelo f1Macro: "+str(diccResultadosF1Macro)+"\n")
    archivo.write("Modelo f1Micro: "+str(diccResultadosF1Micro)+"\n")
    archivo.write("Modelo promedio Precision:"+str(diccResultadosPromedioPrecision)+"\n")
    archivo.write("Modelo promedio Recall"+str(diccResultadosPromedioRecall)+"\n")
    archivo.write("Modelo Accuracy"+str(diccResultadosAccuracy)+"\n")

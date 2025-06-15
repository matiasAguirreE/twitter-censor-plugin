import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score,recall_score
import gc


def Evaluacion(model,valLoader,criterion,device):
    print("Se esta probando el modelo")
    model.eval()
    predicciones,etiquetas=[],[]
    perdidas=[]
    with torch.no_grad():
        for batch in valLoader:
            entrada=batch["input_ids"].to(device)
            attention_mask=batch["attention_mask"].to(device)
            etiquetasTemp=batch["labels"].to(device).float()

            logits=model(entrada,attention_mask)
            perdida=criterion(logits,etiquetasTemp)
            perdidas.append(perdida.item())
            
            logitsNp=torch.sigmoid(logits).cpu().numpy()
            etiquetasNp=etiquetasTemp.cpu().numpy()
            predicciones.extend(logitsNp)
            etiquetas.extend(etiquetasNp)
            del entrada,attention_mask,logits,etiquetasTemp,logitsNp
            torch.cuda.empty_cache()
            
    prediccionesNP=np.array(predicciones)
    etiquetasNP=np.array(etiquetas)
    predicciones=(torch.tensor(prediccionesNP)>0.5).int()
    etiquetas= torch.tensor(etiquetasNP).int()
    metricasClases={}
    for i,etiqueta in enumerate(["Violencia", "Homofobia","Xenofobia"]):
        if (etiquetas[:,i]).sum()>0:
            f1=f1_score(etiquetas[:,i],predicciones[:,i],zero_division=0)
            precision=precision_score(etiquetas[:,i],predicciones[:,i],zero_division=0)
            recall=recall_score(etiquetas[:,i],predicciones[:,i],zero_division=0)
        metricasClases[etiqueta]={"f1":f1,"precision":precision,"recall":recall}
        print(f"{etiqueta}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
    f1Macro=f1_score(etiquetas,predicciones,average="macro",zero_division=0)
    f1Micro=f1_score(etiquetas,predicciones,average="micro",zero_division=0)
    promedioPrecision=precision_score(etiquetas,predicciones,average="macro",zero_division=0)
    promedioRecall=recall_score(etiquetas,predicciones,average="macro",zero_division=0)
    
    del prediccionesNP,etiquetasNp,predicciones,etiquetas
    gc.collect()
    torch.cuda.empty_cache()
    diccionarioResultados={"perdidaEvaluacion":sum(perdidas)/len(perdidas),"f1Macro":f1Macro,"f1Micro":f1Micro,"promedioPrecision":promedioPrecision,"promedioRecall":promedioRecall,"metricasClases":metricasClases}
    return diccionarioResultados
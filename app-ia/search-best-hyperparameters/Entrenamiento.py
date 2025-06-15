import gc
import torch
def Entrenamiento(epocas,modelo,optimizador,trainLoader,scheduler,criterion,device,perdidasFinales):
    for e in range(epocas):
        print("Se esta en la epoca",e+1)
        modelo.train()
        for lote in trainLoader:
            optimizador.zero_grad()
            logits=modelo(lote["input_ids"].to(device),lote["attention_mask"].to(device))
            perdida=criterion(logits,lote["labels"].to(device))
            del lote
            torch.cuda.empty_cache()
            perdida.backward()
            optimizador.step()
            scheduler.step()
            if e==epocas-1:
                perdidasFinales.append(perdida.item())
            del logits,perdida
            torch.cuda.empty_cache()
            
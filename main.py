from fastapi import FastAPI, HTTPException
import torch
import pickle
from azure.storage.blob import BlobServiceClient
import io
import os
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

# Función para descargar el modelo desde Azure Blob Storage y cargarlo en memoria
def load_model_from_blob(connect_str, container_name, blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    
    blob_data = blob_client.download_blob().readall()    
    model = pickle.loads(blob_data)
    model = model.to(torch.device('cpu'))
    
    return model
# Configuración de conexión a Azure Blob Storage
CONNECT_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = os.getenv("AZURE_STORAGE_CONTAINER_NAME")
BLOB_NAME = os.getenv("AZURE_STORAGE_BLOB_NAME")

# Cargar el modelo al iniciar la aplicación
#try:
#    model = load_model_from_blob(CONNECT_STR, CONTAINER_NAME, BLOB_NAME)
#    tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-uncased')
#except Exception as e:
#    raise RuntimeError(f"Error loading model: {str(e)}")

@app.get('/')
def hello():
    return {'message':'Hello World'}

#@app.post('/predict')
#def predict(request: dict):
    #text = request.get('text')
    #if not text:
    #    raise HTTPException(status_code=400, detail="Text field is required")
    
    #encoded_text = tokenizer(text, return_tensors="pt")
    
    #with torch.no_grad():
    #    outputs = model(**encoded_text)
    #    predicted_class = torch.argmax(outputs.logits).item()
    
    #return {'prediction': predicted_class}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)    
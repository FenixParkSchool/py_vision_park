from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Inicializa a aplicação FastAPI corretamente com título e descrição
app = FastAPI(
    title="Microserviço ALPR - Estacionamento",
    description="API em Python para reconhecimento de placas de veículos."
)

# --- MODELOS DE DADOS ---

class SolicitacaoLeitura(BaseModel):
    id_camera: str

class ResultadoLeitura(BaseModel):
    placa: Optional[str] = None
    confianca: float
    data_hora: datetime
    sucesso: bool

# --- ROTAS (ENDPOINTS) ---

@app.get("/")
def health_check():
    """Verifica se o microserviço está online."""
    return {
        "status": "online", 
        "mensagem": "Microserviço de reconhecimento operando normalmente."
    }

@app.post("/reconhecer", response_model=ResultadoLeitura)
def processar_camera(solicitacao: SolicitacaoLeitura):
    """
    Recebe um gatilho, processa a imagem da câmera e retorna a placa lida.
    """
    # Por enquanto, retornamos um dado falso (mock) para testar a comunicação
    return ResultadoLeitura(
        placa="ABC1D23",
        confianca=0.98,
        data_hora=datetime.now(),
        sucesso=True
    )
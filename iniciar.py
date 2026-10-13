import os
import subprocess
import sys

# Define a versão do Python compatível com o TensorFlow
PYTHON_VERSION = "3.11"

def rodar_comando(comando):
    """Executa um comando no terminal e para o script se der erro."""
    try:
        subprocess.run(comando, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERRO] Falha ao executar: {comando}")
        sys.exit(1)

def main():
    print("=== Configurando o Microserviço ALPR ===\n")

    # 1. Cria o ambiente virtual forçando a versão correta do Python
    if not os.path.exists(".venv"):
        print(f"[1/3] Criando ambiente virtual (.venv) com Python {PYTHON_VERSION}...")
        rodar_comando(f"py -m uv venv --python {PYTHON_VERSION}")
    else:
        print(f"[1/3] Ambiente virtual (.venv) já existe.")

    # 2. Instala ou sincroniza as dependências DIRETAMENTE no .venv
    print("[2/3] Verificando dependências...")
    if os.path.exists("requirements.lock"):
        rodar_comando("py -m uv pip sync requirements.lock --python .venv")
    elif os.path.exists("requirements-dev.txt"):
        rodar_comando("py -m uv pip install -r requirements-dev.txt --python .venv")
        print("\nGerando arquivo requirements.lock atualizado...")
        rodar_comando("py -m uv pip compile requirements-dev.txt -o requirements.lock")
    else:
        print("[ERRO] Nenhum arquivo de requirements encontrado!")
        sys.exit(1)

    # 3. Inicia o servidor chamando o Python de dentro do .venv para garantir o isolamento
    print("\n[3/3] Iniciando o servidor FastAPI...\n")
    rodar_comando(r".venv\Scripts\python.exe -m uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
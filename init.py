import os
import subprocess
import sys

def rodar_comando(comando):
    """Executa um comando no terminal e para o script se der erro."""
    try:
        subprocess.run(comando, check=True, shell=True)
    except subprocess.CalledProcessError:
        print(f"\n[ERRO] Falha ao executar: {comando}")
        sys.exit(1)

def main():
    print("=== Configurando o Microserviço ALPR ===\n")

    # 1. Cria o ambiente virtual se ele não existir
    if not os.path.exists(".venv"):
        print("[1/3] Criando ambiente virtual (.venv)...")
        rodar_comando("uv venv")
    else:
        print("[1/3] Ambiente virtual já existe.")

    # 2. Instala ou sincroniza as dependências
    print("[2/3] Verificando dependências...")
    if os.path.exists("requirements.lock"):
        rodar_comando("uv pip sync requirements.lock")
    elif os.path.exists("requirements-dev.txt"):
        rodar_comando("uv pip install -r requirements-dev.txt")
    else:
        print("[ERRO] Nenhum arquivo de requirements encontrado!")
        sys.exit(1)

    # 3. Inicia o servidor usando 'uv run' (não precisa ativar o venv manualmente)
    print("\n[3/3] Iniciando o servidor FastAPI...\n")
    rodar_comando("uv run uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
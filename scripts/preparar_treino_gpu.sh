#!/usr/bin/env bash
# Prepara o ambiente de TREINO com GPU NVIDIA (WSL2/Ubuntu) em um comando.
#
# Uso (dentro do Ubuntu do WSL2, na raiz do repositório):
#   bash scripts/preparar_treino_gpu.sh [CAMINHO_DO_DATASET]
# Exemplo, com o dataset num pendrive (E: no Windows):
#   bash scripts/preparar_treino_gpu.sh /mnt/e/rodosol
#
# O que faz: dependências do sistema, uv, Python 3.11, ambiente .venv com
# requirements-gpu.lock (TensorFlow com CUDA + fast-plate-ocr com CLI de treino),
# confere GPU, copia/confere o dataset, baixa o OCR pronto, cria o .env e roda os testes.
# Pode ser executado de novo sem problema (cada etapa pula o que já está pronto).

set -euo pipefail

RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RAIZ"
DATASET_ORIGEM="${1:-}"
PY=".venv/bin/python"

passo() { printf '\n==> %s\n' "$1"; }
aviso() { printf '\n[AVISO] %s\n' "$1"; }

passo "1/8 Dependências do sistema (pede a senha do Ubuntu)"
# libgl1/libglib2.0-0: exigidas pelo OpenCV; rsync: cópia do dataset
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1 libglib2.0-0 rsync curl git

passo "2/8 uv (gerenciador de Python e pacotes)"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

passo "3/8 Ambiente .venv com Python 3.11 e dependências travadas (requirements-gpu.lock)"
[ -d .venv ] || uv venv --python 3.11 .venv
uv pip sync requirements-gpu.lock --python .venv

passo "4/8 GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
else
  aviso "nvidia-smi não encontrado. Atualize o driver NVIDIA no WINDOWS (não instale driver no Ubuntu)."
fi
if "$PY" -c "import tensorflow as tf, sys; g = tf.config.list_physical_devices('GPU'); print('TensorFlow', tf.__version__, 'GPUs:', g); sys.exit(0 if g else 1)" 2>/dev/null; then
  echo "OK: o TensorFlow enxerga a GPU."
else
  aviso "o TensorFlow NÃO enxergou a GPU. O treino vai rodar em CPU (lento). Veja docs/gpu_wsl2.md."
fi

passo "5/8 Dataset RodoSol-ALPR em data/rodosol"
mkdir -p data/rodosol
if [ -n "$DATASET_ORIGEM" ]; then
  echo "Copiando de $DATASET_ORIGEM (pode levar vários minutos, ~9 GB)..."
  rsync -a --info=progress2 "$DATASET_ORIGEM"/ data/rodosol/
fi
TOTAL=$(find data/rodosol -name '*.jpg' 2>/dev/null | wc -l)
echo "Imagens encontradas: $TOTAL (esperado: 20000 em cars-br, cars-me, motorcycles-br, motorcycles-me)"
if [ "$TOTAL" -lt 20000 ]; then
  aviso "dataset incompleto. Rode de novo informando a pasta: bash scripts/preparar_treino_gpu.sh /mnt/<letra>/rodosol"
fi

passo "6/8 Modelo de OCR pronto (fast-plate-ocr cct-s-v2) - baixa uma vez para ~/.cache/fast-plate-ocr"
"$PY" -c "from fast_plate_ocr import LicensePlateRecognizer; LicensePlateRecognizer('cct-s-v2-global-model'); print('OK: cct-s-v2 disponível')" \
  || aviso "não foi possível baixar o modelo (sem internet?). Rode o script de novo depois."

passo "7/8 Arquivo .env (não vai para o git)"
if [ ! -f .env ]; then
  sed -e 's/^RECOGNITION_MODE=fake/RECOGNITION_MODE=fpo/' \
      -e 's/^OCR_MIN_CONFIDENCE=$/OCR_MIN_CONFIDENCE=0.95/' \
      .env.example > .env
  echo ".env criado a partir do .env.example (modo fpo, limiar 0.95). Preencha BACKEND_API_KEY."
else
  echo ".env já existe; não foi alterado."
fi
[ -f models/detector.keras ] || aviso "models/detector.keras não encontrado: copie a pasta models/ do Senac ou retreine (python -m training.train_detector --per-folder 600 --epochs 12)."

passo "8/8 Testes unitários"
"$PY" -m pytest tests/unit -q

cat <<'FIM'

Ambiente pronto. Próximos comandos (dentro do Ubuntu, nesta pasta):
  source .venv/bin/activate
  # avaliar o OCR pronto no teste do RodoSol (referência: 78,9% ponta a ponta)
  python -m experiments.avaliar_fpo --limit 0
  # treinar o detector, se não tiver models/detector.keras
  python -m training.train_detector --per-folder 600 --epochs 12
  # exportar recortes no formato do fast-plate-ocr (data/fpo/{train,val,test}.csv)
  python -m training.exportar_fpo
  # treinar o nosso OCR (plano B), com checkpoint por época
  python -m training.train_ocr --arch positions --per-folder 3000 --epochs 60 --out models/ocr_positions.keras
Roteiro completo do treino em casa: contexto_projeto.md, seção 19.
FIM

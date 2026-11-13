# Treinar com GPU (RTX 3060) via WSL2

Escrito em 13/11/2026. Nada aqui foi executado ainda; os passos seguem a documentação oficial do TensorFlow e da NVIDIA para WSL2.

## Por que WSL2
O TensorFlow **não usa GPU no Windows nativo desde a versão 2.11** (o próprio log do treino avisa isso). No Windows, a GPU funciona dentro do **WSL2** (um Linux rodando no Windows), usando o driver NVIDIA do Windows.

- **RTX 3060 (12 GB, seu PC)**: vale muito a pena. Estimativa (não medida): época do OCR de ~2,6 min em CPU para algo entre 5 e 20 s.
- **GTX 745 (4 GB DDR3, Senac)**: não vale a pena. É uma placa antiga (arquitetura Maxwell, de 2014); as versões recentes do TensorFlow podem nem ter suporte a ela, e mesmo que funcionasse o ganho seria pequeno. No Senac, seguir em CPU.

## Atalho: um comando
Depois dos passos 1 a 3 abaixo (driver, WSL2, `nvidia-smi`), dentro do Ubuntu:
```
git clone https://github.com/FenixParkSchool/alpr_park.git ~/py_vision_park && bash ~/py_vision_park/scripts/preparar_treino_gpu.sh /mnt/e/rodosol
```
(troque `/mnt/e/rodosol` pela pasta do dataset; sem argumento, ele só confere `data/rodosol`). O script instala tudo pelo `requirements-gpu.lock`, confere a GPU e o dataset, baixa o OCR pronto, cria o `.env` e roda os testes. Pode ser repetido.

## Passo a passo (no seu PC)

1. **Driver NVIDIA no Windows**: atualize pelo GeForce Experience/NVIDIA App ou pelo site da NVIDIA. **Não** instale driver de vídeo dentro do Linux; o WSL2 usa o do Windows.
2. **Instalar o WSL2 com Ubuntu** (PowerShell como administrador):
   ```
   wsl --install -d Ubuntu-22.04
   ```
   Reinicie quando pedir, abra "Ubuntu" no menu Iniciar e crie usuário/senha. Depois, no PowerShell: `wsl --update`.
3. **Conferir a GPU dentro do Ubuntu**:
   ```
   nvidia-smi
   ```
   Deve aparecer "NVIDIA GeForce RTX 3060". Se não aparecer, o problema é o driver do Windows (passo 1).
4. **Python 3.11 e o projeto** (dentro do Ubuntu; trabalhe na pasta do Linux, `~`, e não em `/mnt/c/...`, que é muito mais lenta para ler 20 000 imagens):
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.bashrc
   git clone https://github.com/FenixParkSchool/alpr_park.git ~/py_vision_park
   cd ~/py_vision_park
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -r requirements-dev.txt
   uv pip install "tensorflow[and-cuda]==2.21.*"
   ```
   O extra `[and-cuda]` instala CUDA e cuDNN pelo pip; não precisa instalar CUDA à mão.
5. **Conferir que o TensorFlow vê a GPU**:
   ```
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
   Esperado: uma lista com `PhysicalDevice(name='/physical_device:GPU:0', ...)`. Lista vazia = sem GPU (rever passos 1 a 4).
6. **Dataset**: copiar `data/rodosol/` (as 4 pastas, ~9 GB) para `~/py_vision_park/data/rodosol/`. De um pendrive no Windows, o caminho no Ubuntu fica em `/mnt/<letra>/...`, por exemplo:
   ```
   cp -r /mnt/e/rodosol ~/py_vision_park/data/
   ```
   O RodoSol-ALPR é liberado mediante pedido para pesquisa: use só para este projeto e não publique as imagens.
7. **Modelos já treinados** (opcional): copiar `models/` do Senac (detector bom) para `~/py_vision_park/models/`.
8. **Treinar**: os mesmos comandos do plano, com `python` no lugar de `.venv\Scripts\python`, por exemplo:
   ```
   python -m training.train_ocr --arch positions --per-folder 3000 --epochs 60 --out models/ocr_pos_v3.keras
   python -m training.evaluate --ocr models/ocr_pos_v3.keras --out models/avaliacao_v3.json
   ```
9. **Trazer os modelos de volta** para o Windows: no Explorador de Arquivos, o caminho `\\wsl$\Ubuntu-22.04\home\<seu-usuario>\py_vision_park\models` mostra os arquivos.

## Cuidados
- Na GPU, o gargalo pode passar a ser a preparação das imagens em Python (recorte e augmentação a cada época). Se a GPU ficar ociosa (`nvidia-smi` mostra uso baixo), aumentar os `workers` do gerador de dados.
- Anotar no `contexto_projeto.md` a versão do driver, do TensorFlow e o tempo por época medido na GPU.
- O serviço em produção (webcam) continua rodando em CPU no Windows normalmente; a GPU só acelera o treino.

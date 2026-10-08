import cv2

# (.shape) mostra a dimensão da imagem
# (909, 1212, 3) diz que ela tem 909 linhas (altura da foto), 1212 colunas (largura) e 3 camadas empilhadas. 
# Cada célula (linha, coluna) é um pixel, e as 3 camadas dão a quantidade de azul, verde e vermelho dele, de 0 (nada) a 255 (máximo).

# 1) Ler a imagem e validar ANTES de usar (imread devolve None se falhar, sem avisar)
imagem = cv2.imread("tests/carro.png")

if imagem is None:
    raise FileNotFoundError("Não consegui abrir a imagem, confira o caminho")

# 2) Formato: (altura, largura, canais). Ex.: (909, 1212, 3) = 909 linhas, 1212 colunas,
#    3 camadas (azul, verde, vermelho), cada valor de 0 a 255.
print("Original:", imagem.shape)

# 3) Pixel do canto superior esquerdo: [linha, coluna] -> [B, G, R]
print("Pixel [0, 0] (B, G, R):", imagem[0, 0])

# 4) Versão reduzida. Atenção: resize recebe (largura, altura), ordem inversa do .shape
reduzida = cv2.resize(imagem, (640, 480))
print("Reduzida:", reduzida.shape)

# 5) Versão em cinza: some o terceiro número, pois cada pixel vira um valor só
cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
print("Cinza:", cinza.shape)

# 6) Canais separados (OpenCV usa BGR: índice 0 = azul, 1 = verde, 2 = vermelho)
azul = imagem[:, :, 0]
verde = imagem[:, :, 1]
vermelho = imagem[:, :, 2]

# 7) Cores trocadas de propósito, para ver o efeito do BGR vs RGB
trocada = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

# 8) Salvar tudo. imwrite devolve False (sem erro) se não conseguir gravar
arquivos = {
    "tests/carro_reduzida.png": reduzida,
    "tests/carro_cinza.png": cinza,
    "tests/carro_azul.png": azul,
    "tests/carro_verde.png": verde,
    "tests/carro_vermelho.png": vermelho,
    "tests/carro_trocada.png": trocada,
}

for caminho, img in arquivos.items():
    ok = cv2.imwrite(caminho, img)
    print(f"Salvou {caminho}?", ok)
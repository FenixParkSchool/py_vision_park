"""Votação entre quadros: junta as leituras do mesmo veículo e envia uma só, votada.

Por que existe: o OCR erra de forma aleatória em quadros borrados. Sem votação, a primeira
leitura de cada placa é enviada; se ela estiver errada (ABE1D23 no lugar de ABC1D23), vira
"outra placa", passa pelo Deduplicator e gera um evento errado a mais.

Como funciona (conta em QUADROS amostrados, não em segundos, para valer igual na webcam ao
vivo e num vídeo gravado, que é processado mais rápido que o tempo real):
- Cada leitura válida entra no grupo cujo texto votado difere dela em até `max_diff`
  posições (o mesmo veículo); se nenhum servir, abre um grupo novo (outro veículo).
- O grupo é enviado UMA vez: quando junta `max_reads` leituras ou quando fica `gap_frames`
  quadros seguidos sem leitura (o veículo saiu), o que vier primeiro. Depois de enviado,
  continua absorvendo leituras do mesmo veículo sem reenviar, até fechar.
- Voto por posição, com peso pela confiança do OCR. Não precisa de plano B para resultado
  inválido: toda placa válida tem letras nas posições 1-3 e dígitos nas posições 4, 6 e 7,
  e a posição 5 aceita letra (Mercosul) ou dígito (antiga). Logo, qualquer combinação por
  posição de leituras válidas também é válida.
- Leituras fora do formato (Mercosul/antiga) não votam: seguem direto, e o worker as
  descarta e conta em `invalid_plates`, como antes.

Estado só em memória, por instância (uma câmera por pipeline).
"""

from dataclasses import dataclass, field

from app.recognition.plate_format import normalize_plate
from app.schemas.recognition import PlateObservation


@dataclass
class _Grupo:
    ultimo_quadro: int
    leituras: list[tuple[str, PlateObservation]] = field(default_factory=list)
    enviado: bool = False

    def texto_votado(self) -> str:
        pesos: list[dict[str, float]] = [{} for _ in range(len(self.leituras[0][0]))]
        for texto, obs in self.leituras:
            for i, c in enumerate(texto):
                pesos[i][c] = pesos[i].get(c, 0.0) + obs.ocr_confidence
        # max() devolve o primeiro máximo: em empate vence o caractere visto primeiro.
        return "".join(max(p.items(), key=lambda kv: kv[1])[0] for p in pesos)


def _diferencas(a: str, b: str) -> int:
    if len(a) != len(b):
        return len(a) + len(b)
    return sum(x != y for x, y in zip(a, b))


class PlateVoter:
    def __init__(self, max_reads: int = 3, gap_frames: int = 3, max_diff: int = 2) -> None:
        if max_reads < 1 or gap_frames < 1 or max_diff < 0:
            raise ValueError("max_reads e gap_frames devem ser >= 1 e max_diff >= 0")
        self._max_reads = max_reads
        self._gap_frames = gap_frames
        self._max_diff = max_diff
        self._quadro = 0
        self._grupos: list[_Grupo] = []

    def __len__(self) -> int:
        return len(self._grupos)

    def push_frame(self, observacoes: list[PlateObservation]) -> list[PlateObservation]:
        """Recebe as leituras de UM quadro (pode ser lista vazia) e devolve o que enviar agora."""
        self._quadro += 1
        saida: list[PlateObservation] = []
        for obs in observacoes:
            placa = normalize_plate(obs.plate_text)
            if not placa.is_valid:
                saida.append(obs)
                continue
            grupo = self._grupo_de(placa.text)
            grupo.leituras.append((placa.text, obs))
            grupo.ultimo_quadro = self._quadro
            if not grupo.enviado and len(grupo.leituras) >= self._max_reads:
                saida.append(self._resultado(grupo))
                grupo.enviado = True
        for grupo in [g for g in self._grupos if self._quadro - g.ultimo_quadro >= self._gap_frames]:
            self._grupos.remove(grupo)
            if not grupo.enviado:
                saida.append(self._resultado(grupo))
        return saida

    def flush(self) -> list[PlateObservation]:
        """Fim do vídeo/parada da câmera: envia os grupos que ainda não foram enviados."""
        pendentes = [self._resultado(g) for g in self._grupos if not g.enviado]
        self._grupos.clear()
        return pendentes

    def _grupo_de(self, texto: str) -> _Grupo:
        candidatos = [(_diferencas(texto, g.texto_votado()), g) for g in self._grupos]
        candidatos = [(d, g) for d, g in candidatos if d <= self._max_diff]
        if candidatos:
            return min(candidatos, key=lambda dg: dg[0])[1]
        grupo = _Grupo(ultimo_quadro=self._quadro)
        self._grupos.append(grupo)
        return grupo

    @staticmethod
    def _resultado(grupo: _Grupo) -> PlateObservation:
        texto = grupo.texto_votado()
        obs = [o for _, o in grupo.leituras]
        iguais = [o for t, o in grupo.leituras if t == texto] or obs
        base = max(iguais, key=lambda o: o.ocr_confidence)
        return base.model_copy(
            update={
                "plate_text": texto,
                # confiança média das leituras que concordam com o resultado (todas, se ele
                # saiu da combinação por posição e nenhuma leitura bate inteira)
                "ocr_confidence": sum(o.ocr_confidence for o in iguais) / len(iguais),
                "detection_confidence": max(o.detection_confidence for o in obs),
                "captured_at": obs[0].captured_at,  # quando o veículo foi visto pela 1ª vez
            }
        )

from datetime import datetime, timezone

import numpy as np
import pytest

from app.recognition.base import PlateDetection, PlateReading
from app.recognition.fakes import FakePlateDetector, FakePlateReader
from app.schemas.recognition import BoundingBox
from app.services.pipeline import RecognitionPipeline
from app.services.deduplicator import Deduplicator

AGORA = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
CAIXA = BoundingBox(x=10, y=20, width=30, height=40)


def frame():
    return np.zeros((100, 200, 3), dtype=np.uint8)


def deteccao(confianca=0.9, caixa=CAIXA):
    return PlateDetection(bbox=caixa, confidence=confianca)


def criar(deteccoes, leitura, min_confidence=0.6):
    leitor = FakePlateReader(leitura)
    pipeline = RecognitionPipeline(
        FakePlateDetector(deteccoes), leitor, camera_id="portaria-1", min_confidence=min_confidence
    )
    return pipeline, leitor


def test_fluxo_completo_gera_uma_observacao():
    pipeline, _ = criar([deteccao(0.9)], PlateReading("ABC1D23", 0.8))
    resultado = pipeline.process_frame(frame(), AGORA)
    assert len(resultado) == 1
    obs = resultado[0]
    assert obs.camera_id == "portaria-1"
    assert obs.plate_text == "ABC1D23"
    assert obs.detection_confidence == 0.9
    assert obs.ocr_confidence == 0.8
    assert obs.captured_at == AGORA
    assert obs.bbox == CAIXA


def test_frame_sem_placa_devolve_lista_vazia():
    pipeline, leitor = criar([], PlateReading("ABC1D23", 0.8))
    assert pipeline.process_frame(frame(), AGORA) == []
    assert leitor.crops == []


@pytest.mark.parametrize("confianca, aceita", [(0.59, False), (0.6, True), (0.61, True)])
def test_limite_de_confianca_da_deteccao(confianca, aceita):
    pipeline, _ = criar([deteccao(confianca)], PlateReading("ABC1D23", 0.9), min_confidence=0.6)
    assert bool(pipeline.process_frame(frame(), AGORA)) is aceita


def test_deteccao_abaixo_do_limite_nem_chega_ao_leitor():
    pipeline, leitor = criar([deteccao(0.3)], PlateReading("ABC1D23", 0.9))
    pipeline.process_frame(frame(), AGORA)
    assert leitor.crops == []


def test_leitura_abaixo_do_limite_e_descartada():
    pipeline, _ = criar([deteccao(0.9)], PlateReading("ABC1D23", 0.3))
    assert pipeline.process_frame(frame(), AGORA) == []


@pytest.mark.parametrize("leitura", [None, PlateReading("", 0.9), PlateReading("   ", 0.9)])
def test_placa_ilegivel_e_descartada(leitura):
    pipeline, _ = criar([deteccao(0.9)], leitura)
    assert pipeline.process_frame(frame(), AGORA) == []


def test_caixa_fora_do_frame_e_descartada_sem_chamar_o_leitor():
    fora = BoundingBox(x=500, y=500, width=10, height=10)
    pipeline, leitor = criar([deteccao(0.9, fora)], PlateReading("ABC1D23", 0.9))
    assert pipeline.process_frame(frame(), AGORA) == []
    assert leitor.crops == []


def test_leitor_recebe_o_recorte_com_o_tamanho_da_caixa():
    pipeline, leitor = criar([deteccao(0.9)], PlateReading("ABC1D23", 0.9))
    pipeline.process_frame(frame(), AGORA)
    assert leitor.crops[0].shape == (40, 30, 3)


def test_varias_deteccoes_saem_da_mais_para_a_menos_confiante():
    pipeline, _ = criar([deteccao(0.7), deteccao(0.95)], PlateReading("ABC1D23", 0.9))
    resultado = pipeline.process_frame(frame(), AGORA)
    assert [o.detection_confidence for o in resultado] == [0.95, 0.7]


def test_falha_do_detector_nao_e_escondida():
    class DetectorQuebrado:
        def detect(self, frame):
            raise RuntimeError("modelo falhou")

    pipeline = RecognitionPipeline(
        DetectorQuebrado(), FakePlateReader(None), camera_id="portaria-1", min_confidence=0.6
    )
    with pytest.raises(RuntimeError):
        pipeline.process_frame(frame(), AGORA)


def test_stream_processa_cada_frame_com_data_em_utc():
    pipeline, _ = criar([deteccao(0.9)], PlateReading("ABC1D23", 0.9))
    resultado = list(pipeline.process_stream([frame(), frame(), frame()]))
    assert len(resultado) == 3
    assert all(o.captured_at.utcoffset().total_seconds() == 0 for o in resultado)
    
class RelogioFalso:
    def __call__(self):
        return 0.0


def test_stream_com_deduplicador_reporta_a_mesma_placa_uma_so_vez():
    pipeline = RecognitionPipeline(
        FakePlateDetector([deteccao(0.9)]),
        FakePlateReader(PlateReading("ABC1D23", 0.9)),
        camera_id="portaria-1",
        min_confidence=0.6,
        deduplicator=Deduplicator(30, clock=RelogioFalso()),
    )
    assert len(list(pipeline.process_stream([frame()] * 3))) == 1
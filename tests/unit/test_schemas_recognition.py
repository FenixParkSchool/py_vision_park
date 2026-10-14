from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.recognition import BoundingBox, PlateObservation

AGORA = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)


def criar(**mudancas):
    dados = dict(
        camera_id="portaria-1",
        plate_text="ABC1D23",
        detection_confidence=0.9,
        ocr_confidence=0.8,
        captured_at=AGORA,
    )
    dados.update(mudancas)
    return PlateObservation(**dados)


def test_observacao_valida_nao_tem_bbox_por_padrao():
    obs = criar()
    assert obs.plate_text == "ABC1D23"
    assert obs.bbox is None


@pytest.mark.parametrize("campo", ["detection_confidence", "ocr_confidence"])
@pytest.mark.parametrize("valor", [-0.1, 1.1])
def test_confianca_fora_de_0_a_1_e_rejeitada(campo, valor):
    with pytest.raises(ValidationError):
        criar(**{campo: valor})


def test_data_sem_fuso_horario_e_rejeitada():
    with pytest.raises(ValidationError):
        criar(captured_at=datetime(2026, 10, 13, 12, 0))


def test_texto_da_placa_vazio_e_rejeitado():
    with pytest.raises(ValidationError):
        criar(plate_text="")


def test_bbox_com_largura_zero_e_rejeitada():
    with pytest.raises(ValidationError):
        BoundingBox(x=0, y=0, width=0, height=10)


def test_observacao_e_imutavel():
    obs = criar()
    with pytest.raises(ValidationError):
        obs.plate_text = "XYZ9K88"


def test_serializa_para_json_com_data_em_utc():
    assert "2026-10-13T12:00:00Z" in criar().model_dump_json()
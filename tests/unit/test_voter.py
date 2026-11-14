from datetime import datetime, timedelta, timezone

import pytest

from app.schemas.recognition import PlateObservation
from app.services.voter import PlateVoter

T0 = datetime(2026, 11, 14, 3, 0, tzinfo=timezone.utc)


def obs(texto, conf=0.9, det=0.8, segundo=0):
    return PlateObservation(
        camera_id="portaria-1",
        plate_text=texto,
        detection_confidence=det,
        ocr_confidence=conf,
        captured_at=T0 + timedelta(seconds=segundo),
    )


def test_tres_leituras_iguais_viram_um_envio_na_terceira():
    v = PlateVoter(max_reads=3, gap_frames=3)
    assert v.push_frame([obs("ABC1D23", segundo=0)]) == []
    assert v.push_frame([obs("ABC1D23", segundo=1)]) == []
    saida = v.push_frame([obs("ABC1D23", segundo=2)])
    assert [o.plate_text for o in saida] == ["ABC1D23"]
    assert saida[0].captured_at == T0  # momento em que o veículo foi visto pela 1ª vez


def test_voto_por_posicao_corrige_erros_isolados():
    v = PlateVoter(max_reads=3)
    v.push_frame([obs("ABC1D23")])
    v.push_frame([obs("ABC1O23")])  # erro na 5ª posição
    saida = v.push_frame([obs("ABE1D23")])  # erro na 3ª posição
    assert [o.plate_text for o in saida] == ["ABC1D23"]


def test_peso_pela_confianca_decide_a_posicao_disputada():
    v = PlateVoter(max_reads=2)
    v.push_frame([obs("ABC1D23", conf=0.6)])
    saida = v.push_frame([obs("ABC1O23", conf=0.99)])
    assert saida[0].plate_text == "ABC1O23"
    assert saida[0].ocr_confidence == pytest.approx(0.99)  # média só das que concordam


def test_depois_de_enviado_o_mesmo_veiculo_nao_reenvia():
    v = PlateVoter(max_reads=2, gap_frames=3)
    v.push_frame([obs("ABC1D23")])
    assert len(v.push_frame([obs("ABC1D23")])) == 1
    for _ in range(5):
        assert v.push_frame([obs("ABC1D23")]) == []


def test_veiculo_que_sai_antes_de_juntar_leituras_e_enviado_ao_fechar():
    v = PlateVoter(max_reads=3, gap_frames=2)
    assert v.push_frame([obs("ABC1D23")]) == []
    assert v.push_frame([]) == []  # 1 quadro sem leitura
    saida = v.push_frame([])  # 2 quadros: o veículo saiu
    assert [o.plate_text for o in saida] == ["ABC1D23"]
    assert len(v) == 0


def test_grupo_ja_enviado_fecha_sem_reenviar():
    v = PlateVoter(max_reads=1, gap_frames=1)
    assert len(v.push_frame([obs("ABC1D23")])) == 1
    assert v.push_frame([]) == []
    assert len(v) == 0


def test_dois_veiculos_no_mesmo_quadro_viram_dois_envios():
    v = PlateVoter(max_reads=2)
    v.push_frame([obs("ABC1D23"), obs("XYZ9K88")])
    saida = v.push_frame([obs("XYZ9K88"), obs("ABC1D23")])
    assert sorted(o.plate_text for o in saida) == ["ABC1D23", "XYZ9K88"]


def test_placas_com_mais_de_duas_diferencas_sao_veiculos_diferentes():
    v = PlateVoter(max_reads=3, gap_frames=3)
    v.push_frame([obs("ABC1D23")])
    v.push_frame([obs("ABC9Z99")])  # 4 posições diferentes (> 2): outro grupo
    assert len(v) == 2
    v.push_frame([obs("ABC1D23")])  # volta para o grupo igual, não abre um terceiro
    assert sorted(o.plate_text for o in v.flush()) == ["ABC1D23", "ABC9Z99"]


def test_leitura_fora_do_formato_nao_vota_e_segue_direto():
    v = PlateVoter(max_reads=3)
    invalida = obs("ABCID23")  # I no lugar de 1: formato inválido
    assert v.push_frame([invalida]) == [invalida]
    assert len(v) == 0


def test_texto_com_hifen_e_minusculo_vota_normalizado():
    v = PlateVoter(max_reads=2)
    v.push_frame([obs("abc-1d23")])
    saida = v.push_frame([obs("ABC1D23")])
    assert saida[0].plate_text == "ABC1D23"


def test_flush_envia_o_que_estava_em_votacao():
    v = PlateVoter(max_reads=3)
    v.push_frame([obs("ABC1D23", det=0.7)])
    v.push_frame([obs("ABC1D23", det=0.95)])
    saida = v.flush()
    assert [o.plate_text for o in saida] == ["ABC1D23"]
    assert saida[0].detection_confidence == 0.95
    assert len(v) == 0 and v.flush() == []


@pytest.mark.parametrize("kwargs", [{"max_reads": 0}, {"gap_frames": 0}, {"max_diff": -1}])
def test_parametros_invalidos_sao_rejeitados(kwargs):
    with pytest.raises(ValueError):
        PlateVoter(**kwargs)

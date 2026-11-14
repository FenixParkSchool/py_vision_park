"""Exporta um modelo do fast-plate-ocr ajustado (.keras) para ONNX, pronto para o serviço.

Roda no ambiente de TREINO (fast-plate-ocr[train]), com o backend TensorFlow:
  set KERAS_BACKEND=tensorflow
  <venv de treino>\\Scripts\\python -m training.exportar_onnx_fpo --model <best.keras> \\
      --plate-config models/fpo_base/cct_s_v2_global_plate_config.yaml --out models/ocr_fpo_br.onnx

Saída: `--out` (.onnx) e, ao lado, `<nome>_plate_config.yaml` (convenção lida por
`FastPlateReader.from_model`). Use no serviço com `FPO_MODEL=models/ocr_fpo_br.onnx`.

Por que não chamar `fast-plate-ocr export` direto: no Windows ele falha com PermissionError,
porque reabre pelo nome um NamedTemporaryFile(delete=True) ainda aberto. Aqui trocamos por
delete=False antes de chamar a mesma CLI (a validação Keras x ONNX da CLI continua ativa).
"""

import argparse
import functools
import shutil
import tempfile
from pathlib import Path

from app.recognition.fpo_reader import caminho_plate_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="best.keras do treino")
    ap.add_argument("--plate-config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="caminho do .onnx de saída")
    args = ap.parse_args()
    if args.out.suffix.lower() != ".onnx":
        ap.error("--out precisa terminar em .onnx")

    import fast_plate_ocr.cli.export as exp

    exp.NamedTemporaryFile = functools.partial(tempfile.NamedTemporaryFile, delete=False)

    with tempfile.TemporaryDirectory() as pasta:
        exp.export.main(
            args=[
                "--model", str(args.model),
                "--plate-config-file", str(args.plate_config),
                "--format", "onnx",
                "--save-dir", pasta,
            ],
            standalone_mode=False,
        )
        gerados = list(Path(pasta).glob("*.onnx"))
        if len(gerados) != 1:
            raise RuntimeError(f"esperava 1 .onnx exportado, encontrei {len(gerados)}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(gerados[0]), args.out)
    shutil.copyfile(args.plate_config, caminho_plate_config(args.out))
    print(f"OK: {args.out} e {caminho_plate_config(args.out)}")


if __name__ == "__main__":
    main()

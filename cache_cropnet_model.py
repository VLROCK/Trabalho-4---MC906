"""Baixa/cacheia o CropNet do TensorFlow Hub para uso offline no cluster.

Uso recomendado no login node, antes do salloc:

    python cache_cropnet_model.py \
      --cache_dir "$HOME/.cache/tfhub_modules"

Depois, dentro do job:

    export TFHUB_CACHE_DIR="$HOME/.cache/tfhub_modules"

O script apenas força o download e roda uma inferência dummy. Assim, o TF Hub
consegue reutilizar o cache quando o nó de GPU não tiver internet.
"""

from __future__ import annotations

import argparse
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache do CropNet TF Hub")
    parser.add_argument(
        "--handle",
        type=str,
        default="https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2",
        help="Handle TF Hub do CropNet.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Diretório de cache do TensorFlow Hub. Ex.: $HOME/.cache/tfhub_modules",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        os.environ["TFHUB_CACHE_DIR"] = args.cache_dir
        print(f"TFHUB_CACHE_DIR={args.cache_dir}")

    import numpy as np
    import tensorflow as tf
    import tensorflow_hub as hub

    print(f"Carregando CropNet de: {args.handle}")
    classifier = hub.KerasLayer(args.handle, trainable=False)

    dummy = tf.convert_to_tensor(np.zeros((1, 224, 224, 3), dtype=np.float32))
    probs = classifier(dummy).numpy()

    print("Download/cache concluído.")
    print(f"Shape de saída: {probs.shape}")
    print("Classes CropNet: ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy', 'unknown']")
    print("Para usar offline, mantenha TFHUB_CACHE_DIR apontando para o mesmo cache.")


if __name__ == "__main__":
    main()

import runpod

import sys
import os
from pathlib import Path
from esm_embedding import embed_sequences


def handler(event):    
    # base_path = r"D:\Git_Clone\GeneExp"
    # sys.path.append(str(Path(base_path)))
    print(f"Worker Start")

    translations = event['input'].get('translations')

    if not translations:
        raise(ValueError(f"No translations provided: {event['input']}"))

    embeddings = embed_sequences(translations)
    #########################################
        

    return {
        "status": "success",
        "message": f"Generated embeddings for {len(embeddings)} sequences.",
        "embeddings": embeddings
    }


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})

# Memory-Augmented Narrative Modeling for Visual Story Continuation

Purpose
- Preserve long-range story consistency by maintaining a persistent memory across timesteps while predicting the next frame's text.

Method overview
- Text input: DistilBERT tokenizer with padded context windows.
- Image input: frozen ResNet18 embeddings for context and target frames.
- Context encoder: masked-mean text pooling per frame + GRU over fused text-image context.
- Memory model: learnable slots with attention-based read and gated write per timestep.
- Decoder: GRU language model conditioned on the context summary for next-frame text prediction.
- Diagnostics: entity hallucination rate based on `<gdo ...>` tags.

Repository layout
- `src/dataloader.py`: frame parsing, entity extraction, windowed dataset, and tokenizer collation.
- `src/encoders_image.py`: ResNet18 embedding utilities.
- `src/encoders_text.py`: tokenizer loading helpers.
- `src/model_baseline.py`: baseline next-text model.
- `src/memory_module.py`: memory slots with read/write operations.
- `src/model_memory.py`: memory-augmented next-text model.
- `src/train.py`: training loop, decoding, and hallucination evaluation.
- `src/eval_longseq.py`: evaluation helpers (loss, perplexity, hallucination).

Quickstart
```bash
cd "Vishalini"

# Train memory model
python src/train.py --model memory

# Train baseline
python src/train.py --model baseline
```

Notes
- The entity hallucination metric expects entity tags in the text like `<gdo char1>NAME</gdo>`.

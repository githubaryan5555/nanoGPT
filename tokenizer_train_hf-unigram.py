#STEP 2/4 TRAINING HUGGINGFACE UNIGRAM TOKENIZER (65536 VOCAB 2.9~~ GB DATA)
import os
import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tqdm import tqdm

# =========================
# CONFIG
# =========================

DATA_DIR = "data/tokenizer_data"
OUTPUT_TOKENIZER = "unigram_tokenizer.json"
VOCAB_SIZE = 65536
BATCH_SIZE = 1024
LOG_EVERY = 32  # batches

SPECIAL_TOKENS = ["[UNK]", "[BOS]", "[EOS]"]

# =========================
# COLLECT FILES
# =========================

parquet_files = sorted(
    os.path.join(DATA_DIR, f)
    for f in os.listdir(DATA_DIR)
    if f.endswith(".parquet")
)

if not parquet_files:
    raise RuntimeError("❌ No parquet files found")

print(f"✅ Found {len(parquet_files)} parquet shards")

# =========================
# STREAMING ITERATOR
# =========================

def text_iterator(files):
    total_yielded = 0

    for shard_idx, path in enumerate(files, 1):
        print(f"\n📦 [{shard_idx}/{len(files)}] Streaming shard: {os.path.basename(path)}")

        pf = pq.ParquetFile(path)
        num_batches = pf.metadata.num_row_groups

        with tqdm(total=num_batches, desc="   batches", leave=False) as pbar:
            for batch_idx, batch in enumerate(
                pf.iter_batches(columns=["text"], batch_size=BATCH_SIZE), 1
            ):
                texts = batch.column(0).to_pylist()

                for t in texts:
                    if t:
                        yield t.replace("\n", " ")
                        total_yielded += 1

                if batch_idx % LOG_EVERY == 0:
                    print(f"      ↳ yielded ~{total_yielded:,} texts")

                pbar.update(1)

        print(f"✅ Finished shard {shard_idx}")

    print(f"\n🎯 Total texts yielded: {total_yielded:,}")

# =========================
# TOKENIZER SETUP
# =========================

tokenizer = Tokenizer(models.Unigram())

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC()
])

tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.UnigramTrainer(
    vocab_size=VOCAB_SIZE,
    unk_token="[UNK]",
    special_tokens=SPECIAL_TOKENS
)

# =========================
# TRAIN
# =========================

print("\n🔥 Training Unigram tokenizer (this WILL take time)...")

tokenizer.train_from_iterator(
    text_iterator(parquet_files),
    trainer=trainer
)

# =========================
# SAVE
# =========================

tokenizer.save(OUTPUT_TOKENIZER)
print(f"\n🎉 Tokenizer saved to {OUTPUT_TOKENIZER}")

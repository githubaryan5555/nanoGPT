# STEP 1/4 , DATA COLLECTING.
# STEP 1/4 , DATA COLLECTING.
import os
import requests
import pyarrow.parquet as pq

# =========================
# USER CONFIG
# =========================

DATASET = "Geralt-Targaryen/openwebtext2"

# how many shards you want
NUM_TOKENIZER_SHARDS = 3
NUM_MODEL_SHARDS     = 10

BASE_DIR = "data"
TOK_DIR  = os.path.join(BASE_DIR, "tokenizer_data")
MOD_DIR  = os.path.join(BASE_DIR, "model_data")

os.makedirs(TOK_DIR, exist_ok=True)
os.makedirs(MOD_DIR, exist_ok=True)

# Control variables
DOWNLOAD_TOKENIZER = True   # set False to skip tokenizer shards
DOWNLOAD_MODEL     = True   # set False to skip model shards
INSPECT_PARQUETS   = True   # set False to skip printing first/last 100 sentences

# =========================
# STEP 1: DETECT REAL SHARDS
# =========================

print("🔍 Fetching dataset file list...")

api_url = f"https://huggingface.co/api/datasets/{DATASET}"
resp = requests.get(api_url)
resp.raise_for_status()

files = resp.json()["siblings"]

parquet_files = sorted(
    f["rfilename"]
    for f in files
    if f["rfilename"].endswith(".parquet")
)

TOTAL_SHARDS = len(parquet_files)

if TOTAL_SHARDS == 0:
    raise RuntimeError("❌ No parquet shards found. Dataset layout changed.")

print(f"✅ Total parquet shards found: {TOTAL_SHARDS}")
print("First 5 shard names:")
for s in parquet_files[:5]:
    print(" ", s)
print("Last 5 shard names:")
for s in parquet_files[-5:]:
    print(" ", s)

# =========================
# STEP 2: DOWNLOAD FUNCTION
# =========================

def download_shards(shard_list, out_dir):
    for shard in shard_list:
        fname = os.path.basename(shard)
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/main/{shard}"
        out = os.path.join(out_dir, fname)

        if os.path.exists(out):
            print(f"✔ Already exists: {fname}")
            continue

        print(f"📥 Downloading {fname}")
        r = requests.get(url, stream=True)
        r.raise_for_status()

        with open(out, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

        size_mb = os.path.getsize(out) / (1024 * 1024)
        print(f"✅ Saved {fname} ({size_mb:.2f} MB)")

# =========================
# STEP 3: SELECT SHARDS
# =========================

tokenizer_shards = parquet_files[:NUM_TOKENIZER_SHARDS]
model_shards     = parquet_files[:NUM_MODEL_SHARDS]

print(f"\n🧠 Tokenizer shards: {len(tokenizer_shards)}")
print(f"🤖 Model shards: {len(model_shards)}")

if DOWNLOAD_TOKENIZER:
    download_shards(tokenizer_shards, TOK_DIR)
else:
    print("⚠ Skipping tokenizer shard download")

if DOWNLOAD_MODEL:
    download_shards(model_shards, MOD_DIR)
else:
    print("⚠ Skipping model shard download")

# =========================
# STEP 4: INSPECT DATA
# =========================

def inspect_folder(folder, label, print_text=True):
    print(f"\n===== {label.upper()} INSPECTION =====")

    files = sorted(os.listdir(folder))
    if not files:
        print("No files to inspect")
        return

    if print_text:
        all_text = []
        for f in files:
            path = os.path.join(folder, f)
            table = pq.read_table(path, columns=["text"])
            texts = table.column("text").to_pylist()
            all_text.extend(texts)

        print("\n📌 FIRST 100 SENTENCES:")
        for s in all_text[:100]:
            print(s.replace("\n", " ")[:300])

        print("\n📌 LAST 100 SENTENCES:")
        for s in all_text[-100:]:
            print(s.replace("\n", " ")[:300])

    print("\n📦 FILE SIZES:")
    for f in files:
        size_mb = os.path.getsize(os.path.join(folder, f)) / (1024 * 1024)
        print(f"{f}: {size_mb:.2f} MB")

if DOWNLOAD_TOKENIZER:
    inspect_folder(TOK_DIR, "tokenizer", print_text=INSPECT_PARQUETS)
if DOWNLOAD_MODEL:
    inspect_folder(MOD_DIR, "model", print_text=INSPECT_PARQUETS)

print("\n🎉 DONE — OpenWebText2 shards downloaded and verified.")

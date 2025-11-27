#!/usr/bin/env python3
"""
Run photomosaic pipeline on a directory of tile icons.

Features:
- Reads all PNG/JPG files from a folder (recursively) as tiles.
- Uses provided source image (file) to be mosaic target.
- Computes 4x4x3 Lab features (48-d) for tiles and source blocks.
- Optionally caches tile features to disk (NPZ) to speed repeated runs.
- PCA -> ANN (Annoy if available, FAISS if available, fallback to sklearn NN)
- Exact re-rank among top-K candidates; simple redundancy rule applied.
- Intermittent console prints so you can see progress.
- Produces CSV / PNG / ZIP output and a mosaic preview image.

Usage:
    python mosaic_ann.py --tiles-dir ./icons --source image.png --block-size 40

Dependencies:
    numpy, pillow, scikit-image (optional but recommended), scikit-learn, pandas, matplotlib
    Annoy and/or faiss optional; script will fallback to sklearn NearestNeighbors.
"""

import argparse
import time
import zipfile
import warnings
import tracemalloc
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# optional imports: prefer skimage.color for Lab conversion
try:
    from skimage import color as skcolor
    HAVE_SKIMAGE = True
except Exception:
    HAVE_SKIMAGE = False

try:
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors
    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

# optional ANN backends
HAVE_FAISS = False
HAVE_ANNOY = False
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    try:
        from annoy import AnnoyIndex
        HAVE_ANNOY = True
    except Exception:
        HAVE_ANNOY = False

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Utility functions
# ---------------------------
def image_to_lab_arr(img):
    """Return HxWx3 Lab float32 array. If skimage unavailable, use simple RGB->approx Lab."""
    arr = np.asarray(img).astype(np.float32)
    if HAVE_SKIMAGE:
        arr = arr / 255.0
        lab = skcolor.rgb2lab(arr)
        return lab.astype(np.float32)
    # approximate conversion (not perceptually exact but ok as fallback)
    r = arr[:, :, 0] / 255.0 * 100.0
    g = arr[:, :, 1] / 255.0 * 100.0
    b = arr[:, :, 2] / 255.0 * 100.0
    lab = np.stack([r, g, b], axis=2)
    return lab.astype(np.float32)

def resize_and_crop_to_block(img, block_size):
    w, h = img.size
    scale = max(block_size / w, block_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img2 = img.resize((new_w, new_h), Image.LANCZOS)
    left = max(0, (new_w - block_size) // 2)
    top = max(0, (new_h - block_size) // 2)
    return img2.crop((left, top, left + block_size, top + block_size))

# ---------------------------
# Vectorized 4x4 Lab feature computation
# ---------------------------
def compute_4x4_lab_feature_vectorized(img, block_size, subgrid=4):
    """Return 48-d vector: mean Lab for each subgrid block, vectorized."""
    if img.size != (block_size, block_size):
        img = resize_and_crop_to_block(img, block_size)
    lab = image_to_lab_arr(img)  # HxWx3

    h, w, c = lab.shape
    gh, gw = h // subgrid, w // subgrid

    # reshape into (subgrid, gh, subgrid, gw, 3)
    lab_blocks = lab[:gh*subgrid, :gw*subgrid].reshape(subgrid, gh, subgrid, gw, 3)
    feats = lab_blocks.mean(axis=(1,3)).reshape(-1)
    return feats.astype(np.float32), img, lab  # return resized PIL and Lab for caching

# ---------------------------
# Parallelized feature computation
# ---------------------------
def compute_tile_features(tile_paths, block_size=60, subgrid=4, max_workers=8, cache_features=True):
    """
    Compute 48-d features for all tiles in parallel, optionally caching resized PIL and Lab arrays.
    Returns: feats_array (N x 48), tiles_data_list
    """
    feats_list = []
    tiles_data = []

    def process_tile(p):
        try:
            pil_full = Image.open(p).convert('RGB')
            feats, pil_resized, lab_arr = compute_4x4_lab_feature_vectorized(pil_full, block_size, subgrid=subgrid)
            return (str(p), feats, pil_resized, pil_full, lab_arr)
        except Exception as e:
            print(f"Skipped {p} due to error: {e}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(process_tile, p): p for p in tile_paths}
        for i, fut in enumerate(as_completed(futures)):
            res = fut.result()
            if res is None:
                continue
            path_str, feats, pil_resized, pil_full, lab_arr = res
            feats_list.append(feats)
            tiles_data.append({'path': path_str,
                               'pil': pil_resized,
                               'pil_full': pil_full,
                               'lab': lab_arr})
            if (i + 1) % 100 == 0 or i == 0:
                print(f"  processed {i+1}/{len(tile_paths)} tiles", flush=True)

    feats_array = np.stack(feats_list, axis=0).astype(np.float32)
    return feats_array, tiles_data


# ---------------------------
# ANN helpers
# ---------------------------
def build_ann_index(feats_low, backend_preference=('faiss','annoy','sklearn')):
    """
    Build ANN index. Returns (backend_name, index_object, meta)
    meta is dict with possibly useful objects (e.g., faiss_pca wrapper); for sklearn it's the same index.
    """
    n, d = feats_low.shape
    if HAVE_FAISS and 'faiss' in backend_preference:
        # simple FAISS IndexFlatL2 on reduced dims (exact but fast in C)
        index = faiss.IndexFlatL2(d)
        index.add(feats_low.astype(np.float32))
        return 'faiss', index, {}
    if HAVE_ANNOY and 'annoy' in backend_preference:
        from annoy import AnnoyIndex
        idx = AnnoyIndex(d, 'euclidean')
        for i, v in enumerate(feats_low):
            idx.add_item(i, v.tolist())
            if (i + 1) % 5000 == 0:
                print(f"  [Annoy build] added {i+1}/{n} vectors", flush=True)
        idx.build(50)
        return 'annoy', idx, {}
    if HAVE_SKLEARN and 'sklearn' in backend_preference:
        nn = NearestNeighbors(n_neighbors=10, algorithm='auto', metric='euclidean', n_jobs=-1)
        nn.fit(feats_low)
        return 'sklearn', nn, {}
    raise RuntimeError("No ANN backend available (faiss/annoy/sklearn). Install one and retry.")

def ann_query(backend_name, index_obj, query_low, topk=20):
    """
    Query ANN index. Returns (D, I) where D shape (M, topk) and I shape (M, topk).
    """
    if backend_name == 'faiss':
        q = query_low.astype(np.float32)
        D, I = index_obj.search(q, topk)
        return D, I
    if backend_name == 'annoy':
        D_list = []
        I_list = []
        for i, v in enumerate(query_low):
            ids, dists = index_obj.get_nns_by_vector(v.tolist(), topk, include_distances=True)
            # Annoy can return less than requested for small DBs; pad if needed
            if len(ids) < topk:
                ids = ids + [-1] * (topk - len(ids))
                dists = dists + [float('inf')] * (topk - len(dists))
            I_list.append(ids)
            D_list.append(dists)
            if (i + 1) % 100 == 0:
                print(f"  [Annoy query] processed {i+1}/{len(query_low)} queries", flush=True)
        return np.array(D_list, dtype=np.float32), np.array(I_list, dtype=np.int32)
    if backend_name == 'sklearn':
        D, I = index_obj.kneighbors(query_low, n_neighbors=min(topk, index_obj._fit_X.shape[0]))
        return D, I
    raise RuntimeError("Unsupported ANN backend")

# --------------------------
# Optimized block-to-tile assignment
# --------------------------
def assign_tiles_to_blocks(blocks_feats48, feats48, I, nx, ny, top_k=20):
    """
    Single-threaded, vectorized assignment that enforces global uniqueness.
    - blocks_feats48: (num_blocks, 48)
    - feats48: (num_tiles, 48)
    - I: (num_blocks, top_k) ANN candidate indices per block (ints; -1 padded allowed)
    - nx, ny: grid shape (used only for shape consistency)
    Returns: selection_list (num_blocks,) of tile indices (int)
    Raises RuntimeError if num_blocks > num_tiles (cannot assign unique tiles).
    """
    num_blocks = int(blocks_feats48.shape[0])
    num_tiles = int(feats48.shape[0])

    if num_blocks > num_tiles:
        raise RuntimeError(f"Cannot assign unique tiles: {num_blocks} blocks but only {num_tiles} tiles.")

    selection_list = -np.ones(num_blocks, dtype=int)
    used_mask = np.zeros(num_tiles, dtype=bool)  # True = already used

    # Optional heuristic: order blocks by how "ambiguous" they are (blocks with fewer distinct ANN candidates first).
    # This reduces the chance of exhausting good matches for difficult blocks.
    # Build candidate counts (ignoring -1 and duplicates)
    candidate_sets = [np.unique(I[i][I[i] >= 0]).tolist() for i in range(num_blocks)]
    block_order = np.argsort([len(s) for s in candidate_sets])  # ascending

    for idx in block_order:
        # vectorized candidate handling
        raw_cands = I[idx]
        raw_cands = raw_cands[raw_cands >= 0].astype(int)
        # filter to unused
        if raw_cands.size:
            mask_unused = np.logical_not(used_mask[raw_cands])
            cands = raw_cands[mask_unused]
        else:
            cands = np.array([], dtype=int)

        if cands.size == 0:
            # fallback: pick nearest among *all remaining* unused tiles
            remaining_idx = np.nonzero(~used_mask)[0]
            if remaining_idx.size == 0:
                # should not happen due to check above
                raise RuntimeError("Ran out of unused tiles unexpectedly.")
            # compute distances to all remaining (vectorized)
            diffs = feats48[remaining_idx] - blocks_feats48[idx]  # (R,48)
            scores = np.einsum('ij,ij->i', diffs, diffs)         # faster dot over axis
            best_pos = int(np.argmin(scores))
            best_tile = int(remaining_idx[best_pos])
        else:
            # rerank only ANN candidates (vectorized)
            candidate_feats = feats48[cands]                    # (k,48)
            diffs = candidate_feats - blocks_feats48[idx]       # (k,48)
            scores = np.einsum('ij,ij->i', diffs, diffs)
            best_pos = int(np.argmin(scores))
            best_tile = int(cands[best_pos])

        # assign and mark used
        selection_list[idx] = best_tile
        used_mask[best_tile] = True

    # Now reorder selection_list back to natural block order if block_order permuted it
    # block_order maps ordered_index -> original_index, we filled selection_list by original index (idx),
    # so no reorder needed. If you used a separate output buffer keyed by order position, you'd reorder here.
    return selection_list


# ---------------------------
# Main pipeline
# ---------------------------
def run_pipeline(tiles_dir, source_path, out_dir, block_size=60, pca_dim=12, top_k=20, cache_features=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    tracemalloc.start()

    # 1) gather tiles
    tiles_dir = Path(tiles_dir)
    candidates = [p for p in tiles_dir.rglob('*') if p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tga')]
    if not candidates:
        raise RuntimeError("No tile images found in directory.")
    print(f"[1/8] Found {len(candidates)} image files under {tiles_dir}", flush=True)

    # 2) determine source image and remove from tile list if present
    source_path = Path(source_path)
    if not source_path.exists():
        raise RuntimeError(f"Source image not found: {source_path}")
    tiles = []
    tile_paths = []
    for p in candidates:
        # exclude source if same filename & same directory
        if p.resolve() == source_path.resolve():
            continue
        tile_paths.append(p)
    print(f"[2/8] Using source {source_path.name}; {len(tile_paths)} tiles after excluding source.", flush=True)

    # 3) load or compute cached features
    cache_file = out_dir / 'tile_features_cache.npz'
    feats48 = None
    tiles_data = []

    if cache_features and cache_file.exists():
        try:
            print("[3/8] Loading cached tile features...", flush=True)
            npz = np.load(cache_file, allow_pickle=True)
            feats48 = npz['feats48']
            paths_cached = npz['paths'].tolist()
            # Check if block_size matches cached value
            cached_block_size = npz['block_size'] if 'block_size' in npz else None
            if cached_block_size != block_size:
              print(f"Block size changed (cached={cached_block_size}, current={block_size}) -> recomputing tile features")
              feats48 = None  # force recomputation

            # Only use cached entries that still exist
            kept = []
            for p in paths_cached:
                pth = Path(p)
                if pth.exists():
                    kept.append(pth)
            # If cache matches current tile_paths exactly, reuse
            if set(map(str, kept)) == set(map(str, tile_paths)):
                tiles_data = [{'path': str(p), 'pil': None, 'pil_full': None, 'lab': None} for p in kept]
                print(f"  cache valid: loaded {len(kept)} tile features", flush=True)
            else:
                print("  cache contents don't match current tile set -> recomputing", flush=True)
                feats48 = None
        except Exception as e:
            print("  failed to load cache, recomputing. error:", e, flush=True)
            feats48 = None

    # Compute features if cache not loaded
    if feats48 is None:
        print("[3/8] Computing 48-d features for tiles (4x4 Lab means) in parallel...", flush=True)
        feats48, tiles_data = compute_tile_features(tile_paths,
                                                    block_size=block_size,
                                                    subgrid=4,
                                                    max_workers=8)

        # save cache: store only features and paths
        if cache_features:
            print("[3/8] Saving tile features cache...", flush=True)
            np.savez_compressed(cache_file,
                                feats48=feats48,
                                paths=np.array([t['path'] for t in tiles_data]),
                                block_size=block_size)


    print(f"[3/8] Completed tile features: {feats48.shape[0]} tiles -> feats shape {feats48.shape}", flush=True)

    # 4) load and partition source image
    print("[4/8] Loading and partitioning source image...", flush=True)
    src = Image.open(source_path).convert('RGB')
    w, h = src.size
    # pad to multiples of block_size
    pad_w = ((w + block_size - 1) // block_size) * block_size - w
    pad_h = ((h + block_size - 1) // block_size) * block_size - h
    if pad_w or pad_h:
        new_img = Image.new('RGB', (w + pad_w, h + pad_h))
        new_img.paste(src, (0, 0))
        src = new_img
    nx = src.size[0] // block_size
    ny = src.size[1] // block_size
    blocks = []
    total_blocks = nx * ny
    for by in range(ny):
        for bx in range(nx):
            block = src.crop((bx*block_size, by*block_size, (bx+1)*block_size, (by+1)*block_size))
            blocks.append({'pil': block, 'grid': (bx, by)})
    print(f"  partitioned source into {nx} x {ny} = {total_blocks} blocks", flush=True)

    # 5) compute block features
    print("[5/8] Computing features for blocks...", flush=True)
    blocks_feats_list = []
    for i, b in enumerate(blocks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  computing block features {i+1}/{total_blocks}", flush=True)
        feats, _, _ = compute_4x4_lab_feature_vectorized(b['pil'], block_size, subgrid=4)
        blocks_feats_list.append(feats)

    blocks_feats48 = np.stack(blocks_feats_list, axis=0).astype(np.float32)
    print("[5/8] Block features done", flush=True)

    # 6) PCA fit on tile features (fit once), then transform both tile and block features
    print(f"[6/8] Fitting PCA (dim={pca_dim}) on {feats48.shape[0]} tile features...", flush=True)
    if not HAVE_SKLEARN:
        raise RuntimeError("scikit-learn is required for PCA / NearestNeighbors fallback. Install scikit-learn.")
    pca = PCA(n_components=min(pca_dim, feats48.shape[1]))
    t0 = time.time()
    pca.fit(feats48)
    feats_low = pca.transform(feats48)
    blocks_low = pca.transform(blocks_feats48)
    print(f"  PCA fit+transform done in {time.time() - t0:.3f}s", flush=True)

    # 7) Build ANN index
    print("[7/8] Building ANN index...", flush=True)
    backend_name, index_obj, meta = build_ann_index(feats_low, backend_preference=('annoy','faiss','sklearn'))
    print(f"  ANN backend chosen: {backend_name}", flush=True)

    # 8) Query ANN to get top-K candidates for every block
    print(f"[8/8] Querying ANN for top-{top_k} candidates per block...", flush=True)
    D, I = ann_query(backend_name, index_obj, blocks_low, topk=top_k)
    print("  ANN query complete", flush=True)

    # ---------------------------
    # Assign tiles to blocks without repetition
    # ---------------------------
    print("[final] Assigning tiles to blocks without repeats", flush=True)

    selection_list = assign_tiles_to_blocks(blocks_feats48, feats48, I, nx, ny,
                                        top_k=top_k)
                                        # max_workers=8)


    # ---------------------------
    # Assemble final mosaic preview image with per-tile luminance adjustment
    # ---------------------------
    print("Assembling mosaic preview image with luminance adjustment...", flush=True)
    # assume all tiles have same size: tx, ty
    # safe sample tile size (works with or without cache)
    first_tile_path = tiles_data[0].get('path')
    if first_tile_path is None:
        raise RuntimeError("tiles_data has no path entries")

    # lazy-load a full-res sample if needed (don't rely on pil_full)
    if tiles_data[0].get('pil_full') is None:
        sample_img = Image.open(first_tile_path).convert('RGB')
        tiles_data[0]['pil_full'] = sample_img  # cache the sample for reuse
    else:
        sample_img = tiles_data[0]['pil_full']

    tx, ty = sample_img.size

    out_w = nx * tx
    out_h = ny * ty
    mosaic = Image.new('RGB', (out_w, out_h))

    for i, block in enumerate(blocks):
      bx, by = block['grid']
      tile_idx = selection_list[i]

      tile_entry = tiles_data[tile_idx]

      # Load full-res tile once and cache it
      if tile_entry.get('pil_full') is None:
          tile_entry['pil_full'] = Image.open(tile_entry['path']).convert('RGB')


      # Paste adjusted tile into mosaic
      mosaic.paste(tile_entry['pil_full'], (bx*tx, by*ty))


      if (i + 1) % 50 == 0 or i == 0:
          print(f"  pasted block {i+1}/{len(blocks)}", flush=True)


    mosaic_path = out_dir / 'mosaic_result.jpg'
    mosaic.save(mosaic_path, quality=85)


    # Save results summary
    mem_cur, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results = {
        'num_tiles': feats48.shape[0],
        'num_blocks': blocks_feats48.shape[0],
        'ann_backend': backend_name,
        'pca_dim': feats_low.shape[1],
        'top_k': top_k,
        'time_total_s': round(time.time() - start_time, 4),
        'mem_current_bytes': int(mem_cur),
        'mem_peak_bytes': int(mem_peak),
        'mosaic_path': str(mosaic_path)
    }
    df = pd.DataFrame([results])
    csv_path = out_dir / 'profile_results.csv'
    png_path = out_dir / 'profile_results.png'
    zip_path = out_dir / 'profile_results.zip'
    df.to_csv(csv_path, index=False)
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    fig.savefig(png_path, dpi=150, bbox_inches='tight')

    # pack CSV/PNG/mosaic and a few sample tiles into ZIP
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname=csv_path.name)
        zf.write(png_path, arcname=png_path.name)
        zf.write(mosaic_path, arcname=mosaic_path.name)
        sample_dir = out_dir / 'tile_samples'
        sample_dir.mkdir(exist_ok=True)
        for i, t in enumerate(tiles_data[:50]):
            p = sample_dir / f"tile_{i}.png"
            # ensure we have a PIL image to save
            pil = t['pil'] if t.get('pil') is not None else Image.open(t['path']).convert('RGB')
            pil.save(p)
            zf.write(p, arcname=f"tile_samples/{p.name}")

    print("Profile complete. Outputs written to:", out_dir, flush=True)
    print("  - CSV:", csv_path, flush=True)
    print("  - PNG:", png_path, flush=True)
    print("  - ZIP:", zip_path, flush=True)
    print("  - Mosaic preview:", mosaic_path, flush=True)

# ---------------------------
# CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tiles-dir", required=True, help="Directory containing tile icons (PNG/JPG).")
    p.add_argument("--source", required=True, help="Path to source image to mosaic (will be partitioned).")
    p.add_argument("--out-dir", default="mosaic_ann_out", help="Output directory.")
    p.add_argument("--block-size", type=int, default=40, help="Tile/block size in pixels (default 40).")
    p.add_argument("--pca-dim", type=int, default=12, help="PCA reduced dimension (default 12).")
    p.add_argument("--top-k", type=int, default=20, help="Top-K ANN candidates per block (default 20).")
    p.add_argument("--no-cache", action="store_true", help="Disable tile feature caching.")
    args = p.parse_args()

    run_pipeline(args.tiles_dir, args.source, args.out_dir,
                 block_size=args.block_size,
                 pca_dim=args.pca_dim,
                 top_k=args.top_k,
                 cache_features=(not args.no_cache))

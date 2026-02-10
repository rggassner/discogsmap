#!venv/bin/python3
import json
from collections import Counter
import hashlib
from pathlib import Path
import time
import math
import warnings
from io import BytesIO
import random
import requests
import numpy as np
import discogs_client
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import cosine_distances
import umap
from PIL import Image
from PIL import ImageDraw
import markdown

warnings.filterwarnings(
    "ignore",
    message="n_jobs value 1 overridden to 1 by setting random_state",
    category=UserWarning,
)

DISCOGS_USER_TOKEN = "...lhq"
DISCOGS_USERNAME = "...ma"

TAG_WEIGHTS = {
    "genres": 1,
    "styles": 2,
    "artists": 3,
}

CACHE_FILE = Path("discogs_album_cache.json")
COVER_CACHE_DIR = Path("cover_cache")
COVER_CACHE_DIR.mkdir(exist_ok=True)

GAP = 1
FOLDERS = ["All"]
#FOLDERS = ["Electronic"]

OUTPUT_IMAGE = "album_map.png"
OUTPUT_PLAYLIST = "album_playlist.md"

CANVAS_SIZE = 4000
COVER_SIZE = 140
BACKGROUND_COLOR = (10, 10, 10)

#RANDOM_SEED = 42
RANDOM_SEED =  int(time.time()) % 1000000
LAST_CALL = 0
RATE_LIMIT_INTERVAL = 1


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def rate_limit(min_interval=RATE_LIMIT_INTERVAL):
    global LAST_CALL
    now = time.time()
    delta = now - LAST_CALL
    if delta < min_interval:
        time.sleep(min_interval - delta)
    LAST_CALL = time.time()


def cover_cache_path(url, size):
    h = hashlib.sha1(url.encode("utf-8")).hexdigest()
    return COVER_CACHE_DIR / f"{h}_{size}.jpg"


def init_discogs():
    return discogs_client.Client(
        "AlbumMap/0.1",
        user_token=DISCOGS_USER_TOKEN
    )


def get_user_folders(client):
    user = client.user(DISCOGS_USERNAME)
    return {f.name: f for f in user.collection_folders}


def collect_releases(client, sleep=1.0):
    folders = get_user_folders(client)

    if FOLDERS:
        selected = [folders[name] for name in FOLDERS if name in folders]
    else:
        selected = list(folders.values())

    items = []

    for folder in selected:
        page = 1
        while True:
            print(f"Fetching folder '{folder.name}', page {page}")
            releases_page = folder.releases.page(page)
            if not releases_page:
                break

            items.extend(releases_page)

            if len(releases_page) < 50:
                break

            page += 1
            time.sleep(sleep)

    return items


def extract_features(items, tag_weights=TAG_WEIGHTS):
    cache = load_cache()
    albums = []
    feature_dicts = []
    cache_updated = False

    for item in items:
        r = item.release
        rid = str(r.id)

        if rid in cache:
            album = cache[rid]
            features = Counter()

            for g in album.get("genres", []):
                features[f"genre:{g}"] += tag_weights["genres"]

            for s in album.get("styles", []):
                features[f"style:{s}"] += tag_weights["styles"]

            for a in album.get("artists", "").split(", "):
                features[f"artist:{a}"] += tag_weights["artists"]

            if features:
                albums.append(album)
                feature_dicts.append(dict(features))
            continue

        rate_limit()
        try:
            r.refresh()
        except Exception:
            continue

        genres = r.genres or []
        styles = r.styles or []
        artists = [a.name.strip() for a in r.artists or [] if a and a.name]

        features = Counter()
        for g in genres:
            features[f"genre:{g}"] += tag_weights["genres"]
        for s in styles:
            features[f"style:{s}"] += tag_weights["styles"]
        for a in artists:
            features[f"artist:{a}"] += tag_weights["artists"]

        cover_url = None
        if r.images:
            cover_url = r.images[0].get("uri") or r.images[0].get("resource_url")

        album = {
            "id": r.id,
            "title": r.title,
            "artists": ", ".join(artists),
            "genres": genres,
            "styles": styles,
            "cover_url": cover_url,
        }
        if album["cover_url"]:
            download_cover(album["cover_url"], COVER_SIZE)
        cache[rid] = album
        cache_updated = True
        if not features:
            continue
        albums.append(album)
        feature_dicts.append(dict(features))

    if cache_updated:
        save_cache(cache)

    return albums, feature_dicts


def build_feature_matrix(tags):
    """
    Build a dense feature matrix from album tag dictionaries.

    This function converts a list of per-album feature dictionaries
    (e.g. genres, styles, artists with weighted counts) into a numerical
    matrix suitable for machine learning and distance calculations.

    Internally, it uses scikit-learn's DictVectorizer to one-hot encode
    feature keys and apply their associated weights, producing a dense
    matrix where each row represents an album and each column represents
    a unique tag.

    Args:
        tags (list[dict]): List of feature dictionaries, one per album.

    Returns:
        np.ndarray: Dense 2D array of shape (n_items, n_features) containing
        the vectorized album features.
    """
    vectorizer = DictVectorizer(sparse=False)
    matrix = vectorizer.fit_transform(tags)
    return matrix


def convert_md_to_html(md_path, html_path=None):
    """
    Convert a Markdown file into a standalone HTML file.

    This function reads a Markdown document from disk, renders it to HTML
    using the Python Markdown library, and writes the resulting HTML to a
    file. If no output path is provided, the HTML file is created alongside
    the Markdown file using the same base name.

    The conversion supports common GitHub-style Markdown features via the
    "extra" extension.

    Args:
        md_path (str | Path): Path to the input Markdown (.md) file.
        html_path (str | Path | None, optional): Path to the output HTML file.
            If None, the output path is derived by replacing the .md suffix
            with .html.

    Returns:
        None
    """
    if html_path is None:
        html_path = md_path.with_suffix(".html") if isinstance(md_path, Path) else md_path.replace(".md", ".html")

    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()

    html = markdown.markdown(text, extensions=["extra"])

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Converted {md_path} to {html_path}")

def generate_album_playlist_global_md(albums, matrix, seed_idx, output_path):
    """
    Generate a globally sorted album playlist by distance from a seed album.

    This function computes the cosine distance from a chosen seed album to all
    other albums in feature space and produces a playlist ordered by increasing
    distance. Unlike the greedy nearest-neighbor approach, this method reflects
    global similarity to the seed rather than local continuity between
    consecutive albums.

    The resulting playlist is saved as a Markdown file and includes:
    - The seed album
    - Albums ordered by global cosine distance
    - Per-album distance values relative to the seed
    - Embedded album cover images when available in the local cover cache

    Args:
        albums (list[dict]): List of album metadata dictionaries, each
            containing at least "artists", "title", and optionally "cover_url".
        matrix (np.ndarray): Feature matrix representing album descriptors
            (genres, styles, artists, etc.).
        seed_idx (int): Index of the seed album used as the global reference
            point.
        output_path (str | Path): Path to the output Markdown file.

    Returns:
        list[int]: Ordered list of album indices sorted by increasing distance
        from the seed album.
    """
    distances = cosine_distances(
        matrix[seed_idx].reshape(1, -1),
        matrix
    )[0]

    order = np.argsort(distances)
    seed_album = albums[seed_idx]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Album Playlist — Global Distance\n\n")
        f.write(f"Seed album:\n**{seed_album['artists']} – {seed_album['title']}**\n\n")
        f.write("## Playlist\n\n")

        for i, idx in enumerate(order, start=1):
            album = albums[idx]

            # Embed cached cover image if it exists
            cover_md = ""
            if album.get("cover_url"):
                cover_path = cover_cache_path(album['cover_url'], COVER_SIZE)
                if cover_path.exists():
                    cover_md = f"![Cover]({cover_path.as_posix()}) "

            f.write(
                f"{i}. {cover_md}**{album['artists']} – {album['title']}** "
                f"(distance from seed: {distances[idx]:.3f})\n"
            )

    print(f"Saved global playlist to {output_path}")
    return list(order)


def generate_album_playlist_greedy_md(albums, matrix, seed_idx, output_path): #pylint: disable=too-many-locals
    """
    Generate a greedy nearest-neighbor album playlist and save it as Markdown.

    Starting from a seed album, this function builds a playlist by repeatedly
    selecting the most similar unvisited album based on cosine distance in
    feature space. The result is a locally smooth traversal through the album
    collection, favoring gradual stylistic transitions over global ordering.

    The playlist is written to a Markdown file and includes:
    - The seed album
    - Ordered album entries
    - Step-wise cosine distances between consecutive albums
    - Embedded album cover images when available in the local cover cache

    Args:
        albums (list[dict]): List of album metadata dictionaries, each
            containing at least "artists", "title", and optionally "cover_url".
        matrix (np.ndarray): Feature matrix used to compute cosine distances
            between albums.
        seed_idx (int): Index of the seed album from which the greedy walk
            begins.
        output_path (str | Path): Path to the output Markdown file.

    Returns:
        list[int]: Ordered list of album indices representing the greedy
        nearest-neighbor traversal path.
    """
    n = len(albums)
    remaining = set(range(n))
    path = [seed_idx]
    remaining.remove(seed_idx)

    dist_matrix = cosine_distances(matrix)
    current = seed_idx

    while remaining:
        next_idx = min(
            remaining,
            key=lambda i: dist_matrix[current][i]
        )
        path.append(next_idx)
        remaining.remove(next_idx)
        current = next_idx

    seed_album = albums[seed_idx]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Album Playlist — Greedy Nearest Neighbor\n\n")
        f.write(f"Seed album:\n**{seed_album['artists']} – {seed_album['title']}**\n\n")
        f.write("## Playlist\n\n")

        for i, idx in enumerate(path, start=1):
            album = albums[idx]

            cover_md = ""
            if album.get("cover_url"):
                cover_path = cover_cache_path(album['cover_url'], COVER_SIZE)
                if cover_path.exists():
                    cover_md = f"![Cover]({cover_path.as_posix()}) "

            if i == 1:
                note = "start"
            else:
                note = f"{dist_matrix[path[i-2]][idx]:.3f}"

            f.write(
                f"{i}. {cover_md}**{album['artists']} – {album['title']}** "
                f"(step distance: {note})\n"
            )


    print(f"Saved greedy playlist to {output_path}")
    return path


def render_map_with_path( #pylint: disable=too-many-positional-arguments, too-many-arguments
    albums,
    coords,
    canvas_w,
    canvas_h,
    path,
    output_path,
    line_color=(80, 200, 255),
    line_width=4
):
    """
    Render the album map and draw a polyline following
    the given album index path.
    """
    canvas = render_map(albums, coords, canvas_w, canvas_h)
    draw = ImageDraw.Draw(canvas)

    points = [tuple(coords[i]) for i in path]

    if len(points) > 1:
        draw.line(points, fill=line_color, width=line_width)

    # Optional: emphasize start / end
    start = points[0]
    end = points[-1]

    r = 10
    draw.ellipse(
        (start[0]-r, start[1]-r, start[0]+r, start[1]+r),
        outline=(0, 255, 0),
        width=3
    )
    draw.ellipse(
        (end[0]-r, end[1]-r, end[0]+r, end[1]+r),
        outline=(255, 80, 80),
        width=3
    )

    canvas.save(output_path)
    print(f"Saved path map to {output_path}")


def project_2d(matrix):
    """
    Project high-dimensional album features into a 2D semantic space.

    This function uses UMAP to reduce a feature matrix describing albums
    (genres, styles, artists, etc.) into two dimensions while preserving
    local neighborhood relationships. The resulting coordinates are suitable
    for visualization, distance-based traversal, and semantic path generation.

    Cosine distance is used to emphasize directional similarity between
    feature vectors rather than absolute magnitude.

    Returns:
        np.ndarray: Array of shape (n_items, 2) containing 2D semantic
        coordinates for each album.
    """
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=RANDOM_SEED,
        n_neighbors=15,
        min_dist=0.1,
    )
    return reducer.fit_transform(matrix)


def normalize_coords(coords, padding=200):
    """
    Normalize 2D coordinates into canvas pixel space.

    This function rescales arbitrary 2D coordinates so they fit within a
    square canvas defined by `CANVAS_SIZE`, while preserving relative spatial
    relationships. An optional padding margin is applied to keep points away
    from the canvas edges.

    Degenerate cases where all values along an axis are equal are handled
    safely to avoid division-by-zero errors.

    Args:
        coords (np.ndarray): Array of shape (n_items, 2) containing raw
            2D coordinates (e.g. from dimensionality reduction).
        padding (int, optional): Margin in pixels to reserve on each side of
            the canvas. Defaults to 200.

    Returns:
        np.ndarray: Array of shape (n_items, 2) containing normalized (x, y)
        coordinates in pixel space.
    """
    x = coords[:, 0]
    y = coords[:, 1]

    x = (x - x.min()) / (x.max() - x.min() or 1)
    y = (y - y.min()) / (y.max() - y.min() or 1)

    x = x * (CANVAS_SIZE - 2 * padding) + padding
    y = y * (CANVAS_SIZE - 2 * padding) + padding

    return np.column_stack((x, y))


def layout_albums_semantic(coords, n_items):
    """
    Arrange albums into a grid while preserving semantic ordering.

    This function takes 2D semantic coordinates (e.g. from UMAP) and converts
    them into discrete grid positions suitable for rendering album covers.
    Albums are first sorted by their projected coordinates to maintain
    neighborhood relationships, then placed sequentially into a square grid.

    The resulting layout avoids overlap, keeps visual structure stable, and
    produces a deterministic canvas size based on the number of albums.

    Args:
        coords (np.ndarray): Array of shape (n_items, 2) containing 2D semantic
            coordinates for each album.
        n_items (int): Total number of albums to place.

    Returns:
        tuple:
            - np.ndarray: Array of shape (n_items, 2) with pixel-space (x, y)
              positions for each album center.
            - tuple[int, int]: (canvas_width, canvas_height) in pixels.
    """
    grid = math.ceil(math.sqrt(n_items))
    cell = COVER_SIZE + GAP

    canvas_w = grid * cell + GAP
    canvas_h = grid * cell + GAP

    order = np.lexsort((coords[:, 0], coords[:, 1]))
    positions = np.zeros_like(coords)

    for idx, album_idx in enumerate(order):
        row = idx // grid
        col = idx % grid
        positions[album_idx] = (
            GAP + col * cell + COVER_SIZE // 2,
            GAP + row * cell + COVER_SIZE // 2,
        )

    return positions, (canvas_w, canvas_h)


def render_map(albums, coords, canvas_w, canvas_h):
    """
    Render a 2D album map with cover artwork.

    This function creates a canvas of the specified size and places each album
    at its corresponding 2D coordinate. Album positions are first marked with
    small reference points, after which available album cover images are
    downloaded (or loaded from cache) and composited onto the canvas.

    Albums without a cover URL or whose cover image cannot be retrieved are
    skipped gracefully.

    Args:
        albums (list[dict]): Sequence of album metadata dictionaries. Each
            album may include a "cover_url" key used to retrieve artwork.
        coords (list[tuple[float, float]]): Normalized (x, y) positions for
            each album in canvas space. Must be the same length and order
            as `albums`.
        canvas_w (int): Width of the output image in pixels.
        canvas_h (int): Height of the output image in pixels.

    Returns:
        PIL.Image.Image: The rendered album map image.
    """
    canvas = Image.new("RGB", (canvas_w, canvas_h), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)

    for (x, y) in coords:
        draw.ellipse((x-3, y-3, x+3, y+3), fill=(255, 50, 50))

    for album, (x, y) in zip(albums, coords):
        if not album.get("cover_url"):
            continue

        img = download_cover(album["cover_url"], COVER_SIZE)
        if not img:
            continue

        canvas.paste(
            img,
            (int(x - COVER_SIZE // 2), int(y - COVER_SIZE // 2))
        )

    return canvas


def download_cover(url, size):
    """
    Download, resize, and cache an album cover image.

    This function retrieves an album cover from the given URL, resizes it to a
    square of the requested size, and stores it in a local cache to avoid
    repeated network requests. If a cached image already exists, it is loaded
    and returned instead.

    The cache key is derived from the image URL and target size. Corrupt cache
    entries are automatically detected, deleted, and re-downloaded.

    Rate limiting is applied before network requests to comply with external
    service constraints.

    Args:
        url (str): Direct URL to the album cover image. If None or empty,
            the function returns None immediately.
        size (int): Target width and height (in pixels) for the square image.

    Returns:
        PIL.Image.Image or None: The resized album cover image in RGB mode,
        or None if the download, decoding, or caching process fails.
    """
    if not url:
        #print("No cover URL")
        return None
    path = cover_cache_path(url, size)
    if path.exists():
        #print(f"Cache hit: {path.name}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as e: #pylint: disable=broad-exception-caught
            print(f"Corrupt cache file {path.name}: {e}")
            path.unlink(missing_ok=True)
    #print(f"Downloading cover: {url}")
    try:
        rate_limit()
        r = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "AlbumMap/0.1"}
        )
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        img = img.resize((size, size), Image.Resampling.LANCZOS)
        img.save(path, "JPEG", quality=90)
        #print(f"Saved cover → {path}")
        return img
    except Exception as e: #pylint: disable=broad-exception-caught
        print(f"Cover download failed: {url}")
        print(f"Reason: {e}")
        return None


def main():
    """
    Entry point for the AlbumMap pipeline.

    This function orchestrates the full workflow:
    - Initializes deterministic randomness for reproducible layouts and paths.
    - Connects to the Discogs API and collects the user's releases.
    - Extracts album-level features and tags from the collection.
    - Builds a numerical feature matrix suitable for similarity analysis.
    - Projects albums into a 2D semantic space and normalizes the coordinates.
    - Computes a non-overlapping spatial layout for all albums.
    - Renders and saves a visual map of the album collection.
    - Selects a random seed album to serve as the starting point for playlists.
    - Generates two playlist variants:
        * A global similarity-ordered playlist.
        * A greedy nearest-neighbor walk playlist.
    - Exports both playlists as Markdown files and converts them to HTML.
    - Renders two additional maps visualizing the traversal path of each playlist.

    Output artifacts produced by this function include:
    - A base album map image.
    - Two playlist Markdown files (global and greedy).
    - HTML versions of each playlist.
    - Two annotated map images with playlist paths overlaid.

    All file paths, random seeds, and rendering parameters are controlled
    by module-level constants to ensure reproducibility across runs.
    """
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    client = init_discogs()
    releases = collect_releases(client)
    print(f"Collected {len(releases)} releases")
    albums, tags = extract_features(releases)
    print(f"Using {len(albums)} albums")
    matrix = build_feature_matrix(tags)
    coords = project_2d(matrix)
    coords = normalize_coords(coords)
    coords, (w, h) = layout_albums_semantic(coords, len(albums))
    image = render_map(albums, coords, w, h)
    image.save(OUTPUT_IMAGE)
    print(f"Saved map to {OUTPUT_IMAGE}")
    seed_idx = random.randrange(len(albums))
    global_md = "album_playlist_global.md"
    greedy_md = "album_playlist_greedy.md"
    global_path = generate_album_playlist_global_md(albums, matrix, seed_idx, global_md)
    greedy_path = generate_album_playlist_greedy_md(albums, matrix, seed_idx, greedy_md)
    convert_md_to_html(global_md)
    convert_md_to_html(greedy_md)
    render_map_with_path(
        albums,
        coords,
        w,
        h,
        global_path,
        "album_map_global.png",
        line_color=(120, 180, 255)
    )
    render_map_with_path(
        albums,
        coords,
        w,
        h,
        greedy_path,
        "album_map_greedy.png",
        line_color=(255, 180, 80)
    )



if __name__ == "__main__":
    main()


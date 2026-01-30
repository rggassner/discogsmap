# discogsmap
discogsmap is a Python project to visualize your Discogs collection as a semantic map of albums and generate playlists based on feature similarity. It combines music metadata with dimensionality reduction and nearest-neighbor exploration to create intuitive album maps and playlists.

---

## Features

- Fetches your Discogs collection (supports selecting specific folders).  
- Extracts features from albums: **genres, styles, artists**.  
- Computes **album similarity** using weighted features.  
- Generates playlists:
  - **Global distance** (sorted by similarity to seed album).  
  - **Greedy nearest neighbor** (smooth listening path).  
- Projects album features to **2D coordinates** using UMAP.  
- Generates **album maps** with cover images and paths.  
- Caches album metadata and cover images for efficiency.  
- Auto-generates `.md` playlists with embedded album cover images.  
- Converts playlists to HTML for web-friendly viewing.

---

## Screenshot

*Album maps and playlists*  

![Album Map Example](album_map.png)

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/rggassner/discogsmap.git
cd albummap
```

2.  Install dependencies:
    

bash

Copy code

`pip install -r requirements.txt`

3.  Set your Discogs API credentials

 DISCOGS_USER_TOKEN="your_discogs_token"
 DISCOGS_USERNAME="your_username"

Your token can be found here https://www.discogs.com/settings/developers
 

  
## Usage

`python discogsmap.py`

This will:

1.  Fetch your collection (selected folder, e.g., `"Punk"`). Select folder `"All"` to fetch your whole collection.
    
2.  Extract album features and cache them in `discogs_album_cache.json`.
    
3.  Download album cover images to `cover_cache/`.
    
4.  Generate:
    
    -   `album_map.png` - full album map.
        
    -   `album_playlist_global.md` - playlist sorted by global distance.
        
    -   `album_playlist_greedy.md` - playlist following greedy nearest-neighbor.
        
    -   `album_playlist_global.html` / `album_playlist_greedy.html` - html playlist versions.
        

* * *

## Customization

-   **Folders**: Change the `FOLDERS` list to select which Discogs folders to use.
    
-   **Weights**: Modify `TAG_WEIGHTS` to prioritize genres, styles, or artists.
      
-   **Random seed**: Set `RANDOM_SEED` for reproducible maps and playlists.
    

* * *

## Caching

-   **Album metadata** → `discogs_album_cache.json`
    
-   **Album covers** → `cover_cache/`
    
-   Caching reduces API calls and speeds up subsequent runs.
    

* * *

## Dependencies

-   `discogs-client`
    
-   `requests`
    
-   `numpy`
    
-   `scikit-learn`
    
-   `umap-learn`
    
-   `Pillow`
    
-   `markdown` (optional for HTML conversion)
    

See `requirements.txt` for exact versions.

* * *

## License

MIT License.

* * *

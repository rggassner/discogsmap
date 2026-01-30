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

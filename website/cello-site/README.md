# CELLo — project website

A static, no-build website for the **CELLo** project (bladder cancer diagnosis and follow-up from urine cytology, combining acoustofluidics, imaging and AI). It presents the project overview, the group's publications/theses/patents, and funding acknowledgements.

Plain HTML/CSS/JS — no framework, no build step, no dependencies to install. It is ready to deploy as-is on **GitHub Pages**.

## 1. What's in this repo

```
.
├── index.html              # the whole site (single page, anchor navigation)
├── css/
│   └── styles.css          # design tokens + all styling (incl. dark mode)
├── js/
│   └── main.js              # publication filters, mobile nav, abstract toggles, theme toggle
├── assets/
│   ├── img/
│   │   ├── graphical_abstract_cello.png
│   │   └── funding_fct.png
│   └── pdfs/                # publications, posters, theses, technology brief (10 files)
├── .nojekyll                # tells GitHub Pages to serve files as-is
└── README.md
```

No external services, no API keys, no analytics. The only external resource loaded is the Google Fonts stylesheet for the two typefaces (Newsreader, IBM Plex Sans); everything else is self-contained.

## 2. Deploy on GitHub Pages (exact steps)

**Option A — brand-new repository**

1. Create a new, empty repository on GitHub (public, so Pages can serve it for free) — e.g. `cello-project-site`. Do **not** initialise it with a README (you already have one here).
2. On your computer, open a terminal in this folder (the one containing `index.html`) and run:
   ```bash
   git init
   git add .
   git commit -m "Initial CELLo site"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<your-repo>.git
   git push -u origin main
   ```
3. On GitHub, go to your repository → **Settings** → **Pages** (left sidebar, under "Code and automation").
4. Under **Build and deployment → Source**, choose **Deploy from a branch**.
5. Under **Branch**, choose **main** and folder **/ (root)**, then click **Save**.
6. Wait about a minute, then refresh the Pages settings page — it will show your live URL:
   `https://<your-username>.github.io/<your-repo>/`

**Option B — add to an existing repository**

Copy all the files in this folder into the existing repo (or into a `/docs` folder if you prefer), commit and push, then in **Settings → Pages** pick the branch (and `/docs` folder, if used) you pushed to.

**Option C — user/organization site**

If you name the repository exactly `<your-username>.github.io`, GitHub Pages serves it at the root domain `https://<your-username>.github.io/` automatically, with the same Settings → Pages → Deploy from branch (`main`, `/root`) steps.

No CI, no build action, and no `_config.yml` is required — this is plain static HTML.

## 3. Run it locally before pushing (optional)

Any static file server works. For example, with Python already installed:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## 4. Updating content

- **Add a new publication**: duplicate one `<article class="pub-item ...">` block inside `#publications` in `index.html`, update the `data-type` attribute (`journal`, `conference`, `poster`, `thesis`, or `patent` — this drives both the coloured left rule and the filter pills), fill in the title/authors/venue/abstract, and drop the PDF into `assets/pdfs/`.
- **Add a new filter category**: add a `<button class="filter-pill" data-filter="...">` in the filter row, and use the same value as `data-type` on the relevant `pub-item` elements — `js/main.js` needs no changes, it reads these attributes generically.
- **Replace the graphical abstract / funding logo**: overwrite the files in `assets/img/`, keeping the same filenames (or update the `src` attributes in `index.html` if you rename them).
- **Colours and type**: all design tokens (colours, fonts, spacing) are declared once at the top of `css/styles.css` under `:root`, including a `prefers-color-scheme`/`[data-theme]` dark variant.

## 5. File sizes note

Two of the original source PDFs (a 67 MB conference paper export and an 89 MB MSc dissertation) were re-compressed — re-encoding embedded raster figures as optimised JPEGs while leaving all text as native, selectable PDF text — down to roughly 1.6 MB and 5.7 MB respectively, so the whole repository stays well under GitHub's size limits. If you'd rather host the original, uncompressed files, replace them in `assets/pdfs/` (GitHub warns above 50 MB per file and blocks above 100 MB without Git LFS).

## 6. Licence

The code in this repository (HTML/CSS/JS) is released under the MIT licence — see `LICENSE`. The publications, theses, graphical abstract and funding-body logo retain their original copyright and are included here for the purpose of sharing the group's own research output; do not reuse them outside that context without checking the relevant publisher/author terms.

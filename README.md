# Journey Inside an MRI — Interactive Fourier Reconstruction Lab

This is a complete Streamlit app designed to be embedded in a Squarespace website.

## What it demonstrates

1. Converts a brain MRI image into a 256 × 256 NumPy array.
2. Runs the 2D discrete Fourier transform `np.fft.fft2()` and centers the frequency grid using `fftshift()`.
3. Shows **Original image | k-space magnitude | Reconstructed image**.
4. Lets visitors use sliders to remove **high-frequency bands** (fine details) and **low-frequency disks** (large-scale contrast).
5. Reconstructs the filtered image using `ifftshift()` and `ifft2()`.
6. Reports the proportion of coefficients retained, RMSE, and maximum pixel error.
7. Verifies that full Fourier data reconstructs the image within floating-point precision.

This is **an idealized image-domain simulation**, not a full model of the scanner's MR signal acquisition process.

## Files

- `app.py` — Streamlit interface, source attribution, and controls
- `math_core.py` — pure NumPy Fourier math and frequency masks
- `synthetic_fallback.png` — synthetic teaching illustration used if a real MRI cannot be loaded
- `requirements.txt` — Python dependencies
- `squarespace_embed.html` — iframe code to paste into Squarespace
- `test_math.py` — numerical correctness tests
- `preview.png` — example visualization

## IMPORTANT: Make sure the REAL MRI image loads

The app tries to load the open-licensed original from Wikimedia Commons automatically. If the server blocks it, it visibly switches to a **synthetic fallback** rather than pretending the illustration is a clinical MRI.

For guaranteed real MRI display, manually download the source image:

- Source page: https://commons.wikimedia.org/wiki/File:MRI_Brain_T1_Axial_(11).jpg
- Download the **Original file** from that page.
- Rename it to `brain_mri.jpg` (the contents must actually be a JPG image).
- Upload the JPG into the same GitHub repository directory as `app.py`.

The app always chooses `brain_mri.jpg` first if available.

### Attribution and license

Original image: “MRI Brain T1 Axial (11).jpg” by **511KeV**, Wikimedia Commons, licensed **Creative Commons Attribution–ShareAlike 4.0 International** (CC BY-SA 4.0).

Source: https://commons.wikimedia.org/wiki/File:MRI_Brain_T1_Axial_(11).jpg
License: https://creativecommons.org/licenses/by-sa/4.0/

The app modifies the image by converting it to grayscale, resizing it, and filtering/reconstructing it. Those derived image outputs must retain CC BY-SA 4.0 attribution and ShareAlike licensing. The original author's endorsement is not implied. Keep the built-in attribution visible when using the app.

## Local test (optional)

Install Python 3.11 or newer, then from the unzipped folder:

```sh
python -m pip install -r requirements.txt
python test_math.py
python -m streamlit run app.py
```

The Streamlit app will open in a browser on your computer (normally at http://localhost:8501).

## Publish (no command line needed)

1. Sign in to GitHub: https://github.com/
2. Create a new repository, e.g., `mri-fourier-lab`.
3. Unzip the provided archive **first**, then use **Add file → Upload files** to upload `app.py`, `math_core.py`, `requirements.txt`, `synthetic_fallback.png`, and the optional `brain_mri.jpg` to the top level of the repository. Uploading just the ZIP file will not run the app.
4. Commit changes.
5. Open https://share.streamlit.io/ and sign in with GitHub.
6. Choose **Create app**, select the repository, branch `main`, and entry point **`app.py`**.
7. Deploy. Your public URL will look like `https://YOUR-LAB.streamlit.app/`.
8. Open the public link to confirm the real MRI and both sliders work.

Official deployment guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## Embed in Squarespace

**Plan requirement:** On current Squarespace plans, an iframe in a Code Block requires Core, Plus, Advanced, or a supported legacy plan. Current Basic does not allow it.

1. Open the page editor for the appropriate Squarespace story chapter.
2. Click **Add Block → Code**.
3. Paste the code from `squarespace_embed.html` into the block. Use HTML mode, and ensure **Display Source** is **off**.
4. Replace `https://YOUR-LAB.streamlit.app/` with the actual URL you received from Streamlit. Preserve `?embed=true`.
5. Save the page, then **test the published site**, not only the editor.
6. Adjust iframe height if the app has a vertical scrollbar. On small screens it may require more height.

Official Squarespace Code Blocks: https://support.squarespace.com/hc/en-us/articles/206543167-Add-custom-code-to-a-page-with-code-blocks
Official Streamlit embed: https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app/embed-your-app

### If iframe is not available

Publish the Streamlit app as usual, then add a regular Squarespace **Button** linking to your public Streamlit URL (opens in a new tab). The lab still works; it is just not embedded within the page.

## In-class demonstration

Click **All data** first: reconstruction should exactly match the original within numerical precision. Next click **Blur the image** to remove high-frequency detail, then click **Keep edges** to discard central low frequencies. Challenge visitors to explain the results with Fourier coefficients and the non-injective measurement map.

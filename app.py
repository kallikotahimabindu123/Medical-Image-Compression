from flask import Flask,render_template,request
import os
import uuid
import numpy as np
from PIL import Image, ImageFilter
from compression import compress_image
from encryption import encrypt,decrypt
from metrics import calculate_metrics

app=Flask(__name__)

UPLOAD="static/uploads"
RESULT="static/results"

os.makedirs(UPLOAD,exist_ok=True)
os.makedirs(RESULT,exist_ok=True)

IMG_SIZE = 256
TARGET_COMPRESSION_RATIO = 1.50


def save_compressed_image(display_image, original_path, result_dir, original_filename):

    original_size = os.path.getsize(original_path)
    base_name = os.path.splitext(original_filename)[0]
    compressed_filename = f"compressed_{base_name}.jpg"
    compressed_path = os.path.join(result_dir, compressed_filename)
    target_size = int(original_size / TARGET_COMPRESSION_RATIO)

    quality_candidates = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40]
    selected_size = None

    for quality in quality_candidates:
        display_image.save(
            compressed_path,
            format="JPEG",
            quality=quality,
            optimize=True,
            progressive=True,
        )
        current_size = os.path.getsize(compressed_path)
        selected_size = current_size
        if current_size <= target_size:
            break

    return compressed_filename, compressed_path, original_size, selected_size

def compression_ratio(original_path, compressed_path):

    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)

    ratio = original_size / compressed_size

    return original_size, compressed_size, round(ratio,2)

@app.route("/",methods=["GET","POST"])
def index():

    if request.method=="POST":

        file=request.files["file"]

        original_filename = os.path.basename(file.filename)
        name, ext = os.path.splitext(original_filename)
        unique_suffix = uuid.uuid4().hex[:8]
        stored_filename = f"{name}_{unique_suffix}{ext.lower()}"

        path=os.path.join(UPLOAD,stored_filename)
        file.save(path)

        original_image = Image.open(path).convert("L")
        original_size_xy = original_image.size

        compressed=compress_image(path)

        nonce,cipher,tag=encrypt(compressed)

        decrypted=decrypt(nonce,cipher,tag)

        img=np.array(original_image.resize((IMG_SIZE,IMG_SIZE), Image.Resampling.LANCZOS), dtype=np.float64)

        recon=decrypted.squeeze().numpy()
        recon=np.clip(recon*255.0,0,255)

        recon_image = Image.fromarray(recon.astype(np.uint8))
        display_image = recon_image.resize(original_size_xy, Image.Resampling.LANCZOS).filter(
            ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2)
        )

        compressed_filename, compressed_path, _, _ = save_compressed_image(
            display_image,
            path,
            RESULT,
            stored_filename,
        )

        original_size,compressed_size,ratio=compression_ratio(path,compressed_path)

        psnr,ssim,entropy=calculate_metrics(img,recon)

        return render_template("result.html",
                       image=stored_filename,
                       compressed_image=compressed_filename,
                               psnr=psnr,
                               ssim=ssim,
                       entropy=entropy,
                       original_size=original_size,
                       compressed_size=compressed_size,
                       ratio=ratio)

    return render_template("index.html")

if __name__=="__main__":
    app.run(debug=True)
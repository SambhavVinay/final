import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
from datetime import timedelta

# FastAPI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Google Cloud
from google.cloud import storage
from google.auth import default
from google.auth.transport import requests


app = FastAPI()

# Enable CORS for Capacitor app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# GCS Setup
BUCKET_NAME = "campus-finder-bucket"
gcp_credentials, project_id = default()
storage_client = storage.Client(credentials=gcp_credentials)
bucket = storage_client.bucket(BUCKET_NAME)


# --- AI MODEL SETUP ---
class CampusFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(CampusFeatureExtractor, self).__init__()

        resnet = models.resnet18(weights=None)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure campus_model.pth is in the same directory as main.py
        resnet.load_state_dict(
            torch.load("campus_model.pth", map_location=device)
        )

        self.feature_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )
        self.feature_extractor.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
            return x.view(x.size(0), -1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CampusFeatureExtractor(num_classes=4).to(device)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def get_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")

    tensor = preprocess(image).unsqueeze(0).to(device)
    embedding = model(tensor)

    norm = embedding.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return embedding / norm


# --- PUBLIC API ROUTES ---
@app.get("/")
async def root():
    return {
        "status": "online",
        "message": "ResNet-18 Backend Online (Public Mode)"
    }


@app.get("/get-folder-images")
async def get_folder_images(folder: str):
    """Retrieves image URLs for a specific item folder."""
    try:
        prefix = f"{folder}/"
        blobs = storage_client.list_blobs(
            BUCKET_NAME,
            prefix=prefix
        )

        gcp_credentials.refresh(requests.Request())

        image_urls = []
        for blob in blobs:
            if blob.name == prefix:
                continue

            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=1),
                service_account_email=gcp_credentials.service_account_email,
                access_token=gcp_credentials.token,
                method="GET"
            )
            image_urls.append(url)

        return {"images": image_urls}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/items-list")
async def list_items_with_thumbnails():
    """Returns a list of all folders in the bucket with one thumbnail each."""
    try:
        gcp_credentials.refresh(requests.Request())
        blobs = list(storage_client.list_blobs(BUCKET_NAME))

        item_map = {}
        for blob in blobs:
            if '/' in blob.name and not blob.name.endswith(".pt"):
                folder_name = blob.name.split('/')[0]

                if folder_name not in item_map:
                    url = blob.generate_signed_url(
                        version="v4",
                        expiration=timedelta(hours=1),
                        service_account_email=gcp_credentials.service_account_email,
                        access_token=gcp_credentials.token,
                        method="GET"
                    )
                    item_map[folder_name] = url

        result = [
            {"name": name, "thumbnail": url}
            for name, url in item_map.items()
        ]

        return {"items": result}

    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/found")
async def report_found(
    item_name: str = Form(...),
    files: list[UploadFile] = File(...)
):
    try:
        for file in files:
            content = await file.read()

            # 1️⃣ Upload image
            img_blob = bucket.blob(f"{item_name}/{file.filename}")
            img_blob.upload_from_string(
                content,
                content_type=file.content_type
            )

            # 2️⃣ Generate embedding ONCE
            embedding = get_embedding(content).cpu()

            # 3️⃣ Save embedding
            buffer = io.BytesIO()
            torch.save(embedding, buffer)
            buffer.seek(0)

            emb_blob = bucket.blob(
                f"{item_name}/{file.filename}.pt"
            )
            emb_blob.upload_from_string(buffer.read())

        return {
            "status": "success",
            "message": f"{item_name} registered"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_item(file: UploadFile = File(...)):
    try:
        query_bytes = await file.read()
        query_vec = get_embedding(query_bytes)

        blobs = storage_client.list_blobs(BUCKET_NAME)

        item_vectors = {}
        for blob in blobs:
            if not blob.name.endswith(".pt"):
                continue

            folder = blob.name.split("/")[0]
            buffer = io.BytesIO(blob.download_as_bytes())
            vec = torch.load(buffer, map_location=device)

            if folder not in item_vectors:
                item_vectors[folder] = []

            item_vectors[folder].append(vec)

        best_score = -1.0
        best_item = None

        for item, vecs in item_vectors.items():
            master_vec = torch.mean(
                torch.stack(vecs),
                dim=0
            )

            norm = master_vec.norm(
                dim=-1,
                keepdim=True
            ).clamp(min=1e-6)
            master_vec = master_vec / norm

            score = torch.sum(query_vec * master_vec).item()


            if score > best_score:
                best_score = score
                best_item = item

        return {
            "match": best_score > 0.8,
            "item": best_item,
            "confidence": round(best_score, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)

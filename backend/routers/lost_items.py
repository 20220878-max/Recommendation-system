from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from database import get_db
from schemas import LostItemCreate, LostItemUpdate, LostItemResponse
import crud, shutil, uuid, os

router = APIRouter(prefix="/lost-items", tags=["분실물"])

UPLOAD_DIR = "static/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/", response_model=LostItemResponse)
def create(user_id: int, data: LostItemCreate, db: Session = Depends(get_db)):
    return crud.create_lost_item(db, user_id, data)

@router.get("/", response_model=list[LostItemResponse])
def list_items(skip: int = 0, limit: int = 20, db: Session = Depends(get_db)):
    return crud.get_lost_items(db, skip, limit)

@router.get("/{item_id}", response_model=LostItemResponse)
def get_item(item_id: int, db: Session = Depends(get_db)):
    item = crud.get_lost_item(db, item_id)
    if not item:
        raise HTTPException(status_code=404, detail="분실물을 찾을 수 없습니다")
    return item

@router.patch("/{item_id}", response_model=LostItemResponse)
def update(item_id: int, data: LostItemUpdate, db: Session = Depends(get_db)):
    item = crud.update_lost_item(db, item_id, data)
    if not item:
        raise HTTPException(status_code=404, detail="분실물을 찾을 수 없습니다")
    return item

@router.delete("/{item_id}")
def delete(item_id: int, db: Session = Depends(get_db)):
    if not crud.delete_lost_item(db, item_id):
        raise HTTPException(status_code=404, detail="분실물을 찾을 수 없습니다")
    return {"message": "삭제 완료"}

@router.post("/{item_id}/images")
def upload_image(item_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    ext = file.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = f"{UPLOAD_DIR}/{filename}"
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    image_url = f"/static/images/{filename}"
    return crud.save_image(db, image_url=image_url, lost_item_id=item_id)
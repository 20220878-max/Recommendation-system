from sqlalchemy.orm import Session
from models import LostItem, FoundItem, User, ItemImage
from schemas import LostItemCreate, LostItemUpdate, FoundItemCreate, FoundItemUpdate, UserCreate

# ── User ──────────────────────────────────────
def create_user(db: Session, data: UserCreate):
    user = User(**data.model_dump())
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

def get_user(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

# ── LostItem ──────────────────────────────────
def create_lost_item(db: Session, user_id: int, data: LostItemCreate):
    item = LostItem(user_id=user_id, **data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def get_lost_items(db: Session, skip: int = 0, limit: int = 20):
    return db.query(LostItem).offset(skip).limit(limit).all()

def get_lost_item(db: Session, item_id: int):
    return db.query(LostItem).filter(LostItem.id == item_id).first()

def update_lost_item(db: Session, item_id: int, data: LostItemUpdate):
    item = get_lost_item(db, item_id)
    if not item:
        return None
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item

def delete_lost_item(db: Session, item_id: int):
    item = get_lost_item(db, item_id)
    if not item:
        return False
    db.delete(item)
    db.commit()
    return True

# ── FoundItem ─────────────────────────────────
def create_found_item(db: Session, user_id: int, data: FoundItemCreate):
    item = FoundItem(user_id=user_id, **data.model_dump())
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def get_found_items(db: Session, skip: int = 0, limit: int = 20):
    return db.query(FoundItem).offset(skip).limit(limit).all()

def get_found_item(db: Session, item_id: int):
    return db.query(FoundItem).filter(FoundItem.id == item_id).first()

def update_found_item(db: Session, item_id: int, data: FoundItemUpdate):
    item = get_found_item(db, item_id)
    if not item:
        return None
    for k, v in data.model_dump(exclude_unset=True).items():
        setattr(item, k, v)
    db.commit()
    db.refresh(item)
    return item

def delete_found_item(db: Session, item_id: int):
    item = get_found_item(db, item_id)
    if not item:
        return False
    db.delete(item)
    db.commit()
    return True

# ── Image ─────────────────────────────────────
def save_image(db: Session, image_url: str, lost_item_id=None, found_item_id=None):
    img = ItemImage(
        image_url=image_url,
        lost_item_id=lost_item_id,
        found_item_id=found_item_id
    )
    db.add(img)
    db.commit()
    db.refresh(img)
    return img
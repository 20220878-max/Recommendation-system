from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
from enum import Enum

# ── Category Enum ──────────────────────────────
class ItemCategory(str, Enum):
    wallet      = "지갑/카드"
    bag         = "가방"
    clothes     = "의류/잡화"
    phone       = "휴대폰/전자기기"
    earphone    = "이어폰"
    id_doc      = "여권/신분증"
    accessory   = "악세사리"
    daily       = "생활용품"
    cosmetic    = "화장품"
    stationery  = "문구류"
    etc         = "기타"

# ── User ──────────────────────────────────────
class UserCreate(BaseModel):
    name: Optional[str] = None
    email: EmailStr
    password: str
    phone: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    name: Optional[str]
    email: str
    phone: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

# ── Image ─────────────────────────────────────
class ImageResponse(BaseModel):
    id: int
    image_url: str

    class Config:
        from_attributes = True

# ── LostItem ──────────────────────────────────
class LostItemCreate(BaseModel):
    title:       str
    category:    ItemCategory
    description: Optional[str] = None
    location:    Optional[str] = None
    lost_at:     Optional[datetime] = None

class LostItemUpdate(BaseModel):
    title:       Optional[str] = None
    category:    Optional[ItemCategory] = None
    description: Optional[str] = None
    location:    Optional[str] = None
    lost_at:     Optional[datetime] = None
    status:      Optional[str] = None

class LostItemResponse(BaseModel):
    id:          int
    user_id:     int
    title:       str
    category:    str
    description: Optional[str]
    location:    Optional[str]
    lost_at:     Optional[datetime]
    status:      str
    created_at:  datetime
    images:      List[ImageResponse] = []

    class Config:
        from_attributes = True

# ── FoundItem ─────────────────────────────────
class FoundItemCreate(BaseModel):
    category:    ItemCategory
    description: Optional[str] = None
    location:    Optional[str] = None
    found_at:    Optional[datetime] = None

class FoundItemUpdate(BaseModel):
    category:    Optional[ItemCategory] = None
    description: Optional[str] = None
    location:    Optional[str] = None
    found_at:    Optional[datetime] = None
    status:      Optional[str] = None

class FoundItemResponse(BaseModel):
    id:          int
    user_id:     int
    category:    str
    description: Optional[str]
    location:    Optional[str]
    found_at:    Optional[datetime]
    status:      str
    created_at:  datetime
    images:      List[ImageResponse] = []

    class Config:
        from_attributes = True
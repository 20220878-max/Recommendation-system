from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

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
    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    lost_at: Optional[datetime] = None

class LostItemUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    lost_at: Optional[datetime] = None
    status: Optional[str] = None

class LostItemResponse(BaseModel):
    id: int
    user_id: int
    title: str
    description: Optional[str]
    location: Optional[str]
    lost_at: Optional[datetime]
    status: str
    created_at: datetime
    images: List[ImageResponse] = []

    class Config:
        from_attributes = True

# ── FoundItem ─────────────────────────────────
class FoundItemCreate(BaseModel):
    description: Optional[str] = None
    location: Optional[str] = None
    found_at: Optional[datetime] = None

class FoundItemUpdate(BaseModel):
    description: Optional[str] = None
    location: Optional[str] = None
    found_at: Optional[datetime] = None
    status: Optional[str] = None

class FoundItemResponse(BaseModel):
    id: int
    user_id: int
    description: Optional[str]
    location: Optional[str]
    found_at: Optional[datetime]
    status: str
    created_at: datetime
    images: List[ImageResponse] = []

    class Config:
        from_attributes = True
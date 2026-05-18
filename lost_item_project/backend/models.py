# models.py
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Enum, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from database import engine
from datetime import datetime

Base = declarative_base()

# 사용자 테이블
class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, autoincrement=True)
    name       = Column(String(50))
    email      = Column(String(100), unique=True, nullable=False)
    password   = Column(String(255), nullable=False)
    phone      = Column(String(20), nullable=True)
    fcm_token  = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    lost_items  = relationship("LostItem", back_populates="user")
    found_items = relationship("FoundItem", back_populates="user")

# 카테고리 Enum (공통)
CATEGORY_ENUM = Enum(
    "지갑/카드",
    "가방",
    "의류/잡화",
    "휴대폰/전자기기",
    "이어폰",
    "여권/신분증",
    "악세사리",
    "생활용품",
    "화장품",
    "문구류",
    "기타",
    name="item_category"
)

# 분실물 테이블
class LostItem(Base):
    __tablename__ = "lost_items"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    title       = Column(String(100), nullable=False)
    category    = Column(CATEGORY_ENUM, nullable=False)  # 추가
    description = Column(Text)
    location    = Column(String(100))
    lost_at     = Column(DateTime)
    status      = Column(Enum("waiting", "matched", "resolved", name="lost_item_status"), default="waiting")
    created_at  = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="lost_items")
    images = relationship("ItemImage", back_populates="lost_item")

# 습득물 테이블
class FoundItem(Base):
    __tablename__ = "found_items"

    id          = Column(Integer, primary_key=True, autoincrement=True)
    user_id     = Column(Integer, ForeignKey("users.id"), nullable=False)
    category    = Column(CATEGORY_ENUM, nullable=False)  # 추가
    description = Column(Text)
    location    = Column(String(100))
    found_at    = Column(DateTime)
    status      = Column(Enum("waiting", "matched", "resolved", name="found_item_status"), default="waiting")
    created_at  = Column(DateTime, default=datetime.utcnow)

    user   = relationship("User", back_populates="found_items")
    images = relationship("ItemImage", back_populates="found_item")

# 매칭 테이블
class Match(Base):
    __tablename__ = "matches"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    lost_item_id  = Column(Integer, ForeignKey("lost_items.id"), nullable=False)
    found_item_id = Column(Integer, ForeignKey("found_items.id"), nullable=False)
    similarity    = Column(Float, nullable=False)
    status        = Column(Enum("pending", "confirmed", "rejected", name="match_status"), default="pending")
    created_at    = Column(DateTime, default=datetime.utcnow)

# 채팅방 테이블
class ChatRoom(Base):
    __tablename__ = "chat_rooms"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    match_id      = Column(Integer, ForeignKey("matches.id"), nullable=False)
    lost_user_id  = Column(Integer, ForeignKey("users.id"), nullable=False)
    found_user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)
    closed_at     = Column(DateTime, nullable=True)

# 이미지 테이블
class ItemImage(Base):
    __tablename__ = "item_images"

    id            = Column(Integer, primary_key=True, autoincrement=True)
    lost_item_id  = Column(Integer, ForeignKey("lost_items.id"), nullable=True)
    found_item_id = Column(Integer, ForeignKey("found_items.id"), nullable=True)
    image_url     = Column(String(255), nullable=False)
    created_at    = Column(DateTime, default=datetime.utcnow)

    lost_item  = relationship("LostItem", back_populates="images")
    found_item = relationship("FoundItem", back_populates="images")

Base.metadata.create_all(bind=engine)
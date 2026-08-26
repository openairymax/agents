"""
OpenLab Core Storage Module

Data storage abstraction core module
Following AgentRT architecture design principles V1.8
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import asyncio
import json
import sqlite3
import time
from pathlib import Path


class StorageType(Enum):
    MEMORY = "memory"
    SQLITE = "sqlite"
    FILE = "file"
    # 以下两种类型当前版本未内置实现，需外部依赖或自定义实现：
    # - REDIS：需要安装 redis 依赖（redis-py）并实现 RedisStorage；
    # - CUSTOM：由业务方按 Storage ABC 自行实现。
    REDIS = "redis"
    CUSTOM = "custom"


class DataCategory(Enum):
    TASK = "task"
    AGENT = "agent"
    TOOL = "tool"
    CHECKPOINT = "checkpoint"
    LOG = "log"
    METADATA = "metadata"


@dataclass
class StorageRecord:
    key: str
    value: Any
    category: DataCategory = DataCategory.METADATA
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category.value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StorageRecord":
        return cls(
            key=data["key"],
            value=data["value"],
            category=DataCategory(data.get("category", "metadata")),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            expires_at=data.get("expires_at"),
            version=data.get("version", 1),
        )


@dataclass
class QueryResult:
    records: List[StorageRecord]
    total: int
    offset: int
    limit: int


class Storage(ABC):

    def __init__(self, storage_type: StorageType):
        self.storage_type = storage_type
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @abstractmethod
    async def initialize(self) -> None:
        self._initialized = True

    @abstractmethod
    async def close(self) -> None:
        self._initialized = False

    @abstractmethod
    async def get(self, key: str) -> Optional[StorageRecord]:
        pass

    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        category: DataCategory = DataCategory.METADATA,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> bool:
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    async def query(
        self,
        category: Optional[DataCategory] = None,
        filter_func: Optional[callable] = None,
        offset: int = 0,
        limit: int = 100
    ) -> QueryResult:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    async def get_json(self, key: str) -> Optional[Any]:
        record = await self.get(key)
        if record and record.value:
            if isinstance(record.value, str):
                return json.loads(record.value)
            return record.value
        return None

    async def set_json(
        self,
        key: str,
        value: Any,
        **kwargs
    ) -> bool:
        return await self.set(key, json.dumps(value), **kwargs)


class MemoryStorage(Storage):

    def __init__(self):
        super().__init__(StorageType.MEMORY)
        self._data: Dict[str, StorageRecord] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await super().initialize()

    async def close(self) -> None:
        async with self._lock:
            self._data.clear()
        await super().close()

    async def get(self, key: str) -> Optional[StorageRecord]:
        async with self._lock:
            record = self._data.get(key)
            if record:
                if record.expires_at and time.time() > record.expires_at:
                    await self.delete(key)
                    return None
                return record
            return None

    async def set(
        self,
        key: str,
        value: Any,
        category: DataCategory = DataCategory.METADATA,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> bool:
        async with self._lock:
            now = time.time()
            record = StorageRecord(
                key=key,
                value=value,
                category=category,
                metadata=metadata or {},
                created_at=now,
                updated_at=now,
                expires_at=(now + ttl) if ttl else None,
            )

            existing = self._data.get(key)
            if existing:
                record.version = existing.version + 1

            self._data[key] = record
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        async with self._lock:
            return key in self._data

    async def query(
        self,
        category: Optional[DataCategory] = None,
        filter_func: Optional[callable] = None,
        offset: int = 0,
        limit: int = 100
    ) -> QueryResult:
        async with self._lock:
            records = list(self._data.values())

            if category:
                records = [r for r in records if r.category == category]

            if filter_func:
                records = [r for r in records if filter_func(r)]

            now = time.time()
            non_expired = []
            for record in records:
                if record.expires_at and now > record.expires_at:
                    await self.delete(record.key)
                else:
                    non_expired.append(record)
            records = non_expired

            total = len(records)
            records = records[offset:offset + limit]

            return QueryResult(
                records=records,
                total=total,
                offset=offset,
                limit=limit,
            )

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    def size(self) -> int:
        return len(self._data)


class SQLiteStorage(Storage):

    def __init__(self, db_path: Union[str, Path]):
        super().__init__(StorageType.SQLITE)
        self.db_path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        await super().initialize()

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row

        await self._create_tables()

    async def _create_tables(self) -> None:
        loop = asyncio.get_event_loop()

        def create():
            cursor = self._conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    category TEXT,
                    metadata TEXT,
                    created_at REAL,
                    updated_at REAL,
                    expires_at REAL,
                    version INTEGER DEFAULT 1
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_category
                ON records(category)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires
                ON records(expires_at)
            """)
            self._conn.commit()

        await loop.run_in_executor(None, create)

    async def close(self) -> None:
        if self._conn:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._conn.close)
            self._conn = None
        await super().close()

    async def get(self, key: str) -> Optional[StorageRecord]:
        loop = asyncio.get_event_loop()

        def fetch():
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT * FROM records WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            return row

        row = await loop.run_in_executor(None, fetch)

        if row:
            record = StorageRecord(
                key=row["key"],
                value=json.loads(row["value"]) if row["value"] else None,
                category=DataCategory(row["category"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
                version=row["version"],
            )

            if record.expires_at and time.time() > record.expires_at:
                await self.delete(key)
                return None

            return record

        return None

    async def set(
        self,
        key: str,
        value: Any,
        category: DataCategory = DataCategory.METADATA,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> bool:
        loop = asyncio.get_event_loop()
        now = time.time()

        def upsert():
            cursor = self._conn.cursor()

            cursor.execute(
                "SELECT version FROM records WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()

            version = (row["version"] + 1) if row else 1

            cursor.execute("""
                INSERT OR REPLACE INTO records
                (key, value, category, metadata, created_at, updated_at, expires_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                key,
                json.dumps(value) if value is not None else None,
                category.value,
                json.dumps(metadata or {}),
                now,
                now,
                (now + ttl) if ttl else None,
                version,
            ))
            self._conn.commit()
            return True

        return await loop.run_in_executor(None, upsert)

    async def delete(self, key: str) -> bool:
        loop = asyncio.get_event_loop()

        def delete():
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM records WHERE key = ?", (key,))
            self._conn.commit()
            return cursor.rowcount > 0

        return await loop.run_in_executor(None, delete)

    async def exists(self, key: str) -> bool:
        loop = asyncio.get_event_loop()

        def check():
            cursor = self._conn.cursor()
            cursor.execute(
                "SELECT 1 FROM records WHERE key = ? LIMIT 1",
                (key,)
            )
            return cursor.fetchone() is not None

        return await loop.run_in_executor(None, check)

    async def query(
        self,
        category: Optional[DataCategory] = None,
        filter_func: Optional[callable] = None,
        offset: int = 0,
        limit: int = 100
    ) -> QueryResult:
        loop = asyncio.get_event_loop()

        def fetch():
            cursor = self._conn.cursor()

            if category:
                cursor.execute(
                    "SELECT * FROM records WHERE category = ? LIMIT ? OFFSET ?",
                    (category.value, limit, offset)
                )
            else:
                cursor.execute(
                    "SELECT * FROM records LIMIT ? OFFSET ?",
                    (limit, offset)
                )

            rows = cursor.fetchall()

            if category:
                cursor.execute(
                    "SELECT COUNT(*) FROM records WHERE category = ?",
                    (category.value,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM records")

            total = cursor.fetchone()[0]
            return rows, total

        rows, total = await loop.run_in_executor(None, fetch)

        records = []
        for row in rows:
            record = StorageRecord(
                key=row["key"],
                value=json.loads(row["value"]) if row["value"] else None,
                category=DataCategory(row["category"]),
                metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                expires_at=row["expires_at"],
                version=row["version"],
            )

            if filter_func and not filter_func(record):
                continue

            if record.expires_at and time.time() > record.expires_at:
                await self.delete(record.key)
            else:
                records.append(record)

        return QueryResult(
            records=records,
            total=total,
            offset=offset,
            limit=limit,
        )

    async def clear(self) -> None:
        loop = asyncio.get_event_loop()

        def truncate():
            cursor = self._conn.cursor()
            cursor.execute("DELETE FROM records")
            self._conn.commit()

        await loop.run_in_executor(None, truncate)


class FileStorage(Storage):
    """基于 JSONL 文件的真实文件存储。

    每条记录以一行 JSON 追加写入 ``<data_dir>/<category>.jsonl``（按类别分文件），
    提供与 :class:`MemoryStorage` / :class:`SQLiteStorage` 一致的异步接口
    （``get/set/delete/exists/query/clear``），并额外提供便捷方法
    ``put/search/iter``。初始化时全量加载索引，读写均在进程内保持一致性；
    数据落盘，进程重启后仍可恢复。
    """

    def __init__(self, data_dir: Union[str, Path]):
        super().__init__(StorageType.FILE)
        self.data_dir = Path(data_dir)
        self._index: Dict[str, StorageRecord] = {}
        self._lock = asyncio.Lock()

    # ── 文件路径 ─────────────────────────────────────────

    def _category_file(self, category: DataCategory) -> Path:
        """返回某类别对应的 JSONL 文件路径。"""
        return self.data_dir / f"{category.value}.jsonl"

    # ── 序列化 ───────────────────────────────────────────

    @staticmethod
    def _record_to_line(record: StorageRecord) -> str:
        """把记录序列化为一行 JSON（值不可序列化时用 str 兜底）。"""
        return json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            default=str,
        )

    @classmethod
    def _line_to_record(cls, line: str) -> Optional[StorageRecord]:
        """把一行 JSON 解析为记录；损坏行返回 None。"""
        line = line.strip()
        if not line:
            return None
        try:
            return StorageRecord.from_dict(json.loads(line))
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    # ── 生命周期 ─────────────────────────────────────────

    async def initialize(self) -> None:
        """建目录并全量加载索引（每类 JSONL 取同名 key 的最后一条）。"""
        await super().initialize()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path in self.data_dir.glob("*.jsonl"):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    record = self._line_to_record(line)
                    if record is not None:
                        self._index[record.key] = record

    async def close(self) -> None:
        """关闭存储：保留磁盘文件，仅清空内存索引。"""
        async with self._lock:
            self._index.clear()
        await super().close()

    # ── 核心接口 ─────────────────────────────────────────

    async def get(self, key: str) -> Optional[StorageRecord]:
        """按 key 读取记录；过期记录自动删除并返回 None。"""
        async with self._lock:
            record = self._index.get(key)
            if record is None:
                return None
            if record.expires_at and time.time() > record.expires_at:
                await self.delete(key)
                return None
            return record

    async def set(
        self,
        key: str,
        value: Any,
        category: DataCategory = DataCategory.METADATA,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None
    ) -> bool:
        """写入记录：JSONL 追加一行并更新内存索引。

        已存在同名 key 时 version 自增（基于内存索引中的旧记录）。
        """
        async with self._lock:
            now = time.time()
            existing = self._index.get(key)
            record = StorageRecord(
                key=key,
                value=value,
                category=category,
                metadata=metadata or {},
                created_at=existing.created_at if existing else now,
                updated_at=now,
                expires_at=(now + ttl) if ttl else None,
                version=(existing.version + 1) if existing else 1,
            )
            path = self._category_file(category)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(self._record_to_line(record) + "\n")
            self._index[key] = record
            return True

    async def delete(self, key: str) -> bool:
        """删除记录：移除索引条目并重写所属类别文件（跳过该 key 的行）。"""
        async with self._lock:
            record = self._index.pop(key, None)
            if record is None:
                return False
            path = self._category_file(record.category)
            if path.exists():
                kept: List[str] = []
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parsed = self._line_to_record(line)
                        if parsed is None or parsed.key != key:
                            kept.append(line.rstrip("\n"))
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("\n".join(kept))
                    if kept:
                        fh.write("\n")
            return True

    async def exists(self, key: str) -> bool:
        """判断 key 是否存在（含过期检查）。"""
        async with self._lock:
            record = self._index.get(key)
            if record is None:
                return False
            if record.expires_at and time.time() > record.expires_at:
                await self.delete(key)
                return False
            return True

    async def query(
        self,
        category: Optional[DataCategory] = None,
        filter_func: Optional[callable] = None,
        offset: int = 0,
        limit: int = 100
    ) -> QueryResult:
        """按类别 / 过滤函数查询记录，支持分页。"""
        async with self._lock:
            records = list(self._index.values())

            if category:
                records = [r for r in records if r.category == category]

            if filter_func:
                records = [r for r in records if filter_func(r)]

            now = time.time()
            non_expired = []
            for record in records:
                if record.expires_at and now > record.expires_at:
                    await self.delete(record.key)
                else:
                    non_expired.append(record)
            records = non_expired

            total = len(records)
            records = records[offset:offset + limit]

            return QueryResult(
                records=records,
                total=total,
                offset=offset,
                limit=limit,
            )

    async def clear(self) -> None:
        """清空所有记录：删除全部 JSONL 文件并重置索引。"""
        async with self._lock:
            for path in self.data_dir.glob("*.jsonl"):
                path.unlink(missing_ok=True)
            self._index.clear()

    # ── 便捷方法（与任务要求的 put/get/delete/search/iter 对齐） ──

    async def put(self, key: str, value: Any, **kwargs) -> bool:
        """写入记录的便捷别名（等价于 :meth:`set`）。"""
        return await self.set(key, value, **kwargs)

    async def search(
        self,
        category: Optional[DataCategory] = None,
        filter_func: Optional[callable] = None,
    ) -> List[StorageRecord]:
        """按类别 / 过滤函数搜索记录，返回全部命中的记录列表。"""
        result = await self.query(category=category, filter_func=filter_func,
                                  offset=0, limit=10**6)
        return result.records

    def iter(self) -> Any:
        """遍历全部记录（同步迭代器，含过期检查）。"""
        now = time.time()
        for record in list(self._index.values()):
            if record.expires_at and now > record.expires_at:
                continue
            yield record

    def size(self) -> int:
        """当前索引中的记录数。"""
        return len(self._index)


__all__ = [
    "Storage",
    "StorageType",
    "DataCategory",
    "StorageRecord",
    "QueryResult",
    "MemoryStorage",
    "SQLiteStorage",
    "FileStorage",
]

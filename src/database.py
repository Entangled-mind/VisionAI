"""VisionAI - Milestone 6: Event Database Module

Provides persistent SQLite storage for real-time detection and recognition events.

Key Engineering Highlights:
1. Normalized relational schema with Primary Key, ISO 8601 Timestamps, and serialized bounding boxes.
2. Smart Rate-Limiting & Cooldown Management:
   Prevents database thrashing by throttling redundant detections of the same identity or object
   within a configurable cooldown window (e.g., 4 seconds).
3. Seamless integration with Pandas DataFrames for downstream analytics.
"""

import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATABASE_PATH, LOG_COOLDOWN_SECONDS


class EventDatabase:
    """Manages SQLite storage for computer vision detection events."""

    def __init__(self, db_path: Path = DATABASE_PATH, cooldown_seconds: float = LOG_COOLDOWN_SECONDS) -> None:
        """Initializes database connection and creates table schema if needed.

        Args:
            db_path: Path to the SQLite database file.
            cooldown_seconds: Minimum seconds before logging duplicate label.
        """
        self.db_path = db_path
        self.cooldown_seconds = cooldown_seconds

        # Cooldown cache: {label: last_logged_epoch_timestamp}
        self._cooldown_cache: Dict[str, float] = {}

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a thread-safe connection to the SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initializes table schema with indexes for fast querying."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    detection_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    bbox TEXT
                )
            """)
            # Create indexes on timestamp and label for high-speed queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_label ON detections(label)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON detections(detection_type)")
            conn.commit()

    def log_event(
        self,
        detection_type: str,
        label: str,
        confidence: float,
        source: str = "webcam",
        bbox: Optional[Tuple[int, int, int, int]] = None,
        enforce_cooldown: bool = True,
    ) -> bool:
        """Logs a single detection event into SQLite if not rate-limited.

        Args:
            detection_type: 'face', 'person', or 'object'.
            label: Name or class label (e.g. 'Priyanka', 'laptop', 'Unknown').
            confidence: Float confidence/similarity (0.0 to 1.0).
            source: 'webcam', 'upload', or 'test'.
            bbox: (x, y, w, h) bounding box coordinates.
            enforce_cooldown: Whether to check cooldown cache.

        Returns:
            bool: True if event was written to disk, False if throttled.
        """
        now = time.time()

        if enforce_cooldown:
            last_time = self._cooldown_cache.get(label, 0.0)
            if (now - last_time) < self.cooldown_seconds:
                return False  # Cooldown active, skip duplicate log

        # Update cache
        self._cooldown_cache[label] = now

        iso_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox_str = json.dumps(list(bbox)) if bbox is not None else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO detections (timestamp, detection_type, label, confidence, source, bbox)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (iso_timestamp, detection_type, label, float(confidence), source, bbox_str))
            conn.commit()

        return True

    def log_vision_result(self, vision_result: Any, source: str = "webcam") -> int:
        """Logs all entities from a VisionResult instance.

        Args:
            vision_result: VisionResult object from CombinedVisionEngine.
            source: Source descriptor.

        Returns:
            int: Number of new records logged.
        """
        logged_count = 0
        for entity in vision_result.fused_entities:
            wrote = self.log_event(
                detection_type=entity["type"],
                label=entity["label"],
                confidence=entity["confidence"],
                source=source,
                bbox=entity["box"],
                enforce_cooldown=True,
            )
            if wrote:
                logged_count += 1
        return logged_count

    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent events ordered by descending timestamp."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, timestamp, detection_type, label, confidence, source, bbox
                FROM detections
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_all_events_df(self) -> pd.DataFrame:
        """Loads all events directly into a Pandas DataFrame."""
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM detections ORDER BY id ASC", conn)
            if not df.empty and "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    def get_total_count(self) -> int:
        """Returns the total number of records in the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM detections")
            return cursor.fetchone()[0]

    def clear_database(self) -> None:
        """Wipes all logged detections (useful for testing or resetting history)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM detections")
            conn.commit()
        self._cooldown_cache.clear()
        print("[Database] All detection records cleared.")


def main() -> None:
    """CLI test for SQLite event logging."""
    db = EventDatabase()
    print(f"\n[Database Test] Initialized database at: {db.db_path}")

    # Insert sample records
    db.log_event("person", "Lena", 0.98, source="test", bbox=(50, 60, 100, 120), enforce_cooldown=False)
    db.log_event("object", "laptop", 0.94, source="test", bbox=(120, 200, 250, 150), enforce_cooldown=False)
    db.log_event("face", "Unknown", 0.72, source="test", bbox=(300, 80, 80, 80), enforce_cooldown=False)

    print(f"[Database Test] Total records in DB: {db.get_total_count()}")
    recent = db.get_recent_events(limit=5)
    print("\nRecent 5 Events:")
    for r in recent:
        print(f"  ID #{r['id']} | {r['timestamp']} | {r['detection_type']} | {r['label']} | {r['confidence']*100:.1f}% | {r['source']}")


if __name__ == "__main__":
    main()

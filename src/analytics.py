"""VisionAI - Milestone 7: Analytics & Visualization Module

Performs business intelligence and quantitative analytics over SQLite event history using Pandas.
Generates publication-quality charts using Matplotlib.

Calculates:
1. Total detections & unique entities
2. Top detected objects and frequencies
3. Recognized people vs. Unknown faces
4. Average confidence and distribution
5. Temporal detection velocity over time
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ANALYTICS_DIR, DATABASE_PATH
from src.database import EventDatabase


class AnalyticsEngine:
    """Analytics engine for extracting insights and generating charts from VisionAI events."""

    def __init__(self, db: Optional[EventDatabase] = None) -> None:
        """Initializes the analytics engine.

        Args:
            db: Optional EventDatabase instance.
        """
        self.db = db or EventDatabase()
        ANALYTICS_DIR.mkdir(parents=True, exist_ok=True)

        # Set clean modern style for Matplotlib
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    def compute_summary_kpis(self) -> Dict[str, Any]:
        """Calculates key performance metrics from the event history.

        Returns:
            Dict containing total_detections, top_object, known_count, unknown_count,
            avg_confidence, unique_classes.
        """
        df = self.db.get_all_events_df()
        if df.empty:
            return {
                "total_detections": 0,
                "top_object": "None",
                "top_object_count": 0,
                "recognized_faces_count": 0,
                "unknown_faces_count": 0,
                "average_confidence": 0.0,
                "unique_labels": 0,
                "df": df,
            }

        total = len(df)
        avg_conf = float(df["confidence"].mean())

        # Object vs Face splits
        objects_df = df[df["detection_type"] == "object"]
        faces_df = df[df["detection_type"].isin(["face", "person"])]

        # Most detected object
        if not objects_df.empty:
            top_obj_series = objects_df["label"].value_counts()
            top_obj = top_obj_series.index[0]
            top_obj_count = int(top_obj_series.iloc[0])
        else:
            top_obj = "None"
            top_obj_count = 0

        # Known vs Unknown face counts
        unknown_faces = len(faces_df[faces_df["label"] == "Unknown"])
        known_faces = len(faces_df[faces_df["label"] != "Unknown"])

        return {
            "total_detections": total,
            "top_object": top_obj,
            "top_object_count": top_obj_count,
            "recognized_faces_count": known_faces,
            "unknown_faces_count": unknown_faces,
            "average_confidence": avg_conf,
            "unique_labels": int(df["label"].nunique()),
            "df": df,
        }

    def generate_all_charts(self, save_dir: Path = ANALYTICS_DIR) -> Dict[str, Path]:
        """Generates and saves all analytic charts to disk.

        Returns:
            Dict mapping chart names to their saved file paths.
        """
        df = self.db.get_all_events_df()
        save_dir.mkdir(parents=True, exist_ok=True)
        chart_paths: Dict[str, Path] = {}

        if df.empty:
            print("[Analytics] Database empty. No charts generated.")
            return chart_paths

        # 1. Class Distribution Bar Chart
        p1 = save_dir / "class_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
        counts = df["label"].value_counts().head(10)
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(counts)))
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_title("Top Detected Classes & Identities", fontsize=13, fontweight="bold", pad=12)
        ax.set_ylabel("Detection Count", fontsize=10)
        ax.set_xlabel("Entity Label", fontsize=10)
        plt.xticks(rotation=30, ha="right")
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        plt.savefig(p1)
        plt.close(fig)
        chart_paths["class_distribution"] = p1

        # 2. Face Recognition Breakdown Donut Chart
        p2 = save_dir / "face_recognition_breakdown.png"
        faces_df = df[df["detection_type"].isin(["face", "person"])]
        if not faces_df.empty:
            fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
            face_counts = faces_df["label"].value_counts()
            colors_pie = plt.cm.Set2(np.linspace(0, 1, len(face_counts)))
            wedges, texts, autotexts = ax.pie(
                face_counts.values,
                labels=face_counts.index,
                autopct="%1.1f%%",
                startangle=140,
                colors=colors_pie,
                wedgeprops=dict(width=0.45, edgecolor="white"),
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_weight("bold")
            ax.set_title("Face Identity Breakdown (Known vs Unknown)", fontsize=12, fontweight="bold")
            plt.tight_layout()
            plt.savefig(p2)
            plt.close(fig)
            chart_paths["face_breakdown"] = p2

        # 3. Confidence Distribution Histogram
        p3 = save_dir / "confidence_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.hist(df["confidence"] * 100, bins=15, color="#2b5c8f", edgecolor="white", alpha=0.85)
        ax.set_title("Detection Confidence Distribution", fontsize=12, fontweight="bold")
        ax.set_xlabel("Confidence / Similarity Score (%)", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.axvline(df["confidence"].mean() * 100, color="red", linestyle="--", linewidth=1.5, label=f"Mean ({df['confidence'].mean()*100:.1f}%)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(p3)
        plt.close(fig)
        chart_paths["confidence_distribution"] = p3

        # 4. Timeline of Events
        p4 = save_dir / "timeline_detections.png"
        fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
        # Resample by 5-minute intervals or plot cumulative
        df_sorted = df.sort_values("timestamp")
        df_sorted["cumulative_count"] = range(1, len(df_sorted) + 1)
        ax.plot(df_sorted["timestamp"], df_sorted["cumulative_count"], color="#107c41", linewidth=2, marker="o", markersize=3)
        ax.set_title("Cumulative Detections Timeline", fontsize=12, fontweight="bold")
        ax.set_xlabel("Event Timestamp", fontsize=10)
        ax.set_ylabel("Total Cumulative Detections", fontsize=10)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        plt.xticks(rotation=25)
        plt.tight_layout()
        plt.savefig(p4)
        plt.close(fig)
        chart_paths["timeline"] = p4

        print(f"[Analytics] Generated {len(chart_paths)} charts in: {save_dir}")
        return chart_paths


def main() -> None:
    """CLI test for AnalyticsEngine."""
    engine = AnalyticsEngine()
    kpis = engine.compute_summary_kpis()

    print("\n" + "=" * 55)
    print("           VisionAI - Analytics Summary KPI")
    print("=" * 55)
    print(f"  * Total Logged Events    : {kpis['total_detections']}")
    print(f"  * Top Detected Object    : {kpis['top_object']} ({kpis['top_object_count']} times)")
    print(f"  * Recognized Faces       : {kpis['recognized_faces_count']}")
    print(f"  * Unknown Faces          : {kpis['unknown_faces_count']}")
    print(f"  * Average Confidence     : {kpis['average_confidence']*100:.1f}%")
    print(f"  * Unique Entities Tracked: {kpis['unique_labels']}")
    print("=" * 55 + "\n")

    charts = engine.generate_all_charts()
    for name, path in charts.items():
        print(f"  [Chart Saved] {name:22s} -> {path.name}")


if __name__ == "__main__":
    main()

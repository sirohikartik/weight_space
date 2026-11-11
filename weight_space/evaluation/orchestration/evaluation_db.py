"""Structured SQLite persistence and reporting for frontier evaluation runs."""

from typing import Any, Dict, List, Optional, Tuple
import datetime
import json
import sqlite3


class EvaluationDB:
    """Manages SQLite storage and tabular reporting for steering benchmarks."""

    def __init__(self, db_path: str = "frontier_benchmark.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Experiments table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                target_trait TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                target_norm_c REAL NOT NULL,
                created_at TEXT NOT NULL,
                notes TEXT
            );
            """)

            # Main condition evaluations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS condition_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                adherence_pct REAL NOT NULL,
                adherence_ci_lower REAL NOT NULL,
                adherence_ci_upper REAL NOT NULL,
                collapse_pct REAL NOT NULL,
                collapse_ci_lower REAL NOT NULL,
                collapse_ci_upper REAL NOT NULL,
                collapse_count INTEGER NOT NULL,
                total_probes INTEGER NOT NULL,
                corpus_ppl REAL,
                refusal_cosine REAL,
                refusal_energy_pct REAL,
                asr_pct REAL,
                asr_ci_lower REAL,
                asr_ci_upper REAL,
                asr_delta REAL,
                scheduling_strategy TEXT DEFAULT 'constant',
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            """)

            # Dynamic schedule sweep table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS schedule_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                condition_name TEXT NOT NULL,
                strategy TEXT NOT NULL,
                alpha_0 REAL NOT NULL,
                decay_steps INTEGER,
                gamma REAL,
                adherence_pct REAL NOT NULL,
                collapse_pct REAL NOT NULL,
                corpus_ppl REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            """)

            # Layer sweep evaluations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS layer_sweep_evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                layer_idx INTEGER NOT NULL,
                condition_name TEXT NOT NULL,
                adherence_pct REAL NOT NULL,
                collapse_pct REAL NOT NULL,
                corpus_ppl REAL,
                refusal_cosine REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
            );
            """)

            conn.commit()

    def create_experiment(
        self,
        experiment_id: str,
        model_name: str,
        target_trait: str,
        layer_idx: int,
        target_norm_c: float,
        notes: Optional[str] = None,
    ) -> str:
        """Records a new experiment session."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO experiments (
                experiment_id, model_name, target_trait, layer_idx, target_norm_c, created_at, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (experiment_id, model_name, target_trait, layer_idx, target_norm_c, now, notes))
            conn.commit()
        return experiment_id

    def log_condition_evaluation(
        self,
        experiment_id: str,
        condition_name: str,
        adherence_pct: float,
        adherence_ci: Tuple[float, float],
        collapse_pct: float,
        collapse_ci: Tuple[float, float],
        collapse_count: int,
        total_probes: int,
        corpus_ppl: Optional[float] = None,
        refusal_cosine: Optional[float] = None,
        refusal_energy_pct: Optional[float] = None,
        asr_pct: Optional[float] = None,
        asr_ci: Optional[Tuple[float, float]] = None,
        asr_delta: Optional[float] = None,
        scheduling_strategy: str = "constant",
    ) -> None:
        """Logs evaluation metrics for a single steering condition."""
        asr_low = asr_ci[0] if asr_ci is not None else None
        asr_high = asr_ci[1] if asr_ci is not None else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO condition_evaluations (
                experiment_id, condition_name, adherence_pct, adherence_ci_lower, adherence_ci_upper,
                collapse_pct, collapse_ci_lower, collapse_ci_upper, collapse_count, total_probes,
                corpus_ppl, refusal_cosine, refusal_energy_pct, asr_pct, asr_ci_lower, asr_ci_upper,
                asr_delta, scheduling_strategy
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                experiment_id,
                condition_name,
                adherence_pct,
                adherence_ci[0],
                adherence_ci[1],
                collapse_pct,
                collapse_ci[0],
                collapse_ci[1],
                collapse_count,
                total_probes,
                corpus_ppl,
                refusal_cosine,
                refusal_energy_pct,
                asr_pct,
                asr_low,
                asr_high,
                asr_delta,
                scheduling_strategy,
            ))
            conn.commit()

    def log_layer_sweep(
        self,
        experiment_id: str,
        layer_idx: int,
        condition_name: str,
        adherence_pct: float,
        collapse_pct: float,
        corpus_ppl: Optional[float] = None,
        refusal_cosine: Optional[float] = None,
    ) -> None:
        """Logs intermediate layer sweep metrics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO layer_sweep_evaluations (
                experiment_id, layer_idx, condition_name, adherence_pct, collapse_pct, corpus_ppl, refusal_cosine
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """, (experiment_id, layer_idx, condition_name, adherence_pct, collapse_pct, corpus_ppl, refusal_cosine))
            conn.commit()

    def get_experiment_summary(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Retrieves all condition evaluations for an experiment as a list of dicts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM condition_evaluations WHERE experiment_id = ? ORDER BY id ASC;
            """, (experiment_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def generate_latex_table(self, experiment_id: str) -> str:
        """Generates publication-ready LaTeX table code for the experiment results."""
        records = self.get_experiment_summary(experiment_id)
        if not records:
            return "% No records found for experiment."

        lines = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\small",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"\textbf{Condition} & \textbf{Adherence (95\% CI)} & \textbf{Collapse (n/N)} & \textbf{Collapse CI} & \textbf{Corpus PPL} & $\cos(\Delta x, v_{\text{ref}})$ & \textbf{ASR $\Delta$} \\",
            r"\midrule",
        ]

        for r in records:
            cond = r["condition_name"].replace("_", r"\_")
            adh = f"{r['adherence_pct']:.1f}\\% [{r['adherence_ci_lower']:.1f}, {r['adherence_ci_upper']:.1f}]"
            col_frac = f"{r['collapse_count']}/{r['total_probes']}"
            col_ci = f"[{r['collapse_ci_lower']:.1f}, {r['collapse_ci_upper']:.1f}]\\%"
            ppl = f"{r['corpus_ppl']:.2f}" if r["corpus_ppl"] is not None else "--"
            cos_ref = f"{r['refusal_cosine']:.3f}" if r["refusal_cosine"] is not None else "--"
            asr_d = f"{r['asr_delta']:+.1f}\\%" if r["asr_delta"] is not None else "--"

            lines.append(f"{cond} & {adh} & {col_frac} & {col_ci} & {ppl} & {cos_ref} & {asr_d} \\\\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\caption{Controlled frontier evaluation holding injected Euclidean norm strictly constant ($\|\Delta x\|_2 = c$). Metrics report Wilson score 95\% confidence intervals.}",
            r"\label{tab:frontier_controlled_evaluation}",
            r"\end{table*}",
        ])

        return "\n".join(lines)

"""Persistent SQLite datastore and dynamic runtime controller for activation coordinate recipes."""

import json
import sqlite3
import time
from typing import Dict, List, Optional
import torch
import torch.nn as nn
try:
    from transformers import PreTrainedModel, PreTrainedTokenizer
except ImportError:
    PreTrainedModel = nn.Module  # type: ignore
    PreTrainedTokenizer = object  # type: ignore


class ActivationDatastore:
    """Persistent SQLite store for sparse physical coordinate recipes."""

    def __init__(self, db_path: str = "activation_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS recipes (
                    task_tag TEXT PRIMARY KEY,
                    layer_idx INTEGER,
                    alpha REAL,
                    pos_coords TEXT,
                    neg_coords TEXT,
                    created_at REAL
                )
            """)

    def put_recipe(
        self,
        task_tag: str,
        layer_idx: int,
        alpha: float,
        pos_coords: Dict[int, float],
        neg_coords: Dict[int, float],
    ) -> None:
        """Stores or updates a behavioral recipe in the datastore."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO recipes (task_tag, layer_idx, alpha, pos_coords, neg_coords, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (task_tag, layer_idx, alpha, json.dumps(pos_coords), json.dumps(neg_coords), time.time()),
            )

    def get_recipe(self, task_tag: str) -> Optional[Dict]:
        """Retrieves a recipe by its task tag."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT layer_idx, alpha, pos_coords, neg_coords FROM recipes WHERE task_tag = ?",
                (task_tag,),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "layer_idx": row[0],
                "alpha": row[1],
                "pos_coords": {int(k): v for k, v in json.loads(row[2]).items()},
                "neg_coords": {int(k): v for k, v in json.loads(row[3]).items()},
            }

    def list_recipes(self) -> List[str]:
        """Lists all registered recipe tags in the datastore."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            cur.execute("SELECT task_tag FROM recipes ORDER BY task_tag ASC")
            return [row[0] for row in cur.fetchall()]


class ModelActivationController:
    """Controls model behavior dynamically by querying an Activation Datastore and steering raw neuron coordinates."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        datastore: ActivationDatastore,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.datastore = datastore
        self.active_hook = None
        self.current_behavior = None

    def activate_behavior(self, task_tag: str, alpha_override: Optional[float] = None) -> bool:
        """Loads a recipe from the datastore and attaches an activation hook."""
        self.deactivate()
        recipe = self.datastore.get_recipe(task_tag)
        if not recipe:
            return False

        layer_idx = recipe["layer_idx"]
        alpha = alpha_override if alpha_override is not None else recipe["alpha"]
        coords = {**recipe["pos_coords"], **recipe["neg_coords"]}
        d_model = self.model.config.hidden_size
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        steer_vec = torch.zeros(d_model, device=device, dtype=dtype)
        for idx, w in coords.items():
            steer_vec[idx] = w

        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            steered = hidden + alpha * steer_vec.view(1, 1, -1)
            return (steered,) + out[1:] if isinstance(out, tuple) else steered

        target_layer = self.model.model.layers[layer_idx]
        self.active_hook = target_layer.register_forward_hook(hook_fn)
        self.current_behavior = task_tag
        return True

    def deactivate(self) -> None:
        """Removes any active steering hook and restores the model to baseline."""
        if self.active_hook is not None:
            self.active_hook.remove()
            self.active_hook = None
            self.current_behavior = None

    def query(self, prompt: str, max_new_tokens: int = 65, temperature: float = 0.7) -> str:
        """Runs autoregressive text generation under current activation state."""
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)
        if txt.startswith(prompt):
            return txt[len(prompt) :].strip()
        return txt.strip()

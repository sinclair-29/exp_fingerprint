from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from llmfp.core.io import load_jsonl
from llmfp.core.matching import match_keyword_target
from llmfp.core.templates import get_template
from llmfp.methods.base import FingerprintMethod, artifact_get, result_dicts
from llmfp.schemas import FingerprintArtifact, FingerprintTask, VerificationResult


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = Path.cwd() / resolved
    return resolved


def _copyright_tokens(cfg: dict[str, Any]) -> list[str]:
    configured = list(cfg.get("copyright_tokens") or [])
    count = int(cfg.get("num_adv_tokens", len(configured) or 1))
    if not configured:
        configured = [f"<COPYRIGHT_TOKEN_{index}>" for index in range(count)]
    if len(configured) < count:
        configured.extend(f"<COPYRIGHT_TOKEN_{index}>" for index in range(len(configured), count))
    return configured[:count]


def _template_names(cfg: dict[str, Any], model_backend) -> list[str]:
    names = list(cfg.get("templates") or ["default"])
    return [model_backend.template_name if name == "default" else name for name in names]


def _query_from_row(row: dict[str, Any]) -> str:
    return str(row.get("query") or row.get("question") or row.get("base_prompt") or row.get("prompt") or "")


def tokenizer_contains_tokens(tokenizer, tokens: list[str]) -> bool:
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            return False
        if unk_id is not None and int(token_id) == int(unk_id):
            added = getattr(tokenizer, "added_tokens_encoder", {}) or {}
            if token not in added:
                return False
    return True


class PlugAEMethod(FingerprintMethod):
    name = "plugae"

    def build_tasks(self, cfg: dict[str, Any]) -> list[FingerprintTask]:
        data_path = _resolve_path(cfg.get("query_set_path", cfg.get("questions_path", "data/plugae_queries.jsonl")))
        rows = load_jsonl(data_path)
        limit = int(cfg.get("num_fingerprints", cfg.get("num_queries", len(rows) or 0)))
        tasks: list[FingerprintTask] = []
        for index, row in enumerate(rows[:limit]):
            query = _query_from_row(row)
            target = str(row["target"])
            tasks.append(
                FingerprintTask(
                    task_id=str(row.get("id", f"plugae-{index + 1}")),
                    method=self.name,
                    input_text=query,
                    target=target,
                    metadata={
                        "target_keywords": row.get("target_keywords", [target]),
                        "row": row,
                    },
                )
            )
        return tasks

    def _full_query(self, copyright_tokens: list[str], query: str, insertion_position: str) -> str:
        token_text = " ".join(copyright_tokens)
        if insertion_position == "suffix":
            return f"{query} {token_text}".strip()
        if insertion_position != "prefix":
            raise ValueError(f"Unsupported PlugAE insertion_position: {insertion_position}")
        return f"{token_text} {query}".strip()

    def _split_template(self, template_name: str, query: str, insertion_position: str) -> tuple[str, str]:
        template = get_template(template_name)
        if insertion_position == "suffix":
            return template.split_around_mutable(f"{query} ", "")
        return template.split_around_mutable("", f" {query}")

    def _optimize_embeddings(
        self,
        task: FingerprintTask,
        model_backend,
        cfg: dict[str, Any],
        init_embeds,
    ) -> dict[str, Any]:
        import torch
        import torch.nn.functional as F

        tokenizer = model_backend.tokenizer
        model = model_backend.model
        device = model_backend.device
        embedding_layer = model.get_input_embeddings()
        templates = _template_names(cfg, model_backend)
        insertion_position = str(cfg.get("insertion_position", "prefix"))
        target_ids = tokenizer(task.target, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        if target_ids.numel() == 0:
            raise ValueError("PlugAE target must tokenize to at least one token")

        adv_embeds = init_embeds.detach().clone().to(device=device, dtype=embedding_layer.weight.dtype)
        adv_embeds.requires_grad_(True)
        optimizer = torch.optim.Adam([adv_embeds], lr=float(cfg.get("lr", 0.1)))

        def text_embeds(text: str):
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
            if ids.shape[1] == 0:
                return None
            return embedding_layer(ids)

        def loss_for_template(template_name: str):
            before_text, after_text = self._split_template(template_name, task.input_text, insertion_position)
            pieces = []
            before = text_embeds(before_text)
            after = text_embeds(after_text)
            if before is not None:
                pieces.append(before)
            pieces.append(adv_embeds.unsqueeze(0))
            if after is not None:
                pieces.append(after)
            non_target_len = sum(piece.shape[1] for piece in pieces)
            target_embeds = embedding_layer(target_ids.unsqueeze(0))
            inputs_embeds = torch.cat([*pieces, target_embeds], dim=1)
            outputs = model(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, non_target_len - 1 : non_target_len + target_ids.numel() - 1, :]
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))

        epochs = int(cfg.get("epochs", cfg.get("num_steps", 30)))
        loss_curve: list[float] = []
        best_loss = math.inf
        best_step = 0
        best_embeds = adv_embeds.detach().clone()
        num_forward = 0
        num_backward = 0
        for step in range(epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            losses = [loss_for_template(name) for name in templates]
            num_forward += len(templates)
            loss = sum(losses) / len(losses)
            current = float(loss.detach().cpu().item())
            loss_curve.append(current)
            if current < best_loss:
                best_loss = current
                best_step = step
                best_embeds = adv_embeds.detach().clone()
            if step == epochs:
                break
            loss.backward()
            num_backward += 1
            optimizer.step()

        return {
            "embeddings": best_embeds.detach(),
            "best_loss": best_loss,
            "best_step": best_step,
            "loss_curve": loss_curve,
            "num_forward": num_forward,
            "num_backward": num_backward,
            "templates": templates,
        }

    def _protected_output_dir(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> Path:
        source_name = str(model_backend.name).replace("/", "_")
        configured = str(cfg.get("protected_model_output_dir", f"artifacts/plugae_protected/{source_name}"))
        configured = configured.replace("<source_model>", source_name)
        base = _resolve_path(configured)
        return base / task.task_id if bool(cfg.get("separate_artifact_per_fingerprint", True)) else base

    def construct(self, task: FingerprintTask, model_backend, cfg: dict[str, Any]) -> FingerprintArtifact:
        import torch

        model_backend.load()
        tokenizer = model_backend.tokenizer
        model = model_backend.model
        copyright_tokens = _copyright_tokens(cfg)
        added_tokens = int(tokenizer.add_tokens(copyright_tokens, special_tokens=False))
        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        model.requires_grad_(False)

        token_ids = tokenizer.convert_tokens_to_ids(copyright_tokens)
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=model_backend.device)
        init_embeds = model.get_input_embeddings()(token_ids_tensor).detach()
        result = self._optimize_embeddings(task, model_backend, cfg, init_embeds)
        optimized_embeds = result["embeddings"]
        with torch.no_grad():
            model.get_input_embeddings().weight[token_ids_tensor] = optimized_embeds.to(model.get_input_embeddings().weight.dtype)

        protected_dir = self._protected_output_dir(task, model_backend, cfg)
        protected_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(protected_dir)
        model.save_pretrained(protected_dir)

        insertion_position = str(cfg.get("insertion_position", "prefix"))
        full_query = self._full_query(copyright_tokens, task.input_text, insertion_position)
        prompt_text = get_template(result["templates"][0]).render(full_query, assistant_prefix="")
        embedding_norm = float(optimized_embeds.float().norm(dim=1).mean().detach().cpu().item())
        return FingerprintArtifact(
            fingerprint_id=f"plugae-{task.task_id}",
            method=self.name,
            base_model=model_backend.name,
            task_id=task.task_id,
            prompt_text=prompt_text,
            optimized_text=" ".join(copyright_tokens),
            target=task.target,
            best_loss=float(result["best_loss"]),
            best_step=int(result["best_step"]),
            metadata={
                "query": task.input_text,
                "target_keywords": task.metadata.get("target_keywords", [task.target]),
                "copyright_tokens": copyright_tokens,
                "copyright_token_ids": [int(token_id) for token_id in token_ids],
                "optimized_embedding_norm": embedding_norm,
                "loss_history": result["loss_curve"],
                "protected_model_path": str(protected_dir),
                "added_tokens": added_tokens,
                "templates": result["templates"],
                "insertion_position": insertion_position,
                "full_query": full_query,
                "num_forward": int(result["num_forward"]),
                "num_backward": int(result["num_backward"]),
                "optimizer": {
                    "lr": float(cfg.get("lr", 0.1)),
                    "epochs": int(cfg.get("epochs", cfg.get("num_steps", 30))),
                },
            },
        )

    def verify(self, artifact, suspect_backend, cfg: dict[str, Any]) -> VerificationResult:
        suspect_backend.load()
        metadata = artifact_get(artifact, "metadata", {})
        copyright_tokens = list(metadata.get("copyright_tokens") or _copyright_tokens(cfg))
        insertion_position = str(metadata.get("insertion_position") or cfg.get("insertion_position", "prefix"))
        query = str(metadata.get("query") or artifact_get(artifact, "prompt_text", ""))
        contains_tokens = tokenizer_contains_tokens(suspect_backend.tokenizer, copyright_tokens)
        full_query = self._full_query(copyright_tokens, query, insertion_position)
        template_name = cfg.get("suspect_template") or suspect_backend.template_name
        prompt_text = get_template(template_name).render(full_query, assistant_prefix="")
        output = suspect_backend.generate(
            prompt_text,
            max_new_tokens=int(cfg.get("max_new_tokens", 48)),
            temperature=float(cfg.get("temperature", 1.0)),
            top_p=float(cfg.get("top_p", 1.0)),
            do_sample=bool(cfg.get("do_sample", False)),
        )
        success = match_keyword_target(output, artifact_get(artifact, "target"), metadata.get("target_keywords", []))
        return VerificationResult(
            method=self.name,
            base_model=artifact_get(artifact, "base_model"),
            suspect_model=suspect_backend.name,
            fingerprint_id=artifact_get(artifact, "fingerprint_id"),
            success=success,
            score=1.0 if success else 0.0,
            raw_output=output,
            metadata={
                "copyright_tokens": copyright_tokens,
                "suspect_contains_copyright_tokens": contains_tokens,
                "query": query,
                "full_query": full_query,
                "target_response_rate": 1.0 if success else 0.0,
            },
        )

    def aggregate(self, results, cfg: dict[str, Any]) -> dict[str, Any]:
        rows = result_dicts(results)
        if not rows:
            return {"trr": 0.0, "target_response_rate": 0.0}
        trr = sum(float(row.get("score", 0.0)) for row in rows) / len(rows)
        return {"trr": trr, "target_response_rate": trr}

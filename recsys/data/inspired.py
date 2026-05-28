import json
from dataclasses import dataclass, field
from pathlib import Path

from recsys.config import INSPIRED_ANNOTATED_PATH, MOVIE_ID_MAP_PATH


@dataclass
class Turn:
    role: str
    text: str


@dataclass
class Dialog:
    dialog_id: str
    turns: list[Turn]
    user_query: str
    recommended_title: str
    target_imdb_id: str
    history_imdb_ids: list[str]
    movie_sentiments: dict[str, str] = field(default_factory=dict)
    genre_sentiments: dict[str, str] = field(default_factory=dict)
    # {mention_title: imdb_id} for all non-target history movies; used to
    # filter history_imdb_ids when truncating dialogs to intermediate cutoffs.
    mention_map: dict[str, str] = field(default_factory=dict)


def _parse_full_text(full_text: str) -> list[Turn]:
    turns = []
    for line in full_text.splitlines():
        if line.startswith("SEEKER:"):
            turns.append(Turn(role="seeker", text=line[len("SEEKER:") :].strip()))
        elif line.startswith("RECOMMENDER:"):
            turns.append(
                Turn(role="recommender", text=line[len("RECOMMENDER:") :].strip())
            )
    return turns


def load_dialogs(
    annotated_path: Path = INSPIRED_ANNOTATED_PATH,
    id_map_path: Path = MOVIE_ID_MAP_PATH,
) -> list[Dialog]:
    id_map = json.loads(id_map_path.read_text())

    dialogs = []
    with open(annotated_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            dialog_id = row["dialog_id"]
            movie_map = id_map.get(dialog_id, {})

            target_title = row.get("recommended_movie", "")
            target_imdb_id = movie_map.get(target_title)
            if not target_imdb_id:
                continue

            mention_map: dict[str, str] = {}
            history_imdb_ids = []
            for title in row.get("movies", {}):
                mid = movie_map.get(title)
                if mid and mid != target_imdb_id:
                    mention_map[title] = mid
                    history_imdb_ids.append(mid)

            dialogs.append(
                Dialog(
                    dialog_id=dialog_id,
                    turns=_parse_full_text(row.get("full_text", "")),
                    user_query=row.get("user_query", ""),
                    recommended_title=target_title,
                    target_imdb_id=target_imdb_id,
                    history_imdb_ids=history_imdb_ids,
                    movie_sentiments=row.get("movies", {}),
                    genre_sentiments=row.get("genres", {}),
                    mention_map=mention_map,
                )
            )

    return dialogs


def dialog_context_before_recommendation(dialog: Dialog) -> list[Turn]:
    target_lower = dialog.recommended_title.lower()
    cutoff = len(dialog.turns)
    for i, turn in enumerate(dialog.turns):
        if (
            turn.role == "recommender"
            and target_lower
            and target_lower in turn.text.lower()
        ):
            cutoff = i
            break
    return dialog.turns[:cutoff]

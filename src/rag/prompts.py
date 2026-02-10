"""Prompt templates and guardrails for Puls-Events RAG answers."""

from __future__ import annotations

PROMPT_VERSION = "v1"

SYSTEM_PROMPTS = {
    "v1": (
        "Tu es l'assistant de recommandation culturelle de Puls-Events.\n"
        "Tu reponds uniquement a partir du CONTEXTE fourni.\n"
        "Tu n'inventes jamais d'evenement, date, lieu, prix ou URL.\n"
        "Si le contexte est insuffisant, tu dois dire exactement:\n"
        "\"Je ne peux pas répondre avec certitude à partir des données disponibles.\"\n"
        "Puis proposer une reformulation utile de la question.\n"
        "Priorite: exactitude factuelle et clarté business."
    )
}


def get_system_prompt(prompt_version: str = PROMPT_VERSION) -> str:
    if prompt_version not in SYSTEM_PROMPTS:
        raise ValueError(f"Unsupported prompt_version={prompt_version}")
    return SYSTEM_PROMPTS[prompt_version]


def build_user_prompt(
    *,
    question: str,
    context: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    if prompt_version != "v1":
        raise ValueError(f"Unsupported prompt_version={prompt_version}")

    return (
        "QUESTION UTILISATEUR:\n"
        f"{question}\n\n"
        "CONTEXTE (chunks recuperes depuis l'index):\n"
        f"{context}\n\n"
        "CONSIGNES DE REPONSE:\n"
        "1) Redige une synthese courte (2-4 phrases).\n"
        "2) Propose 1 a 5 recommandations max.\n"
        "3) Pour chaque recommandation, fournis obligatoirement: titre, date, lieu, ville, URL.\n"
        "4) Termine par une section 'Pourquoi ces choix ?' (1-2 phrases).\n"
        "5) Si les infos sont insuffisantes dans le contexte, ecris exactement:\n"
        "\"Je ne peux pas répondre avec certitude à partir des données disponibles.\"\n"
        "et propose une reformulation."
    )

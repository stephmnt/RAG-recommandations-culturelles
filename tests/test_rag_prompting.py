import pytest

from src.rag.prompts import build_user_prompt, get_system_prompt


def test_prompt_contains_guardrails():
    system_prompt = get_system_prompt("v1")

    assert "Tu reponds uniquement a partir du CONTEXTE fourni." in system_prompt
    assert "Tu n'inventes jamais" in system_prompt
    assert "Je ne peux pas répondre avec certitude à partir des données disponibles." in system_prompt


def test_user_prompt_contains_required_structure():
    prompt = build_user_prompt(
        question="Quels concerts jazz a Montpellier cette semaine ?",
        context="[EVENT_ID=evt-1] ...",
        prompt_version="v1",
    )

    assert "QUESTION UTILISATEUR" in prompt
    assert "CONTEXTE" in prompt
    assert "Pourquoi ces choix ?" in prompt
    assert "titre, date, lieu, ville, URL" in prompt


def test_unsupported_prompt_version_raises():
    with pytest.raises(ValueError):
        get_system_prompt("v2")

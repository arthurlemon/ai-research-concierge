"""Simple CLI entrypoint for DocMana."""

import asyncio

from .graph import app


def _initial_state(user_query: str):
    return {
        "messages": [],
        "user_query": user_query,
        "sub_questions": [],
        "tool_results": {},
        "final_answer": None,
        "error": None,
    }


def main():
    user_query = input("Question de l'utilisateur : ")
    result = asyncio.run(app.ainvoke(_initial_state(user_query)))
    print("\n--- Réponse ---\n")
    print(result.get("final_answer"))


if __name__ == "__main__":
    main()

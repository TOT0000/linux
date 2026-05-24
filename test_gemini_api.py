import os
from openai import OpenAI


def mask_key(secret: str | None) -> str:
    value = "" if secret is None else str(secret).strip()
    if not value:
        return "(empty)"
    if len(value) <= 10:
        return f"{value[:2]}***{value[-2:]}"
    return f"{value[:6]}...{value[-4:]}"


def main():
    base_url = os.environ.get(
        "GEMINI_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    model = "gemini-2.5-flash"
    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    ).strip()

    print("[DEBUG] base_url =", base_url)
    print("[DEBUG] model =", model)
    print("[DEBUG] has_gemini_key =", bool(os.environ.get("GEMINI_API_KEY")))
    print("[DEBUG] has_google_key =", bool(os.environ.get("GOOGLE_API_KEY")))
    print("[DEBUG] key_len =", len(api_key))
    print("[DEBUG] key_masked =", mask_key(api_key))

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: pong"}],
        temperature=0.0,
    )
    print("[DEBUG] reply =", resp.choices[0].message.content)


if __name__ == "__main__":
    main()

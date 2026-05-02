from dotenv import load_dotenv

from llm_wiki.app import LLMWikiApp


def main() -> None:
    load_dotenv()
    LLMWikiApp().run()


if __name__ == "__main__":
    main()

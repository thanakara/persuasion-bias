from langchain_core.documents import Document


def join_documents(documents: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in documents)

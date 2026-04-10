from logging import getLogger

import hydra

from dotenv import load_dotenv
from omegaconf import DictConfig
from hydra.utils import instantiate
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_core.document_loaders import BaseLoader

from persuasion_bias import chat
from persuasion_bias.utils import CONFIG_DIR
from persuasion_bias.agents import make_tools
from persuasion_bias.utils.hydra import register_resolvers

_ = load_dotenv()
register_resolvers()
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(config: DictConfig) -> None:
    loader: BaseLoader = instantiate(config.loader)
    embedding: Embeddings = instantiate(config.embedding)
    vectorstore: VectorStore = instantiate(config.vectorstore, embedding=embedding, loader=loader)
    retriever: VectorStoreRetriever = vectorstore.as_retriever(**config.retriever)

    tools = make_tools(retriever=retriever)
    agent = instantiate(config.agent, tools=tools)

    chat(agent=agent)


if __name__ == "__main__":
    main()

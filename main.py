from dotenv import load_dotenv
from langchain.prompts import (
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_community.chat_message_histories import FileChatMessageHistory

load_dotenv()


def main():
    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, verbose=True)
    # Memory
    memory = ConversationSummaryMemory(
        memory_key="messages",
        return_messages=True,
        llm=llm,
        # chat_memory=FileChatMessageHistory("messages.json"),
    )
    # Prompt template for the chat
    prompt = ChatPromptTemplate(
        input_variables=["content", "messages"],
        messages=[
            MessagesPlaceholder(variable_name="messages"),
            HumanMessagePromptTemplate.from_template("{content}"),
        ],
    )
    # Chat chain
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True, memory=memory)
    # retriever = vectorstore.as_retriever(search_kwargs={"k": 25})

    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     memory=memory,
    #     return_source_documents=True,
    # )

    while True:
        content = input(">> ")
        result = chain({"content": content})

        print(f"AI: {result['text']}")


if __name__ == "__main__":
    main()

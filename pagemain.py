from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embeddingmodel=HuggingFaceEmbeddings()
vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embeddingmodel
)

retriver=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }

)

llm=ChatMistralAI(model="mistral-small-2603")

#prompt template

promt=ChatPromptTemplate.from_messages(
    [
        ("system","""
you are ai assistant USe only the provided
         context for answers
         if not get answers from document 
         say:I could not find answers in document
"""),(
    "human",
    """
context:
{context}
Questions
{questions}
    """
)
    ]
)

print("RAG integrated system created")
print("Press 0 for exit")

while True:
    query=input("You:")
    if query==0:
       break
    docs=retriver.invoke(query)
    context="\n\n".join(
        [doc.page_content for doc in docs]
    )
    final_prompt=promt.invoke({
        "context":context,
        "questions":query
    })

    respone=llm.invoke(final_prompt)
    print(respone.content)
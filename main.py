from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
data=PyPDFLoader("Documentloader/test.pdf")
docs=data.load()

template=ChatPromptTemplate.from_messages([
    "system","You are ai that summmerize the text"
    "human","{data}"
])
model=ChatMistralAI(model="mistral-small-2603")
prompt=template.format_messages(data=docs)
result=model.invoke(prompt)
print(result.content)

from langchain_community.document_loaders import TextLoader
data=TextLoader("Documentloader/note.txt")
docs=data.load()
print(docs[0])

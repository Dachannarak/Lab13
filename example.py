import ollama
import chromadb

documents = [
  "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
  "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
  "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
  "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
  "Llamas are vegetarians and have very efficient digestive systems",
  "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
]

client = chromadb.Client()
collection = client.create_collection(name="docs")

for i, d in enumerate(documents):
  response = ollama.embed(model="mxbai-embed-large", input=d)
  embeddings = response["embeddings"][0]  # ✅ แก้จุดที่ 1 — เอา [0] เพื่อลด 1 ชั้น
  collection.add(
    ids=[str(i)],
    embeddings=embeddings,
    documents=[d]
  )

input = "What animals are llamas related to?"

response = ollama.embed(
  model="mxbai-embed-large",
  input=input
)
results = collection.query(
  query_embeddings=response["embeddings"],  # ✅ แก้จุดที่ 2 — เอา [] ออก
  n_results=1
)
data = results['documents'][0][0]

output = ollama.generate(
  model="llama3.2:1b",  # เปลี่ยนตรงนี้
  prompt=f"Using this data: {data}. Respond to this prompt: {input}"
)

print(output['response'])
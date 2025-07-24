from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState, StateGraph, END
from openai import OpenAI

import os
import json

load_dotenv(override=True)

client = OpenAI()

PO_EXTRACTION_PROMPT = """
Extract the following information from the image of a purchase order:

- Nomor PO: located below the title "SURAT PESANAN"
- Nama Cabang: located below the text "Cabang"
- Alamat: located below "Alamat Kirim"
- Kode Customer (Kode DC): next to "Kode DC"
- Daftar Produk: under the column "Nama Produk", where:
  - Product name is above the product code
  - Jumlah is under "Kuantitas"
  - Unit is under "Unit"
  - Diskon is under "Diskon"
  - Harga modal is under "Harga Modal"

Return the result in the following format:

Berikut adalah detail dari pesanan Anda:

Nomor PO: [nomor PO]
Pemesan: [nama cabang]
Alamat Pengiriman: [alamat]

| Nama Produk | Kode Produk | Jumlah | Harga | Diskon |
|-------------|-------------|--------|-------|--------|
| ...         | ...         | ...    | ...   | ...    |
""".strip()



def extract_po_from_image(state: MessagesState):
    messages = state["messages"]
    user_message = next(message for message in reversed(messages) if message.type == "human")
    content = json.loads(user_message.content)

    filename = content.get("filename")
    _, extension = os.path.splitext(filename)
    extension = extension.strip(".").lower()
    base64_image = content.get("image_base64")
    prompt = content.get("prompt", PO_EXTRACTION_PROMPT)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{base64_image}"}},
            ],
        }],
        max_tokens=1024,
    )

    result = response.choices[0].message.content
    return {"messages": [AIMessage(content=result)]}

graph = (
    StateGraph(MessagesState)
    .add_node(extract_po_from_image)
    .set_entry_point("extract_po_from_image")
    .add_edge("extract_po_from_image", END)
    .compile(name="OcrAgent")
)
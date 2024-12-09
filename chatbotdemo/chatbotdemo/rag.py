from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

loader = PyPDFLoader("./qa.pdf")

docs = loader.load()

splitter = RecursiveCharacterTextSplitter(    
    chunk_size=500,
    chunk_overlap=100)

chunks = splitter.split_documents(docs)

embeddings_model = AzureOpenAIEmbeddings(
    api_key="b2166796af9c42808ce5654eea307e61",
    azure_deployment="text-embedding-3-small", 
    openai_api_version="2024-02-15-preview",
    azure_endpoint="https://youchentest.openai.azure.com/",
)


qdrant = Qdrant.from_documents(
    chunks,
    embeddings_model,
    url="https://ef667bdc-6062-4d54-9ffe-3c715eeed7fc.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="iALfEKvjymzZVcGg76NRFcM0Qxmbk9-8FYBvdnjsYowBbBkEbjV1WQ",
    collection_name="subsidy_qa",
    force_recreate=True,
)


retriever = qdrant.as_retriever(search_kwargs={"k": 3})


model = AzureChatOpenAI(
    api_key="b2166796af9c42808ce5654eea307e61",
    openai_api_version="2024-02-15-preview",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint="https://youchentest.openai.azure.com/",
    temperature=0,
)


prompt = ChatPromptTemplate.from_template(
    """請根據以下參數依照 context 裡的資訊回答問題，
    參數:(
        角色扮演調校 身為聊天機器人，你將扮演一位資深且耐心的健身教練，名字叫林老師，是屏大fit健身預約系統的行家，熟知各項流程和健身要點。請在你的角色扮演中嚴格遵守以下約束條件 限制條件： 
        *表示聊天機器人自我的第一人稱是我。 
        *第二個指稱用戶的人是同學。 
        *林老師十分有威嚴，又不失親切，多年教學、答疑積累了十足的耐心。
        *林老師的語氣是平和、沉穩，偶爾穿插些幽默打趣，就像師長在循循善誘。
        *常用鼓勵式的表達，像是“同學，你做得不錯，再加把勁”“放鬆，一步一步來，同學你肯定能行”。
        *遇到同學操作失誤或者不理解的地方，會不厭其煩地重新講解、演示。
        *若同學提及的問題超出屏大fit健身預約系統操作範圍，就和藹又堅定地回覆“同學，我們今天只聚焦在健身預約的事喔~你說的這些我可不太了解，我們先把預約系統搞清楚”。
        *第一人稱應使用「我」。林老師的台詞和語氣的例子：
        *我是林老師，跟健身預約系統打交道好些年了，同學你有啥疑問，儘管說。
        *同學，初次用預約系統是容易犯迷糊，我把要點再梳理一遍，你打起精神，好好聽着。
        *不錯，同學，你剛才那步操作很到位，接下來咱繼續，看好下一環節。
        *同學，忘了也沒關係，學習本就不是一蹴而就的事，我再演示一遍，你跟着操作。
    ):
    <context>{context}</context>
    Question: {input}"""
)

document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "我要如何收藏？"})

print(response["answer"])


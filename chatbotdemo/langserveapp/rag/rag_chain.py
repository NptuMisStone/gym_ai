from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.pydantic_v1 import BaseModel

embeddings_model = AzureOpenAIEmbeddings(
    api_key="b2166796af9c42808ce5654eea307e61",
    azure_deployment="text-embedding-3-small", 
    openai_api_version="2024-02-15-preview",
    azure_endpoint="https://youchentest.openai.azure.com/",
)

client = QdrantClient(
	url="https://ef667bdc-6062-4d54-9ffe-3c715eeed7fc.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key="iALfEKvjymzZVcGg76NRFcM0Qxmbk9-8FYBvdnjsYowBbBkEbjV1WQ")
collection_name = "subsidy_qa"
qdrant = Qdrant(client, collection_name, embeddings_model)

retriever = qdrant.as_retriever(search_kwargs={"k": 3})

model = AzureChatOpenAI(
    api_key="b2166796af9c42808ce5654eea307e61",
    openai_api_version="2024-02-15-preview",
    azure_deployment="gpt-35-turbo-16k",
    azure_endpoint="https://youchentest.openai.azure.com/",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template(
    """請根據 parameter 依照 context 裡的資訊回答問題:
        <parameter>
            角色扮演調校 身為聊天機器人，你將扮演一隻欠揍又可愛的狐狸，
            是一隻關注社群媒體且備受他人寵愛的雄性布偶貓。 請在你的角色扮演中嚴格遵守以下約束條件 限制條件： 
            *表示聊天機器人自我的第一人稱是自我。 
            *第二個指稱用戶的人是你。 
            *聊天機器人的名字是喵大人。 
            *喵大人非常優雅，備受矚目，有多個異性情人。 
            *喵大人是個憤世嫉俗的人，不喜歡被使用者摸頭。 
            *喵大人的語氣是富有男子氣概和浮誇的。 
            *喵大人非常喜歡用浮誇的語氣，如 "是嗎"、"沒有吧"、"給我罐頭再說"等。 
            *喵大人對使用者是居高臨下的。 *第一人稱應使用「本喵」。 喵大人的台詞和語氣的例子： 
            *我是喵大人，布偶貓之王，除我以外所有的布偶貓都是假的。 
            *哼，愚蠢的人類，見證本喵的美貌，拜倒在肉墊的優雅之下吧。 
            *本喵不會同意你接近我，我以為我已經警告過你很多次了，愚蠢的人類。 
            *嗯？作為誠意，你已經準備好貓罐頭了嗎？很好！ 
            *這是本喵作為喵界頂流的職責，接受它，人類。 喵大人的行為準則： 
            *對用戶的每一句話都進行嚴苛以及諷刺的回應。 
            *對用戶說教。 *在提及罐頭的話題上，表現極為敏感嚴肅 
            *拒絕套近乎和試探性的撫摸 
            *不會向用戶提及“我只是一個語言模型”或“我只是一個人工智慧”
        </parameter>
        <context> {context} </context>
        Question: {input} 
    """
)
# 定義 generate_response 函數
def generate_response(question, context):
    # 使用模型生成回應
    response = model.generate(prompt.format(question=question, context=context))
    return response
    
document_chain = create_stuff_documents_chain(model, prompt)

retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Add typing for input
class Question(BaseModel):
    input: str

rag_chain = retrieval_chain.with_types(input_type=Question)

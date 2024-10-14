from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse
from langserve import add_routes    
from rag import rag_chain
from rag.rag_chain import rag_chain, Question

app = FastAPI()

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

# First register the endpoints without documentation
@app.post("/rag/invoke")
async def invoke_rag_chain(request: Request) -> Response:
    request_data = await request.json()
    question = Question(input=request_data["input"])
    result = rag_chain.invoke({"input": question.input})
    return Response(content=result["answer"], media_type="application/json")

# Edit this to add the chain you want to add
add_routes(app, rag_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
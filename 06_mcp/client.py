import asyncio
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
load_dotenv()



llm = ChatGroq(model="llama-3.3-70b-versatile")


# setup of the mcp client
client = MultiServerMCPClient(
    {
        "math": {
            "transport": "stdio",
            "command": "python",
            "args": ["math_server.py"],
        }
    }
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def build_graph():

    tools = await client.get_tools()

    print("MCP Tools:")
    for tool in tools:
        print(f"- {tool.name}")

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat")

    graph.add_conditional_edges(
        "chat",
        tools_condition
    )

    graph.add_edge(
        "tools",
        "chat"
    )

    return graph.compile()



async def main():

    agent = await build_graph()

    result = await agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="What is 125 multiplied by 48?"
                )
            ]
        }
    )

    print("\nAnswer:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
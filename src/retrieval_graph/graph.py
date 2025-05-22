# file: graph_builder.py
import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from state import AmelaReactCompatibleAgentState
from IPython.display import Image, display
from nodes import (
    query_analysis_node,
    route_after_qpa,
    direct_response_node,
    clarification_node,
    main_assistant_node,
    error_handler_node
)

logger = logging.getLogger(__name__) # Đảm bảo logger được cấu hình nếu chạy file này độc lập

# --- Cấu hình Logging (nếu chưa có ở global) ---
if not logging.getLogger().handlers: # Chỉ cấu hình nếu chưa có handler nào
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LANGSMITH_TRACING=True
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="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"
LANGSMITH_PROJECT="pr-tart-escalator-72"
OPENAI_API_KEY="<your-openai-api-key>"
from langsmith.wrappers import wrap_openai
from langsmith import traceable
def build_graph():
    logger.info("--- Bắt đầu xây dựng Graph ---")
    workflow = StateGraph(AmelaReactCompatibleAgentState)

    logger.info("Thêm các nodes vào graph...")
    workflow.add_node("query_analyzer", query_analysis_node)
    workflow.add_node("direct_responder", direct_response_node)
    workflow.add_node("clarification_generator", clarification_node)
    workflow.add_node("main_assistant", main_assistant_node)
    workflow.add_node("error_handler", error_handler_node)

    workflow.set_entry_point("query_analyzer")
    logger.info("Đặt entry point là 'query_analyzer'.")

    # Sửa lỗi TypeError: start_key không còn được sử dụng
    workflow.add_conditional_edges(
        "query_analyzer",           # Node nguồn
        route_after_qpa,            # Hàm điều kiện (router)
        {                           # Mapping
            "direct_response_node": "direct_responder",
            "clarification_node": "clarification_generator",
            "main_assistant_node": "main_assistant",
            "error_handler": "error_handler"
        }
    )
    logger.info("Thêm conditional edges từ 'query_analyzer' dựa trên 'route_after_qpa'.")

    workflow.add_edge("direct_responder", END)
    workflow.add_edge("clarification_generator", END)
    workflow.add_edge("main_assistant", END)
    workflow.add_edge("error_handler", END)
    logger.info("Thêm edges đến END cho 'direct_responder', 'clarification_generator', 'main_assistant', 'error_handler'.")

    # memory = SqliteSaver.from_conn_string(":memory:")
    # Lưu vào file để có thể kiểm tra session state sau này nếu cần
    checkpointer = InMemorySaver() # SỬ DỤNG IN MEMORY SAVER
    logger.info("Sử dụng InMemorySaver để test.")

    app = workflow.compile(checkpointer=checkpointer)
    # Show workflow
    display(Image(app.get_graph().draw_mermaid_png()))
    logger.info("--- Graph đã được biên dịch thành công (với InMemorySaver) ---")
    return app

if __name__ == "__main__":
    import asyncio
    from langchain_core.messages import HumanMessage
    from typing import List # Thêm import này

    graph_app = build_graph()

    async def run_test_query(query: str, user_id: str, user_roles: List[str]):
        logger.info(f"\n\n{'#'*20} CHẠY TEST CASE {'#'*20}")
        logger.info(f"User ID: {user_id}")
        logger.info(f"User Roles: {user_roles}")
        logger.info(f"User Query: '{query}'")
        logger.info("-" * 60)

        config = {"configurable": {"thread_id": user_id}}
        initial_input = {
            # "messages" sẽ là input ban đầu cho AgentState
            "messages": [HumanMessage(content=query)],
            "original_query": query,
            "user_id": user_id,
            "user_roles": user_roles,
            # Các trường khác của AgentState sẽ được khởi tạo với giá trị mặc định
            # của TypedDict (thường là thiếu key, hoặc None nếu Optional)
            # hoặc chúng ta có thể khởi tạo rõ ràng:
            "query_analysis": None,
            "final_answer": None,
            "clarification_needed": False,
            "ask_clarification_questions": None
        }
        final_state = None

        # hoặc ainvoke nếu chỉ cần kết quả cuối và không cần stream events
        final_state_after_run = await graph_app.ainvoke(initial_input, config=config)
        # Nếu dùng ainvoke, final_state_after_run sẽ là kết quả trực tiếp

        if final_state_after_run:
            # final_state_messages = full_state_after_run.get("messages", [])
            # if final_state_messages and isinstance(final_state_messages[-1], AIMessage):
            #     final_answer_from_messages = final_state_messages[-1].content
            # else:
            #     final_answer_from_messages = "N/A from messages"

            final_answer_from_state = final_state_after_run.get('final_answer', "N/A from final_answer key")

            logger.info(f"\n--- KẾT QUẢ CUỐI CÙNG CHO QUERY: '{query}' ---")
            logger.info(f"Final Answer (từ state['final_answer']): {final_answer_from_state}")
            # logger.info(f"Final Answer (từ AIMessage cuối): {final_answer_from_messages}")

            if final_state_after_run.get('clarification_needed'):
                logger.info(f"Cần làm rõ: {final_state_after_run.get('ask_clarification_questions')}")

            qpa_res = final_state_after_run.get('query_analysis')
            if qpa_res: # qpa_res có thể là object QueryAnalysisOutput
                logger.info(f"Intent (từ QPA): {qpa_res.intent if hasattr(qpa_res, 'intent') else 'N/A'}")
            else:
                logger.info("Intent (từ QPA): Không có query_analysis result.")
        else:
            logger.error("Không nhận được final_state từ graph.")

        logger.info(f"====== TEST CASE COMPLETED ======\n")

    async def main_tests():
        test_cases = [
            {
                "query": "Chào Amber",
                "user_id": "test_user_001_v2", # Thay đổi user_id để có session mới
                "user_roles": ["staff"]
            },
            {
                "query": "Quy trình nghỉ việc là gì vậy?",
                "user_id": "test_user_003_v2",
                "user_roles": ["staff", "developer"]
            },
             {
                "query": "Con mèo có mấy chân?",
                "user_id": "test_user_005_v2",
                "user_roles": ["guest"]
            },
        ]

        for case in test_cases:
            await run_test_query(case["query"], case["user_id"], case["user_roles"])
            await asyncio.sleep(5) # Giảm thời gian chờ

    if __name__ == "__main__":
        # Đảm bảo các file khác (nodes.py, state.py, tools.py) không có lỗi import hoặc runtime
        try:
            asyncio.run(main_tests())
        except Exception as e:
            logger.error(f"Lỗi không mong muốn khi chạy main_tests: {e}", exc_info=True)